"""
LSTM signal processor for live inference (v4).

Loads Conv1D+LSTM model with StandardScaler normalization.
Returns P(BULLISH) from a 180-second price sequence.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.strategy_brain.signal_processors.base_processor import (
    BaseSignalProcessor,
    TradingSignal,
    SignalDirection,
)
from ml.lstm_features import extract_lstm_sequence, LSTM_SEQ_LEN
from ml.lstm_model import load_model

# Diagnostic: dump raw tick buffer + sequence + prediction to disk on every
# inference call. Gated by env var so it's off by default.
# Set LSTM_DUMP_TICKS=1 to enable.
_TICK_DUMP_ENABLED = os.environ.get("LSTM_DUMP_TICKS", "") == "1"
_TICK_DUMP_DIR = Path("logs/lstm_tick_dumps")


class LSTMProcessor(BaseSignalProcessor):
    """LSTM-based price direction signal processor with normalization."""

    def __init__(
        self,
        asset: str,
        model_dir: Path | str = Path("models"),
        confidence_threshold: float = 0.60,
        model_suffix: str = "",
    ):
        super().__init__(name="LSTM")
        self.asset = asset.upper()
        self.confidence_threshold = confidence_threshold

        model_path = Path(model_dir) / f"{self.asset}{model_suffix}_lstm.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"LSTM model not found: {model_path}")

        self.model, self.metadata = load_model(str(model_path))
        self.model.eval()

        # Load scaler from model metadata
        self.scaler_mean = None
        self.scaler_std = None
        if "scaler_mean" in self.metadata and "scaler_std" in self.metadata:
            self.scaler_mean = np.array(self.metadata["scaler_mean"], dtype=np.float32)
            self.scaler_std = np.array(self.metadata["scaler_std"], dtype=np.float32)
            self.scaler_std[self.scaler_std == 0] = 1.0

    def process(self, current_price, historical_prices, metadata):
        """Not used directly — strategy calls predict_proba instead."""
        return None

    def predict_proba(self, current_price, historical_prices, metadata) -> Optional[float]:
        """Return raw P(BULLISH) without threshold gating."""
        tick_buffer = metadata.get("raw_tick_buffer") or metadata.get("tick_buffer")
        if not tick_buffer:
            return None

        dm = metadata.get("decision_minute", 0)

        # Determine timestamp from latest tick
        if isinstance(tick_buffer, list) and tick_buffer:
            last_tick = tick_buffer[-1]
            ts = last_tick.get("ts", datetime.now(timezone.utc))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        # Extract sequence
        buf = list(tick_buffer) if not isinstance(tick_buffer, list) else tick_buffer
        window_open_price = metadata.get("window_open_price")
        sequence = extract_lstm_sequence(buf, ts, decision_minute=dm, window_open_price=window_open_price)
        if sequence is None:
            from loguru import logger
            logger.warning(
                "LSTM {}: seq=None, buf_len={}, ts={}", self.asset, len(buf), ts
            )
            return None

        # Apply scaler normalization
        if self.scaler_mean is not None:
            sequence = (sequence - self.scaler_mean) / self.scaler_std
            sequence = np.nan_to_num(sequence, nan=0.0, posinf=1e6, neginf=-1e6)

        # Inference
        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            p_bullish = self.model(x).item()

        # Optional diagnostic dump (gated by LSTM_DUMP_TICKS=1 env var)
        if _TICK_DUMP_ENABLED:
            try:
                self._dump_ticks(buf, ts, dm, window_open_price, sequence, p_bullish)
            except Exception as e:
                from loguru import logger
                logger.debug("LSTM tick dump failed for {}: {}", self.asset, e)

        return p_bullish

    def _dump_ticks(self, buf, ts, dm, window_open_price, sequence, p_bullish):
        """Write the exact input state the LSTM just saw, for offline verification."""
        from datetime import timedelta
        _TICK_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        # Only keep the slice that extract_lstm_sequence uses: last LSTM_SEQ_LEN seconds
        cutoff = ts - timedelta(seconds=LSTM_SEQ_LEN)
        sliced = []
        for tick in buf:
            t = tick["ts"]
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if t < cutoff or t > ts:
                continue
            sliced.append({
                "ts": t.isoformat(),
                "price": float(tick["price"]),
                "qty": float(tick.get("qty", 0)),
                "is_buyer": bool(tick.get("is_buyer", False)),
            })
        rec = {
            "asset": self.asset,
            "dump_ts": datetime.now(timezone.utc).isoformat(),
            "inference_ts": ts.isoformat(),
            "dm": int(dm) if dm is not None else None,
            "window_open_price": float(window_open_price) if window_open_price else None,
            "buffer_len_total": len(buf),
            "buffer_len_sliced": len(sliced),
            "sequence_shape": list(sequence.shape),
            "p_bullish": float(p_bullish),
            "ticks": sliced,
        }
        # One file per asset per day; append JSONL
        date_str = ts.strftime("%Y-%m-%d")
        out = _TICK_DUMP_DIR / f"{self.asset}_{date_str}.jsonl"
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
