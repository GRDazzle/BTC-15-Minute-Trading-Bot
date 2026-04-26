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

# In-process reproducibility test. After each predict_proba, immediately re-run
# the full extract+scale+infer path on a deep-copied snapshot of the SAME input
# and log (p1, p2, seq_hash_1, seq_hash_2). If p1 != p2 or hashes differ,
# something inside the live process is non-deterministic across calls.
# Set LSTM_SELFTEST=1 to enable.
_SELFTEST_ENABLED = os.environ.get("LSTM_SELFTEST", "") == "1"
_selftest_env_logged = False


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

        # SELFTEST: snapshot buffer BEFORE extract, so second pass sees identical input
        # even if extract_lstm_sequence or downstream code were to mutate the list.
        selftest_snapshot = None
        if _SELFTEST_ENABLED:
            import copy
            selftest_snapshot = {
                "buf": copy.deepcopy(buf),
                "ts": ts,
                "dm": dm,
                "wop": window_open_price,
            }

        sequence = extract_lstm_sequence(buf, ts, decision_minute=dm, window_open_price=window_open_price)
        if sequence is None:
            from loguru import logger
            logger.warning(
                "LSTM {}: seq=None, buf_len={}, ts={}", self.asset, len(buf), ts
            )
            return None

        # Capture pre-scale md5 so dump + standalone can compare *before*
        # scaling — narrows bug to extract vs scale vs forward-pass.
        pre_scale_md5 = None
        if _TICK_DUMP_ENABLED:
            import hashlib
            pre_scale_md5 = hashlib.md5(sequence.tobytes()).hexdigest()

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
                self._dump_ticks(buf, ts, dm, window_open_price, sequence, p_bullish,
                                 pre_scale_md5=pre_scale_md5)
            except Exception as e:
                from loguru import logger
                logger.debug("LSTM tick dump failed for {}: {}", self.asset, e)

        # In-process reproducibility check (gated by LSTM_SELFTEST=1 env var)
        if _SELFTEST_ENABLED and selftest_snapshot is not None:
            try:
                self._selftest_second_pass(selftest_snapshot, sequence, p_bullish)
            except Exception as e:
                from loguru import logger
                logger.warning("LSTM SELFTEST failed for {}: {}", self.asset, e)

        return p_bullish

    def extract_sequence(self, current_price, historical_prices, metadata):
        """Extract + normalize the LSTM input sequence without running inference.

        Returns the (LSTM_SEQ_LEN, n_features) numpy array ready for model input,
        or None if the sequence can't be extracted. Used by batched callers
        (e.g. pnl_sweep) to collect sequences across many checkpoints and run
        a single batched forward pass — avoids per-call GPU kernel launch overhead.
        """
        tick_buffer = metadata.get("raw_tick_buffer") or metadata.get("tick_buffer")
        if not tick_buffer:
            return None
        dm = metadata.get("decision_minute", 0)
        if isinstance(tick_buffer, list) and tick_buffer:
            last_tick = tick_buffer[-1]
            ts = last_tick.get("ts", datetime.now(timezone.utc))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)
        buf = list(tick_buffer) if not isinstance(tick_buffer, list) else tick_buffer
        window_open_price = metadata.get("window_open_price")
        sequence = extract_lstm_sequence(
            buf, ts, decision_minute=dm, window_open_price=window_open_price,
        )
        if sequence is None:
            return None
        if self.scaler_mean is not None:
            sequence = (sequence - self.scaler_mean) / self.scaler_std
            sequence = np.nan_to_num(sequence, nan=0.0, posinf=1e6, neginf=-1e6)
        return sequence

    def predict_proba_batch(self, sequences: "np.ndarray", batch_size: int = 4096) -> "np.ndarray":
        """Run LSTM on a stack of pre-extracted (+ normalized) sequences.

        sequences: (N, LSTM_SEQ_LEN, n_features) float32 numpy array
        Returns: (N,) float32 numpy array of P(BULLISH) predictions.
        """
        if len(sequences) == 0:
            return np.empty((0,), dtype=np.float32)
        device = next(self.model.parameters()).device
        preds = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                xb = torch.tensor(sequences[i:i + batch_size], dtype=torch.float32).to(device)
                out = self.model(xb)
                preds.append(out.cpu().numpy().reshape(-1))
        return np.concatenate(preds).astype(np.float32)

    def _selftest_second_pass(self, snapshot, seq1, p1):
        """Re-run full extract+scale+infer path on deep-copied snapshot of input.

        Logs:
          - p1 (first pass) vs p2 (second pass on identical snapshot input)
          - md5 of scaled sequence for both passes
          - process-wide torch state ONCE (device, cudnn flags, torch version)
        Any divergence indicates per-call state mutation inside the live process.
        """
        import hashlib
        from loguru import logger
        global _selftest_env_logged

        if not _selftest_env_logged:
            try:
                model_device = next(self.model.parameters()).device
                cudnn_bench = getattr(torch.backends.cudnn, "benchmark", None)
                cudnn_det = getattr(torch.backends.cudnn, "deterministic", None)
                cudnn_enabled = getattr(torch.backends.cudnn, "enabled", None)
                try:
                    det_algos = torch.are_deterministic_algorithms_enabled()
                except Exception:
                    det_algos = None
                logger.warning(
                    "LSTM SELFTEST env: torch={} cuda_avail={} model_device={} "
                    "cudnn.enabled={} cudnn.benchmark={} cudnn.deterministic={} "
                    "use_deterministic={}",
                    torch.__version__,
                    torch.cuda.is_available(),
                    model_device,
                    cudnn_enabled,
                    cudnn_bench,
                    cudnn_det,
                    det_algos,
                )
            except Exception as e:
                logger.warning("LSTM SELFTEST env log failed: {}", e)
            _selftest_env_logged = True

        buf2 = snapshot["buf"]
        ts2 = snapshot["ts"]
        dm2 = snapshot["dm"]
        wop2 = snapshot["wop"]
        seq2 = extract_lstm_sequence(buf2, ts2, decision_minute=dm2, window_open_price=wop2)
        if seq2 is None:
            logger.warning("LSTM SELFTEST: second-pass seq None (asset={})", self.asset)
            return
        if self.scaler_mean is not None:
            seq2 = (seq2 - self.scaler_mean) / self.scaler_std
            seq2 = np.nan_to_num(seq2, nan=0.0, posinf=1e6, neginf=-1e6)
        with torch.no_grad():
            x2 = torch.tensor(seq2, dtype=torch.float32).unsqueeze(0)
            p2 = self.model(x2).item()

        h1 = hashlib.md5(seq1.tobytes()).hexdigest()[:12]
        h2 = hashlib.md5(seq2.tobytes()).hexdigest()[:12]
        delta = abs(p1 - p2)
        verdict = "MATCH" if delta < 1e-4 and h1 == h2 else "DRIFT"
        logger.warning(
            "LSTM SELFTEST {} asset={} dm={} p1={:.6f} p2={:.6f} delta={:.6f} "
            "seq_h1={} seq_h2={} buf_len={}",
            verdict, self.asset, dm2, p1, p2, delta, h1, h2, len(buf2),
        )

    def _dump_ticks(self, buf, ts, dm, window_open_price, sequence, p_bullish,
                    pre_scale_md5=None):
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
        import hashlib
        seq_md5 = hashlib.md5(sequence.tobytes()).hexdigest()  # post-scale
        rec = {
            "asset": self.asset,
            "dump_ts": datetime.now(timezone.utc).isoformat(),
            "inference_ts": ts.isoformat(),
            "dm": int(dm) if dm is not None else None,
            "window_open_price": float(window_open_price) if window_open_price else None,
            "buffer_len_total": len(buf),
            "buffer_len_sliced": len(sliced),
            "sequence_shape": list(sequence.shape),
            "sequence_md5": seq_md5,              # post-scale
            "pre_scale_md5": pre_scale_md5,       # pre-scale (post-extract, pre-normalize)
            "p_bullish": float(p_bullish),
            "ticks": sliced,
        }
        # One file per asset per day; append JSONL
        date_str = ts.strftime("%Y-%m-%d")
        out = _TICK_DUMP_DIR / f"{self.asset}_{date_str}.jsonl"
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
