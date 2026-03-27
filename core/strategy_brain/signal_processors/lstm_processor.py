"""
LSTM signal processor for live inference.

Parallel to MLProcessor (XGBoost). Loads a .pt model and returns
P(BULLISH) from a 120-second price sequence.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from core.strategy_brain.signal_processors.base_processor import (
    BaseSignalProcessor,
    TradingSignal,
    SignalDirection,
)
from ml.lstm_features import extract_lstm_sequence, LSTM_SEQ_LEN
from ml.lstm_model import load_model


class LSTMProcessor(BaseSignalProcessor):
    """LSTM-based price direction signal processor."""

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

    def process(self, current_price, historical_prices, metadata) -> Optional[TradingSignal]:
        """Generate signal if LSTM probability exceeds threshold."""
        p = self.predict_proba(current_price, historical_prices, metadata)
        if p is None:
            return None

        if p >= self.confidence_threshold:
            return TradingSignal(
                direction=SignalDirection.BULLISH,
                confidence=p,
                source="LSTM",
                score=p * 100,
            )
        elif p <= 1.0 - self.confidence_threshold:
            return TradingSignal(
                direction=SignalDirection.BEARISH,
                confidence=1.0 - p,
                source="LSTM",
                score=(1.0 - p) * 100,
            )
        return None

    def predict_proba(self, current_price, historical_prices, metadata) -> Optional[float]:
        """Return raw P(BULLISH) without threshold gating.

        Uses raw_tick_buffer from metadata for volume features,
        falls back to tick_buffer for price-only sequences.
        """
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
            return None

        # Inference
        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            p_bullish = self.model(x).item()

        return p_bullish
