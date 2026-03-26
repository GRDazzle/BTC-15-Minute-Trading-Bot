"""
XGBoost ML Signal Processor.

Replaces the fusion engine as the final decision maker. Takes raw tick features
plus rule-based processor (SpikeDetector, TickVelocity) outputs as inputs.

Trains separately per asset -- each asset loads its own model file.
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

import xgboost as xgb

from core.strategy_brain.signal_processors.base_processor import (
    BaseSignalProcessor,
    SignalDirection,
    SignalStrength,
    SignalType,
    TradingSignal,
)
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from ml.features import FEATURE_NAMES, extract_features

logger = logging.getLogger(__name__)


class MLProcessor(BaseSignalProcessor):
    """XGBoost meta-model signal processor.

    Runs TickVelocity internally to get its outputs, then feeds all features
    into the trained XGBoost model for the final BULLISH/BEARISH/skip decision.
    """

    def __init__(
        self,
        asset: str,
        model_dir: Path,
        confidence_threshold: float = 0.60,
        tickvel_proc: Optional[TickVelocityProcessor] = None,
        model_suffix: str = "",
    ):
        suffix_label = f"-{model_suffix.strip('_')}" if model_suffix else ""
        super().__init__(f"ML-{asset}{suffix_label}")
        self.asset = asset
        self.model_suffix = model_suffix
        self.confidence_threshold = confidence_threshold

        model_path = model_dir / f"{asset.upper()}{model_suffix}_xgb.json"
        if not model_path.exists():
            raise FileNotFoundError(f"No model file for {asset}: {model_path}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        logger.info(
            "MLProcessor loaded model for %s from %s (threshold=%.2f)",
            asset, model_path, confidence_threshold,
        )

        # Sub-processor: used to generate meta-features, not for direct trading
        self.tickvel_proc = tickvel_proc or TickVelocityProcessor(
            velocity_threshold_60s=0.001,
            velocity_threshold_30s=0.0007,
            min_ticks=5,
            min_confidence=0.55,
        )

    def process(
        self,
        current_price: Decimal,
        historical_prices: list,
        metadata: Dict[str, Any] = None,
    ) -> Optional[TradingSignal]:
        """Run ML model to produce a trading signal.

        Args:
            current_price: Current price.
            historical_prices: Recent price history.
            metadata: Must contain 'tick_buffer'. Optionally 'decision_minute'.

        Returns:
            TradingSignal if model confidence exceeds threshold, else None.
        """
        if not self.is_enabled or metadata is None:
            return None

        tick_buffer = metadata.get("tick_buffer")
        if not tick_buffer or len(tick_buffer) < 5:
            return None

        if len(historical_prices) < 20:
            return None

        # Use raw_tick_buffer for ML features (has qty/is_buyer for volume parity)
        # Falls back to tick_buffer if raw_tick_buffer not available (live mode)
        feature_tick_buffer = metadata.get("raw_tick_buffer") or tick_buffer

        # Run sub-processors to get their signal outputs
        tickvel_signal = None
        try:
            tickvel_signal = self.tickvel_proc.process(
                current_price, historical_prices, metadata,
            )
        except Exception:
            pass

        # Get timestamp from tick buffer
        last_tick = tick_buffer[-1]
        ts = last_tick.get("ts", datetime.now(timezone.utc))
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        dm = metadata.get("decision_minute", 0)

        # Extract features (with new args for window_open_price and btc_velocity)
        feats = extract_features(
            tick_buffer=feature_tick_buffer,
            price_history=historical_prices,
            current_price=float(current_price),
            timestamp=ts,
            tickvel_signal=tickvel_signal,
            decision_minute=dm,
            window_open_price=metadata.get("window_open_price"),
            btc_velocity_60s=metadata.get("btc_velocity_60s"),
        )

        # Build feature vector in correct order
        X = [[feats.get(name, 0.0) for name in FEATURE_NAMES]]

        # Predict
        proba = self.model.predict_proba(X)[0]
        # proba[0] = P(BEARISH), proba[1] = P(BULLISH)
        p_bullish = proba[1]

        # Decision
        if p_bullish >= self.confidence_threshold:
            direction = SignalDirection.BULLISH
            confidence = p_bullish
        elif p_bullish <= (1.0 - self.confidence_threshold):
            direction = SignalDirection.BEARISH
            confidence = 1.0 - p_bullish
        else:
            # No signal -- model not confident enough
            return None

        # Map confidence to strength
        if confidence >= 0.80:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.70:
            strength = SignalStrength.STRONG
        elif confidence >= 0.60:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        signal = TradingSignal(
            timestamp=datetime.now(timezone.utc),
            source=self.name,
            signal_type=SignalType.MOMENTUM,
            direction=direction,
            strength=strength,
            confidence=confidence,
            current_price=current_price,
            metadata={
                "p_bullish": round(p_bullish, 4),
                "tickvel_signal": tickvel_signal is not None,
                "model": f"{self.asset}{self.model_suffix}_xgb",
                "dm": dm,
            },
        )
        self._record_signal(signal)

        logger.info(
            "[ML-%s] %s signal: P(bull)=%.3f confidence=%.3f dm=%d",
            self.asset,
            direction.value.upper(),
            p_bullish,
            confidence,
            dm,
        )

        return signal

    def predict_proba(
        self,
        current_price: Decimal,
        historical_prices: list,
        metadata: Dict[str, Any] = None,
    ) -> Optional[float]:
        """Return raw P(BULLISH) without threshold gating.

        Same feature extraction as process(), but returns the float probability
        instead of a TradingSignal. Returns None if insufficient data.
        """
        if not self.is_enabled or metadata is None:
            return None

        tick_buffer = metadata.get("tick_buffer")
        if not tick_buffer or len(tick_buffer) < 5:
            return None

        if len(historical_prices) < 20:
            return None

        feature_tick_buffer = metadata.get("raw_tick_buffer") or tick_buffer

        # Run sub-processor for meta-features
        tickvel_signal = None
        try:
            tickvel_signal = self.tickvel_proc.process(
                current_price, historical_prices, metadata,
            )
        except Exception:
            pass

        last_tick = tick_buffer[-1]
        ts = last_tick.get("ts", datetime.now(timezone.utc))
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        dm = metadata.get("decision_minute", 0)

        feats = extract_features(
            tick_buffer=feature_tick_buffer,
            price_history=historical_prices,
            current_price=float(current_price),
            timestamp=ts,
            tickvel_signal=tickvel_signal,
            decision_minute=dm,
            window_open_price=metadata.get("window_open_price"),
            btc_velocity_60s=metadata.get("btc_velocity_60s"),
        )

        X = [[feats.get(name, 0.0) for name in FEATURE_NAMES]]

        proba = self.model.predict_proba(X)[0]
        return float(proba[1])  # P(BULLISH)
