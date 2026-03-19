"""Kalshi Price Signal Processor — uses Kalshi market pricing as a crowd-wisdom signal.

When one side of a Kalshi binary market is priced >= threshold, real money is
betting that side will win. This is an independent signal from Binance price
analysis and reflects aggregate market participant conviction.

Data source: Kalshi market snapshot passed via metadata['kalshi_market'].
"""
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any

from loguru import logger

from core.strategy_brain.signal_processors.base_processor import (
    BaseSignalProcessor,
    TradingSignal,
    SignalType,
    SignalDirection,
    SignalStrength,
)


class KalshiPriceProcessor(BaseSignalProcessor):
    """Generates signals based on Kalshi market price skew.

    If yes_ask >= threshold -> BULLISH (market says UP is likely)
    If no_ask >= threshold  -> BEARISH (market says DOWN is likely)
    Confidence scales with price: higher price = stronger conviction.
    """

    def __init__(
        self,
        price_threshold: int = 65,
        min_confidence: float = 0.55,
    ):
        super().__init__("KalshiPrice")
        self.price_threshold = price_threshold
        self.min_confidence = min_confidence

        logger.info(
            f"Initialized KalshiPrice Processor: "
            f"threshold={price_threshold}c, min_confidence={min_confidence:.0%}"
        )

    def process(
        self,
        current_price: Decimal,
        historical_prices: list,
        metadata: Dict[str, Any] = None,
    ) -> Optional[TradingSignal]:
        if not self.is_enabled or not metadata:
            return None

        market = metadata.get("kalshi_market")
        if not market:
            return None

        yes_ask = market.get("yes_ask")
        no_ask = market.get("no_ask")

        if yes_ask is None and no_ask is None:
            return None

        # Determine which side has stronger pricing
        yes_strong = yes_ask is not None and yes_ask >= self.price_threshold
        no_strong = no_ask is not None and no_ask >= self.price_threshold

        # Both sides above threshold = no clear signal (tight spread near 50/50)
        if yes_strong and no_strong:
            logger.debug(
                f"KalshiPrice: both sides >= {self.price_threshold}c "
                f"(yes={yes_ask}c, no={no_ask}c) -- no signal"
            )
            return None

        if not yes_strong and not no_strong:
            logger.debug(
                f"KalshiPrice: neither side >= {self.price_threshold}c "
                f"(yes={yes_ask}c, no={no_ask}c) -- no signal"
            )
            return None

        # One side is dominant
        if yes_strong:
            direction = SignalDirection.BULLISH
            dominant_price = yes_ask
        else:
            direction = SignalDirection.BEARISH
            dominant_price = no_ask

        # Confidence scales with price: 60c -> 0.55, 70c -> 0.62, 80c -> 0.70, 90c -> 0.78
        confidence = 0.45 + (dominant_price / 100.0) * 0.37
        confidence = min(0.85, max(self.min_confidence, confidence))

        # Strength by price level
        if dominant_price >= 85:
            strength = SignalStrength.VERY_STRONG
        elif dominant_price >= 75:
            strength = SignalStrength.STRONG
        elif dominant_price >= 65:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        if confidence < self.min_confidence:
            return None

        signal = TradingSignal(
            timestamp=datetime.now(),
            source=self.name,
            signal_type=SignalType.MOMENTUM,
            direction=direction,
            strength=strength,
            confidence=confidence,
            current_price=current_price,
            metadata={
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "dominant_side": "yes" if yes_strong else "no",
                "dominant_price": dominant_price,
            },
        )

        self._record_signal(signal)

        logger.info(
            f"KalshiPrice: {direction.value.upper()} signal -- "
            f"yes={yes_ask}c no={no_ask}c dominant={'yes' if yes_strong else 'no'}@{dominant_price}c "
            f"confidence={confidence:.0%}"
        )

        return signal
