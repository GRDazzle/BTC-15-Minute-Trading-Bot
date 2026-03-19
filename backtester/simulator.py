"""
Backtest simulator.
Replays 15-minute windows minute-by-minute through signal processors + fusion engine.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from loguru import logger

from backtester.data_loader import Window
from core.strategy_brain.signal_processors.base_processor import (
    BaseSignalProcessor,
    SignalDirection,
)
from core.strategy_brain.fusion_engine.signal_fusion import (
    FusedSignal,
    SignalFusionEngine,
)


@dataclass
class WindowResult:
    window_start: datetime
    actual_direction: str        # "BULLISH" / "BEARISH"
    predicted_direction: str     # "BULLISH" / "BEARISH" / "NONE"
    decision_minute: int         # 0-13 for which minute produced signal, -1 = no signal
    confidence: float            # FusedSignal.confidence
    score: float                 # FusedSignal.score
    correct: bool                # predicted == actual
    price_open: float
    price_close: float


class BacktestSimulator:
    def __init__(
        self,
        processors: list[BaseSignalProcessor],
        fusion_engine: SignalFusionEngine,
    ):
        self.processors = processors
        self.fusion_engine = fusion_engine

    def run(
        self,
        windows: list[Window],
        fg_scores: dict[str, int],
    ) -> list[WindowResult]:
        results: list[WindowResult] = []
        # Continuous price_history across windows (like live)
        price_history: deque[Decimal] = deque(maxlen=200)
        tick_buffer: deque[dict] = deque(maxlen=300)

        total = len(windows)
        log_interval = max(1, total // 20)

        for i, window in enumerate(windows):
            if i % log_interval == 0:
                logger.info(f"Processing window {i+1}/{total} ({window.start_time})")

            result = self._run_window(window, fg_scores, price_history, tick_buffer)
            results.append(result)

        return results

    def _run_window(
        self,
        window: Window,
        fg_scores: dict[str, int],
        price_history: deque[Decimal],
        tick_buffer: deque[dict],
    ) -> WindowResult:
        # If price_history is empty (first window), seed from klines_before
        if len(price_history) == 0:
            for k in window.klines_before:
                price_history.append(k.close)
                tick_buffer.append({
                    "ts": k.timestamp.replace(tzinfo=timezone.utc)
                    if k.timestamp.tzinfo is None else k.timestamp,
                    "price": float(k.close),
                })

        fg_score = fg_scores.get(
            window.start_time.strftime("%Y-%m-%d"), 50
        )

        for minute_idx, kline in enumerate(window.klines_during):
            # Append to continuous buffers
            price_history.append(kline.close)
            ts = (
                kline.timestamp.replace(tzinfo=timezone.utc)
                if kline.timestamp.tzinfo is None
                else kline.timestamp
            )
            tick_buffer.append({"ts": ts, "price": float(kline.close)})

            # Compute 5-period ROC for momentum
            if len(price_history) >= 6:
                prev = float(price_history[-6])
                curr = float(kline.close)
                momentum = (curr - prev) / prev if prev != 0 else 0.0
            else:
                momentum = 0.0

            metadata = {
                "tick_buffer": list(tick_buffer),
                "spot_price": float(kline.close),
                "momentum": momentum,
                "sentiment_score": fg_score,
            }

            # Run all processors
            signals = []
            for p in self.processors:
                try:
                    sig = p.process(
                        kline.close,
                        list(price_history),
                        metadata,
                    )
                    if sig is not None:
                        signals.append(sig)
                except Exception as e:
                    logger.debug(f"Processor {p.name} error: {e}")

            # Fuse signals
            if signals:
                fused = self.fusion_engine.fuse_signals(signals)
                if fused and fused.is_actionable:
                    direction_str = str(fused.direction).upper()
                    if "BULLISH" in direction_str:
                        predicted = "BULLISH"
                    elif "BEARISH" in direction_str:
                        predicted = "BEARISH"
                    else:
                        predicted = "NONE"

                    correct = predicted == window.actual_direction

                    return WindowResult(
                        window_start=window.start_time,
                        actual_direction=window.actual_direction,
                        predicted_direction=predicted,
                        decision_minute=minute_idx,
                        confidence=fused.confidence,
                        score=fused.score,
                        correct=correct,
                        price_open=float(window.price_open),
                        price_close=float(window.price_close),
                    )

        # No actionable signal in this window
        return WindowResult(
            window_start=window.start_time,
            actual_direction=window.actual_direction,
            predicted_direction="NONE",
            decision_minute=-1,
            confidence=0.0,
            score=0.0,
            correct=False,
            price_open=float(window.price_open),
            price_close=float(window.price_close),
        )
