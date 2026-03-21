"""
Backtest simulator.
Replays 15-minute windows through signal processors + fusion engine.

Supports two modes:
  1. Minute-by-minute (original): feeds 1 kline/min into processors.
  2. Tick-level: feeds real aggTrade data at ~10s intervals, matching live behavior.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from loguru import logger

from backtester.data_loader import Window
from backtester.data_loader_ticks import TickWindow, Tick, resample_ticks
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
        ml_processor: Optional[object] = None,
        min_dm: int = 0,
        ensemble_weights: Optional[tuple] = None,
    ):
        self.processors = processors
        self.fusion_engine = fusion_engine
        self.ml_processor = ml_processor  # MLProcessor instance (optional)
        self.min_dm = min_dm  # Skip signals before this decision minute
        # ensemble_weights = (ml_weight, ensemble_threshold) or None
        self.ensemble_weights = ensemble_weights

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

    # -- Tick-level simulation -------------------------------------------------

    def run_ticks(
        self,
        windows: list[TickWindow],
        fg_scores: dict[str, int],
        blackout_windows: Optional[list[dict]] = None,
    ) -> list[WindowResult]:
        """Run backtest using tick-level TickWindows.

        Args:
            windows: List of TickWindow with real aggTrade data
            fg_scores: Fear & Greed scores by date
            blackout_windows: Optional list of {"start": time, "end": time}
                to skip windows in blackout periods
        """
        from datetime import time as dt_time

        results: list[WindowResult] = []
        price_history: deque[Decimal] = deque(maxlen=200)
        tick_buffer: deque[dict] = deque(maxlen=300)

        total = len(windows)
        log_interval = max(1, total // 20)
        skipped_blackout = 0

        for i, window in enumerate(windows):
            if i % log_interval == 0:
                logger.info("Processing tick window {}/{} ({})", i + 1, total, window.window_start)

            # Blackout window filtering
            if blackout_windows:
                window_time = window.window_start.time()
                in_blackout = False
                for bw in blackout_windows:
                    if bw["start"] <= window_time < bw["end"]:
                        in_blackout = True
                        break
                if in_blackout:
                    skipped_blackout += 1
                    continue

            result = self._simulate_tick_window(
                window, fg_scores, price_history, tick_buffer,
            )
            results.append(result)

        if skipped_blackout:
            logger.info("Skipped {} windows due to blackout", skipped_blackout)

        return results

    def _simulate_tick_window(
        self,
        window: TickWindow,
        fg_scores: dict[str, int],
        price_history: deque[Decimal],
        tick_buffer: deque[dict],
    ) -> WindowResult:
        """Replay a window using real tick data.

        Key insight: In live mode, Binance WS sends ~1 aggregated ticker/second.
        Raw aggTrades contain hundreds of trades/second for BTC, which would
        flood tick_buffer (maxlen=300) and destroy temporal coverage.

        Solution: Resample ALL ticks to 1-second bars before feeding into buffers.
        This gives tick_buffer ~5 minutes of lookback (300 entries x 1s = 300s),
        matching live behavior exactly.

        Steps:
        1. Feed warmup ticks (1s resampled) into buffers
        2. Pre-resample decision zone ticks to 1s bars
        3. Feed decision bars progressively, checking signals every ~10 seconds

        Volume parity: A parallel raw_tick_buffer maintains real aggTrade data
        (with qty/is_buyer) so ML features match training data exactly.
        """
        # Resample interval: 250ms matches live Binance WS cadence (~3-4 updates/sec).
        # tick_buffer (maxlen=300) covers ~75 seconds at this rate.
        resample_ms = 250

        # Raw tick buffer for ML volume features (parallel to resampled tick_buffer)
        raw_tick_buffer: deque[dict] = deque(maxlen=1200)

        # Step 1: Feed warmup ticks (resampled to 250ms bars)
        if window.ticks_before:
            warmup_bars = resample_ticks(
                window.ticks_before,
                window.ticks_before[0].ts,
                window.ticks_before[-1].ts + timedelta(seconds=1),
                interval_ms=resample_ms,
            )
            for bar in warmup_bars:
                price_history.append(Decimal(str(bar["price"])))
                tick_buffer.append(bar)

            # Populate raw tick buffer from warmup
            for tick in window.ticks_before:
                raw_tick_buffer.append({
                    "ts": tick.ts,
                    "price": tick.price,
                    "qty": tick.qty,
                    "is_buyer": tick.is_buyer,
                })

        # Step 2: Pre-resample decision zone ticks to 250ms bars
        if window.ticks_during:
            decision_bars = resample_ticks(
                window.ticks_during,
                window.ticks_during[0].ts,
                window.ticks_during[-1].ts + timedelta(seconds=1),
                interval_ms=resample_ms,
            )
        else:
            decision_bars = []

        fg_score = fg_scores.get(
            window.window_start.strftime("%Y-%m-%d"), 50
        )

        decision_start = window.window_start + timedelta(minutes=5)
        check_interval = timedelta(seconds=10)
        # First signal check 10s into decision zone (need some data first)
        current_check = decision_start + check_interval
        bar_idx = 0
        raw_tick_idx = 0

        while current_check < window.window_end:
            next_check = current_check + check_interval

            # Feed 1s bars up to current_check into both buffers
            while bar_idx < len(decision_bars) and decision_bars[bar_idx]["ts"] < current_check:
                bar = decision_bars[bar_idx]
                price_history.append(Decimal(str(bar["price"])))
                tick_buffer.append(bar)
                bar_idx += 1

            # Feed raw ticks up to current_check for ML volume features
            while raw_tick_idx < len(window.ticks_during) and window.ticks_during[raw_tick_idx].ts < current_check:
                tick = window.ticks_during[raw_tick_idx]
                raw_tick_buffer.append({
                    "ts": tick.ts,
                    "price": tick.price,
                    "qty": tick.qty,
                    "is_buyer": tick.is_buyer,
                })
                raw_tick_idx += 1

            # Need enough history for processors
            if len(price_history) < 20:
                current_check = next_check
                continue

            current_price = price_history[-1]

            # Compute decision minute (dm)
            elapsed_s = (current_check - window.window_start).total_seconds()
            dm = int((elapsed_s - 300) / 60)

            # Skip early decision minutes (must match training min_dm)
            if dm < self.min_dm:
                current_check = next_check
                continue

            # Compute momentum (5-period ROC)
            if len(price_history) >= 6:
                prev = float(price_history[-6])
                curr = float(current_price)
                momentum = (curr - prev) / prev if prev != 0 else 0.0
            else:
                momentum = 0.0

            metadata = {
                "tick_buffer": list(tick_buffer),
                "raw_tick_buffer": list(raw_tick_buffer),
                "spot_price": float(current_price),
                "momentum": momentum,
                "sentiment_score": fg_score,
                "decision_minute": dm,
                "window_open_price": window.price_open,
            }

            # Ensemble path: blend ML + fusion probabilities
            if self.ensemble_weights is not None and self.ml_processor is not None:
                ml_w, ens_threshold = self.ensemble_weights
                fusion_w = 1.0 - ml_w

                # Get ML raw probability
                ml_p = 0.5
                try:
                    raw_p = self.ml_processor.predict_proba(
                        current_price, list(price_history), metadata,
                    )
                    if raw_p is not None:
                        ml_p = raw_p
                except Exception as e:
                    logger.debug("ML predict_proba error: {}", e)

                # Get fusion probability
                fusion_p = self._get_fusion_probability(
                    current_price, price_history, metadata,
                )

                # Blend
                ensemble_p = ml_w * ml_p + fusion_w * fusion_p

                # Decision
                if ensemble_p >= ens_threshold:
                    predicted = "BULLISH"
                    confidence = ensemble_p
                elif ensemble_p <= 1.0 - ens_threshold:
                    predicted = "BEARISH"
                    confidence = 1.0 - ensemble_p
                else:
                    current_check = next_check
                    continue

                correct = predicted == window.actual_direction
                return WindowResult(
                    window_start=window.window_start,
                    actual_direction=window.actual_direction,
                    predicted_direction=predicted,
                    decision_minute=dm,
                    confidence=confidence,
                    score=confidence * 100,
                    correct=correct,
                    price_open=window.price_open,
                    price_close=window.price_close,
                )

            # ML path: use ML processor as sole decision maker
            elif self.ml_processor is not None:
                try:
                    ml_signal = self.ml_processor.process(
                        current_price, list(price_history), metadata,
                    )
                    if ml_signal is not None:
                        direction_str = str(ml_signal.direction).upper()
                        if "BULLISH" in direction_str:
                            predicted = "BULLISH"
                        elif "BEARISH" in direction_str:
                            predicted = "BEARISH"
                        else:
                            predicted = "NONE"

                        if predicted != "NONE":
                            correct = predicted == window.actual_direction
                            return WindowResult(
                                window_start=window.window_start,
                                actual_direction=window.actual_direction,
                                predicted_direction=predicted,
                                decision_minute=dm,
                                confidence=ml_signal.confidence,
                                score=ml_signal.score,
                                correct=correct,
                                price_open=window.price_open,
                                price_close=window.price_close,
                            )
                except Exception as e:
                    logger.debug("ML processor error: {}", e)
            else:
                # Fusion path (original)
                # Run processors
                signals = []
                for p in self.processors:
                    try:
                        sig = p.process(current_price, list(price_history), metadata)
                        if sig is not None:
                            signals.append(sig)
                    except Exception as e:
                        logger.debug("Processor {} error: {}", p.name, e)

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

                        if predicted != "NONE":
                            correct = predicted == window.actual_direction
                            return WindowResult(
                                window_start=window.window_start,
                                actual_direction=window.actual_direction,
                                predicted_direction=predicted,
                                decision_minute=dm,
                                confidence=fused.confidence,
                                score=fused.score,
                                correct=correct,
                                price_open=window.price_open,
                                price_close=window.price_close,
                            )

            current_check = next_check

        # No actionable signal
        return WindowResult(
            window_start=window.window_start,
            actual_direction=window.actual_direction,
            predicted_direction="NONE",
            decision_minute=-1,
            confidence=0.0,
            score=0.0,
            correct=False,
            price_open=window.price_open,
            price_close=window.price_close,
        )

    # -- Probability collection for ensemble sweep --------------------------------

    def run_ticks_collect_probabilities(
        self,
        windows: list[TickWindow],
        fg_scores: dict[str, int],
    ) -> list[dict]:
        """Run one pass through all windows, collecting ML + fusion probabilities.

        Returns a list of dicts, one per window:
            {
                "window_start": datetime,
                "actual_direction": str,
                "price_open": float,
                "price_close": float,
                "checkpoints": [{"dm": int, "ml_p": float, "fusion_p": float}, ...]
            }

        The sweep then applies weight/threshold combos over pre-computed data
        instead of re-running the full backtest 126 times.
        """
        results: list[dict] = []
        price_history: deque[Decimal] = deque(maxlen=200)
        tick_buffer: deque[dict] = deque(maxlen=300)

        total = len(windows)
        log_interval = max(1, total // 20)

        for i, window in enumerate(windows):
            if i % log_interval == 0:
                logger.info("Collecting probabilities {}/{} ({})", i + 1, total, window.window_start)

            checkpoints = self._collect_window_probabilities(
                window, fg_scores, price_history, tick_buffer,
            )
            results.append({
                "window_start": window.window_start,
                "window_end": window.window_end,
                "actual_direction": window.actual_direction,
                "price_open": window.price_open,
                "price_close": window.price_close,
                "checkpoints": checkpoints,
            })

        return results

    def _collect_window_probabilities(
        self,
        window: TickWindow,
        fg_scores: dict[str, int],
        price_history: deque[Decimal],
        tick_buffer: deque[dict],
    ) -> list[dict]:
        """Replay a window collecting (dm, ml_p, fusion_p) at every checkpoint."""
        resample_ms = 250
        raw_tick_buffer: deque[dict] = deque(maxlen=1200)

        if window.ticks_before:
            warmup_bars = resample_ticks(
                window.ticks_before,
                window.ticks_before[0].ts,
                window.ticks_before[-1].ts + timedelta(seconds=1),
                interval_ms=resample_ms,
            )
            for bar in warmup_bars:
                price_history.append(Decimal(str(bar["price"])))
                tick_buffer.append(bar)
            for tick in window.ticks_before:
                raw_tick_buffer.append({
                    "ts": tick.ts, "price": tick.price,
                    "qty": tick.qty, "is_buyer": tick.is_buyer,
                })

        if window.ticks_during:
            decision_bars = resample_ticks(
                window.ticks_during,
                window.ticks_during[0].ts,
                window.ticks_during[-1].ts + timedelta(seconds=1),
                interval_ms=resample_ms,
            )
        else:
            decision_bars = []

        fg_score = fg_scores.get(window.window_start.strftime("%Y-%m-%d"), 50)

        decision_start = window.window_start + timedelta(minutes=5)
        check_interval = timedelta(seconds=10)
        current_check = decision_start + check_interval
        bar_idx = 0
        raw_tick_idx = 0
        checkpoints = []

        while current_check < window.window_end:
            next_check = current_check + check_interval

            while bar_idx < len(decision_bars) and decision_bars[bar_idx]["ts"] < current_check:
                bar = decision_bars[bar_idx]
                price_history.append(Decimal(str(bar["price"])))
                tick_buffer.append(bar)
                bar_idx += 1

            while raw_tick_idx < len(window.ticks_during) and window.ticks_during[raw_tick_idx].ts < current_check:
                tick = window.ticks_during[raw_tick_idx]
                raw_tick_buffer.append({
                    "ts": tick.ts, "price": tick.price,
                    "qty": tick.qty, "is_buyer": tick.is_buyer,
                })
                raw_tick_idx += 1

            if len(price_history) < 20:
                current_check = next_check
                continue

            current_price = price_history[-1]
            elapsed_s = (current_check - window.window_start).total_seconds()
            dm = int((elapsed_s - 300) / 60)

            if dm < self.min_dm:
                current_check = next_check
                continue

            if len(price_history) >= 6:
                prev = float(price_history[-6])
                curr = float(current_price)
                momentum = (curr - prev) / prev if prev != 0 else 0.0
            else:
                momentum = 0.0

            metadata = {
                "tick_buffer": list(tick_buffer),
                "raw_tick_buffer": list(raw_tick_buffer),
                "spot_price": float(current_price),
                "momentum": momentum,
                "sentiment_score": fg_score,
                "decision_minute": dm,
                "window_open_price": window.price_open,
            }

            # Get ML probability
            ml_p = 0.5
            if self.ml_processor is not None:
                try:
                    raw_p = self.ml_processor.predict_proba(
                        current_price, list(price_history), metadata,
                    )
                    if raw_p is not None:
                        ml_p = raw_p
                except Exception:
                    pass

            # Get fusion probability
            fusion_p = self._get_fusion_probability(
                current_price, price_history, metadata,
            )

            checkpoints.append({"dm": dm, "ml_p": ml_p, "fusion_p": fusion_p, "signal_ts": current_check})
            current_check = next_check

        return checkpoints

    def _get_fusion_probability(
        self,
        current_price: Decimal,
        price_history: deque,
        metadata: dict,
    ) -> float:
        """Run fusion processors and return P(BULLISH) in [0, 1]."""
        signals = []
        for p in self.processors:
            try:
                sig = p.process(current_price, list(price_history), metadata)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.debug("Fusion processor {} error: {}", p.name, e)

        if not signals:
            return 0.5

        fused = self.fusion_engine.fuse_signals(signals)
        if not fused or not fused.is_actionable:
            return 0.5

        direction_str = str(fused.direction).upper()
        if "BULLISH" in direction_str:
            return fused.confidence
        elif "BEARISH" in direction_str:
            return 1.0 - fused.confidence
        return 0.5
