"""Multi-asset Kalshi strategy — orchestrates Binance WS feeds, signal
processors, fusion engine, and the Kalshi execution adapter.

Two-tier entry gate:
  - Early zone (dm 2-3): Binance signal + Kalshi price confirmation required
  - Late zone  (dm 4-9): Binance signal alone (89%+ backtested accuracy)
"""
from __future__ import annotations

import asyncio
import csv
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from core.strategy_brain.signal_processors.base_processor import (
    TradingSignal,
    SignalDirection,
)
from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
from core.strategy_brain.signal_processors.deribit_pcr_processor import DeribitPCRProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.signal_processors.kalshi_price_processor import KalshiPriceProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine, FusedSignal

from data_sources.binance.websocket import BinanceWebSocketSource
from execution.kalshi_execution import KalshiExecutionAdapter, TradeRecord
from sdk.kalshi.client import KalshiClient
from sdk.kalshi.account import AccountManager
from sdk.kalshi.markets import fetch_current_market
from sdk.kalshi.orders import fetch_balance
from sdk.kalshi.ticker import series_for_asset

logger = logging.getLogger(__name__)

# Binance symbols for each supported asset
BINANCE_SYMBOLS: dict[str, str] = {
    "BTC": "btcusdt",
    "ETH": "ethusdt",
    "SOL": "solusdt",
    "XRP": "xrpusdt",
}


@dataclass
class AssetState:
    """Per-asset runtime state — each asset gets independent processor instances."""
    asset: str
    series: str
    binance_symbol: str

    price_history: deque = field(default_factory=lambda: deque(maxlen=200))
    tick_buffer: deque = field(default_factory=lambda: deque(maxlen=300))
    current_price: Optional[Decimal] = None

    current_window_id: Optional[str] = None
    traded_windows: set = field(default_factory=set)
    hedged_windows: set = field(default_factory=set)  # windows where we hedged
    active_trade: Optional[Any] = None  # TradeRecord for current window (for flip detection)
    pending_settlements: list = field(default_factory=list)

    # Per-asset processor instances (no cross-contamination)
    spike: SpikeDetectionProcessor = field(default=None)
    velocity: TickVelocityProcessor = field(default=None)
    sentiment: SentimentProcessor = field(default=None)
    deribit_pcr: DeribitPCRProcessor = field(default=None)
    divergence: PriceDivergenceProcessor = field(default=None)
    kalshi_price: KalshiPriceProcessor = field(default=None)

    ws_source: Optional[BinanceWebSocketSource] = None

    def __post_init__(self):
        # Backtested optimal parameters (sweep-validated at 89%+ accuracy)
        self.spike = SpikeDetectionProcessor(
            spike_threshold=0.003,
            velocity_threshold=0.0015,
            lookback_periods=20,
            min_confidence=0.55,
        )
        self.velocity = TickVelocityProcessor(
            velocity_threshold_60s=0.001,
            velocity_threshold_30s=0.0007,
            min_ticks=5,
            min_confidence=0.55,
        )
        self.sentiment = SentimentProcessor()
        self.deribit_pcr = DeribitPCRProcessor()
        self.divergence = PriceDivergenceProcessor()
        # Divergence disabled — calibrated for Polymarket probabilities
        self.divergence.disable()
        self.kalshi_price = KalshiPriceProcessor(
            price_threshold=65,
            min_confidence=0.55,
        )
        self.ws_source = BinanceWebSocketSource(symbol=self.binance_symbol)


class KalshiMultiAssetStrategy:
    """Orchestrates multi-asset 15-minute trading on Kalshi.

    Architecture:
      Binance WS (per asset) → Price Buffer → Signal Processors → Fusion
        → KalshiExecutionAdapter → Settlement Polling → AccountManager
    """

    TRADE_LOG_PATH = Path("output/trades.csv")
    TRADE_LOG_FIELDS = [
        "timestamp", "asset", "window_id", "market_ticker", "event_ticker",
        "direction", "side", "price_cents", "contracts", "cost",
        "dm", "mtc", "confidence", "score",
        "outcome", "pnl", "balance_after",
    ]

    BALANCE_LOG_PATH = Path("output/balance.csv")
    BALANCE_LOG_FIELDS = ["timestamp", "event", "asset", "balance", "pnl", "kalshi_balance"]

    # Two-tier entry gate:
    #   Early zone (dm 2-3): Binance signal + Kalshi price >= 65c confirmation
    #   Late zone  (dm 4-9): Binance signal alone
    #
    # dm 2 = window_start + 7min (elapsed 420s), mtc=8
    # dm 3 = window_start + 8min (elapsed 480s), mtc=7
    # dm 4 = window_start + 9min (elapsed 540s), mtc=6
    # dm 9 = window_start + 14min (elapsed 840s), mtc=1
    ENTRY_GATE_START_S = 420   # dm=2: 7 min into window (mtc=8)
    EARLY_GATE_END_S = 540     # dm=4: early zone ends, late zone begins
    ENTRY_GATE_END_S = 840     # dm=9: 14 min into window (mtc=1)
    EARLY_KALSHI_THRESHOLD = 65  # cents — Kalshi side must be >= this for early entry

    def __init__(
        self,
        client: KalshiClient,
        account_manager: AccountManager,
        execution_adapter: KalshiExecutionAdapter,
        assets: list[str],
        dry_run: bool = True,
        settlement_delay_seconds: int = 60,
    ):
        self.client = client
        self.account_manager = account_manager
        self.execution = execution_adapter
        self.assets = [a.upper() for a in assets]
        self.dry_run = dry_run
        self.settlement_delay_seconds = settlement_delay_seconds

        self.fusion_engine = SignalFusionEngine()
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Build per-asset state
        self.states: dict[str, AssetState] = {}
        for asset in self.assets:
            symbol = BINANCE_SYMBOLS.get(asset)
            if not symbol:
                raise ValueError(f"No Binance symbol mapped for asset '{asset}'")
            try:
                series = series_for_asset(asset)
            except KeyError as e:
                raise ValueError(str(e)) from e
            self.states[asset] = AssetState(
                asset=asset, series=series, binance_symbol=symbol,
            )

        logger.info(
            "KalshiMultiAssetStrategy initialized: assets=%s dry_run=%s",
            self.assets, self.dry_run,
        )

    # -- lifecycle -------------------------------------------------------------

    async def start(self):
        """Launch all concurrent loops."""
        self._running = True
        logger.info("Starting Kalshi multi-asset strategy...")

        # Log startup balances
        self._log_balance("startup")

        # 1. Binance WS streams (one per asset)
        for asset, state in self.states.items():
            task = asyncio.create_task(
                self._binance_stream(asset, state),
                name=f"binance-ws-{asset}",
            )
            self._tasks.append(task)

        # 2. Window management loop
        self._tasks.append(
            asyncio.create_task(self._window_loop(), name="window-loop"),
        )

        # 3. Settlement polling loop
        self._tasks.append(
            asyncio.create_task(self._settlement_loop(), name="settlement-loop"),
        )

        # 4. Reconciliation loop
        self._tasks.append(
            asyncio.create_task(self._reconciliation_loop(), name="reconcile-loop"),
        )

        logger.info("All loops launched (%d tasks)", len(self._tasks))

    async def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping Kalshi multi-asset strategy...")
        self._running = False

        # Disconnect all WS sources
        for state in self.states.values():
            if state.ws_source:
                await state.ws_source.disconnect()

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("All loops stopped.")

    # -- Binance streaming -----------------------------------------------------

    async def _binance_stream(self, asset: str, state: AssetState):
        """Stream Binance ticker data and populate price_history/tick_buffer."""
        ws = state.ws_source

        async def _on_price(ticker: dict[str, Any]):
            price = ticker["price"]
            ts = ticker["timestamp"]
            state.current_price = price
            state.price_history.append(price)
            state.tick_buffer.append({"ts": ts, "price": float(price)})

        ws.on_price_update = _on_price

        while self._running:
            try:
                logger.info("[ws-%s] Connecting to Binance...", asset)
                await ws.stream_ticker()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[ws-%s] Stream error, reconnecting in 5s", asset)
                await asyncio.sleep(5)

    # -- window management -----------------------------------------------------

    @staticmethod
    def _current_window_boundary() -> datetime:
        """Floor current UTC time to nearest 15-minute boundary."""
        now = datetime.now(timezone.utc)
        minute = (now.minute // 15) * 15
        return now.replace(minute=minute, second=0, microsecond=0)

    @staticmethod
    def _next_window_boundary() -> datetime:
        """Get the start of the next 15-minute window."""
        now = datetime.now(timezone.utc)
        minute = (now.minute // 15) * 15
        boundary = now.replace(minute=minute, second=0, microsecond=0)
        return boundary + timedelta(minutes=15)

    async def _window_loop(self):
        """Main loop: wait for each 15-minute window, process all assets.

        Two-tier gate:
          Early (dm 2-3): Binance signal + Kalshi price confirmation
          Late  (dm 4-9): Binance signal alone (89%+ backtested)
        Hedge-on-flip: if signal reverses after entry, buy opposite side.
        """
        while self._running:
            try:
                boundary = self._current_window_boundary()
                window_id = boundary.strftime("%Y%m%d_%H%M")
                now = datetime.now(timezone.utc)
                elapsed = (now - boundary).total_seconds()

                if self.ENTRY_GATE_START_S <= elapsed <= self.ENTRY_GATE_END_S:
                    dm = int((elapsed - 300) / 60)  # decision_minute equivalent
                    mtc = 10 - dm                    # mins_to_close equivalent
                    early = elapsed < self.EARLY_GATE_END_S  # dm 2-3
                    for asset, state in self.states.items():
                        if window_id in state.traded_windows:
                            # Already traded — check for flip to hedge
                            await self._check_hedge(
                                asset, state, window_id, dm=dm, mtc=mtc,
                            )
                        else:
                            await self._process_asset_window(
                                asset, state, window_id,
                                dm=dm, mtc=mtc, early_zone=early,
                            )
                elif elapsed < self.ENTRY_GATE_START_S:
                    secs_to_gate = self.ENTRY_GATE_START_S - elapsed
                    if secs_to_gate > 30:
                        logger.debug(
                            "[window-loop] Waiting for gate (%.0fs to go)", secs_to_gate,
                        )

                # Sleep until next check (every 10 seconds)
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[window-loop] Error")
                await asyncio.sleep(10)

    async def _process_asset_window(
        self, asset: str, state: AssetState, window_id: str,
        dm: int = 0, mtc: int = 10, early_zone: bool = False,
    ):
        """Process a single asset for the current window.

        Args:
            dm: decision_minute equivalent (2-9)
            mtc: mins_to_close equivalent
            early_zone: True for dm 2-3 (requires Kalshi confirmation)
        """
        # Need enough price history for signals
        if len(state.price_history) < 20:
            logger.debug("[%s] Not enough history (%d)", asset, len(state.price_history))
            return

        if state.current_price is None:
            return

        # Fetch Kalshi market first — needed for both tiers
        series = state.series
        market_info = fetch_current_market(self.client, series)
        if market_info is None:
            logger.warning("[%s] No open Kalshi market for %s", asset, series)
            return

        # Run Binance signal processors (with Kalshi data in metadata)
        signals = self._run_signals(state, kalshi_market=market_info)
        if not signals:
            logger.debug("[%s] No signals generated for window %s", asset, window_id)
            return

        # Fuse signals
        fused = self.fusion_engine.fuse_signals(signals)
        if fused is None or not fused.is_actionable:
            logger.debug(
                "[%s] Fusion not actionable for window %s (fused=%s)",
                asset, window_id, fused,
            )
            return

        # Determine direction string
        direction_str = str(fused.direction).upper()
        if "BULLISH" in direction_str:
            direction = "BULLISH"
        elif "BEARISH" in direction_str:
            direction = "BEARISH"
        else:
            return

        # Early zone (dm 2-3): require Kalshi price confirmation
        if early_zone:
            if not self._kalshi_confirms(direction, market_info):
                logger.info(
                    "[%s] Early zone dm=%d: Binance says %s but Kalshi doesn't confirm "
                    "(yes=%sc no=%sc, threshold=%dc) -- skipping",
                    asset, dm, direction,
                    market_info.get("yes_ask"), market_info.get("no_ask"),
                    self.EARLY_KALSHI_THRESHOLD,
                )
                return

        # Execute trade
        trade = self.execution.execute_trade(
            asset=asset,
            direction=direction,
            confidence=fused.confidence,
            score=fused.score,
            market_info=market_info,
        )

        if trade is not None:
            state.traded_windows.add(window_id)
            state.active_trade = trade
            state.pending_settlements.append(trade)
            state.current_window_id = window_id
            self._log_trade_entry(trade, dm, mtc)
            self._log_balance("trade", asset)
            zone_tag = "EARLY" if early_zone else "LATE"
            logger.info(
                "[%s] %s trade for window %s: dm=%d mtc=%d side=%s @ %dc x%d "
                "(confidence=%.2f score=%.1f)",
                asset, zone_tag, window_id, dm, mtc,
                trade.side, trade.price_cents, trade.count,
                fused.confidence, fused.score,
            )

    def _kalshi_confirms(self, direction: str, market_info: dict[str, Any]) -> bool:
        """Check if Kalshi market pricing confirms the Binance signal direction.

        For early-zone entries (dm 2-3), requires the Kalshi side matching our
        direction to be priced >= EARLY_KALSHI_THRESHOLD.
        """
        threshold = self.EARLY_KALSHI_THRESHOLD
        if direction == "BULLISH":
            yes_ask = market_info.get("yes_ask")
            return yes_ask is not None and yes_ask >= threshold
        elif direction == "BEARISH":
            no_ask = market_info.get("no_ask")
            return no_ask is not None and no_ask >= threshold
        return False

    # -- hedge-on-flip ---------------------------------------------------------

    async def _check_hedge(
        self, asset: str, state: AssetState, window_id: str,
        dm: int = 0, mtc: int = 10,
    ):
        """Check if signal has flipped since our entry. If so, hedge.

        Hedging buys the opposite side to cap the loss. Example:
          Entry:  bought YES @ 40c x5 (cost $2.00)
          Flip:   buy NO @ 35c x5     (cost $1.75)
          Outcome YES wins: YES pays $5.00 - NO loses $1.75 = net +$1.25
          Outcome NO wins:  NO pays $5.00 - YES loses $2.00 = net +$1.25
          Either way we lock in a known result instead of risking full loss.
        """
        # Already hedged this window, or no active trade
        if window_id in state.hedged_windows:
            return
        if state.active_trade is None:
            return

        original = state.active_trade

        # Need price data
        if state.current_price is None or len(state.price_history) < 20:
            return

        # Fetch Kalshi market for current prices
        market_info = fetch_current_market(self.client, state.series)
        if market_info is None:
            return

        # Run signals to see current direction
        signals = self._run_signals(state, kalshi_market=market_info)
        if not signals:
            return

        fused = self.fusion_engine.fuse_signals(signals)
        if fused is None or not fused.is_actionable:
            return

        # Determine current direction
        direction_str = str(fused.direction).upper()
        if "BULLISH" in direction_str:
            current_direction = "BULLISH"
        elif "BEARISH" in direction_str:
            current_direction = "BEARISH"
        else:
            return

        # Check if direction flipped
        if current_direction == original.direction:
            return  # same direction, no flip

        # Direction flipped — hedge by buying the opposite side
        logger.warning(
            "[%s] SIGNAL FLIP detected at dm=%d: entered %s (side=%s @ %dc x%d), "
            "now %s -- hedging",
            asset, dm, original.direction, original.side,
            original.price_cents, original.count,
            current_direction,
        )

        hedge_trade = self.execution.execute_trade(
            asset=asset,
            direction=current_direction,
            confidence=fused.confidence,
            score=fused.score,
            market_info=market_info,
        )

        if hedge_trade is not None:
            state.hedged_windows.add(window_id)
            state.pending_settlements.append(hedge_trade)
            self._log_trade_entry(hedge_trade, dm, mtc)
            self._log_balance("hedge", asset)

            # Calculate locked-in result
            # If we bought YES @ Xc and NO @ Yc, one side pays $1 and the other loses.
            # Net = $1.00 * contracts - entry_cost - hedge_cost (per contract)
            entry_cost = original.cost_dollars + original.fees_dollars
            hedge_cost = hedge_trade.cost_dollars + hedge_trade.fees_dollars
            # One side wins $1/contract; use min contracts for guaranteed lock
            locked_contracts = min(original.count, hedge_trade.count)
            locked_revenue = locked_contracts * 1.00
            locked_pnl = locked_revenue - entry_cost - hedge_cost

            logger.info(
                "[%s] HEDGED window %s: original=%s@%dc x%d ($%.2f), "
                "hedge=%s@%dc x%d ($%.2f) -> locked PnL ~$%.2f",
                asset, window_id,
                original.side, original.price_cents, original.count, entry_cost,
                hedge_trade.side, hedge_trade.price_cents, hedge_trade.count, hedge_cost,
                locked_pnl,
            )

    def _run_signals(
        self, state: AssetState, kalshi_market: dict[str, Any] | None = None,
    ) -> list[TradingSignal]:
        """Run all enabled signal processors for an asset."""
        price = state.current_price
        history = list(state.price_history)
        metadata = self._build_metadata(state)

        # Pass Kalshi market data for the KalshiPriceProcessor
        if kalshi_market is not None:
            metadata["kalshi_market"] = kalshi_market

        signals: list[TradingSignal] = []
        processors = [
            state.spike,
            state.velocity,
            state.sentiment,
            state.deribit_pcr,
            state.divergence,
            state.kalshi_price,
        ]
        for proc in processors:
            if not proc.is_enabled:
                continue
            try:
                sig = proc.process(price, history, metadata)
                if sig is not None:
                    signals.append(sig)
            except Exception:
                logger.exception("[%s] Processor %s error", state.asset, proc.name)

        return signals

    def _build_metadata(self, state: AssetState) -> dict[str, Any]:
        """Build metadata dict consumed by signal processors."""
        meta: dict[str, Any] = {
            "tick_buffer": list(state.tick_buffer),
        }

        # Spot price for divergence processor
        if state.current_price is not None:
            meta["spot_price"] = float(state.current_price)

        # Simple momentum: 5-period ROC
        if len(state.price_history) >= 5:
            recent = list(state.price_history)
            old = float(recent[-5])
            cur = float(recent[-1])
            meta["momentum"] = (cur - old) / old if old else 0.0

        # Sentiment score — placeholder; in production, wire to an actual
        # sentiment feed. Using 50 (neutral) for now.
        meta["sentiment_score"] = 50.0

        return meta

    # -- trade CSV logging -----------------------------------------------------

    def _ensure_trade_log(self):
        """Create the CSV file with headers if it doesn't exist."""
        if self.TRADE_LOG_PATH.exists():
            return
        self.TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.TRADE_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(self.TRADE_LOG_FIELDS)

    def _log_trade_entry(self, trade: TradeRecord, dm: int, mtc: int):
        """Append a row when a trade is placed."""
        self._ensure_trade_log()
        with open(self.TRADE_LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                trade.placed_at.isoformat() if trade.placed_at else "",
                trade.asset,
                trade.window_id,
                trade.market_ticker,
                trade.event_ticker,
                trade.direction,
                trade.side,
                trade.price_cents,
                trade.count,
                f"{trade.cost_dollars + trade.fees_dollars:.4f}",
                dm,
                mtc,
                f"{trade.confidence:.4f}",
                f"{trade.score:.2f}",
                "",  # outcome (filled on settlement)
                "",  # pnl
                "",  # balance_after
            ])

    def _log_settlement(self, trade: TradeRecord):
        """Update the trade's row with outcome and PnL after settlement."""
        if not self.TRADE_LOG_PATH.exists():
            return

        # Read all rows, find the matching trade, update it
        rows = []
        with open(self.TRADE_LOG_PATH, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows.append(header)
            for row in reader:
                # Match by market_ticker + event_ticker (columns 3,4)
                if len(row) > 4 and row[3] == trade.market_ticker and row[4] == trade.event_ticker:
                    won = trade.side == trade.settlement_outcome
                    pnl = trade.revenue_dollars - (trade.cost_dollars + trade.fees_dollars)
                    try:
                        acct = self.account_manager.get_account(trade.series)
                        balance = f"{acct.balance_dollars:.2f}"
                    except KeyError:
                        balance = ""
                    row[14] = trade.settlement_outcome or ""  # outcome
                    row[15] = f"{pnl:+.4f}"                   # pnl
                    row[16] = balance                          # balance_after
                rows.append(row)

        with open(self.TRADE_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerows(rows)

    # -- balance CSV logging ---------------------------------------------------

    def _ensure_balance_log(self):
        """Create the balance CSV with headers if it doesn't exist."""
        if self.BALANCE_LOG_PATH.exists():
            return
        self.BALANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.BALANCE_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(self.BALANCE_LOG_FIELDS)

    def _log_balance(self, event: str, asset: str = ""):
        """Append a balance snapshot row."""
        self._ensure_balance_log()
        now = datetime.now(timezone.utc).isoformat()

        # Fetch real Kalshi balance (best-effort)
        kalshi_bal = ""
        try:
            from sdk.kalshi.orders import fetch_balance
            bal = fetch_balance(self.client)
            if bal is not None:
                kalshi_bal = f"{bal:.2f}"
        except Exception:
            pass

        if asset:
            # Log single asset
            series = series_for_asset(asset)
            try:
                acct = self.account_manager.get_account(series)
                with open(self.BALANCE_LOG_PATH, "a", newline="") as f:
                    csv.writer(f).writerow([
                        now, event, asset,
                        f"{acct.balance_dollars:.2f}",
                        f"{acct.pnl_dollars:.2f}",
                        kalshi_bal,
                    ])
            except KeyError:
                pass
        else:
            # Log all assets
            for a in self.assets:
                series = series_for_asset(a)
                try:
                    acct = self.account_manager.get_account(series)
                    with open(self.BALANCE_LOG_PATH, "a", newline="") as f:
                        csv.writer(f).writerow([
                            now, event, a,
                            f"{acct.balance_dollars:.2f}",
                            f"{acct.pnl_dollars:.2f}",
                            kalshi_bal,
                        ])
                except KeyError:
                    pass

    # -- settlement polling ----------------------------------------------------

    async def _settlement_loop(self):
        """Poll outcomes for pending trades after their windows close."""
        while self._running:
            try:
                for state in self.states.values():
                    settled = []
                    for trade in state.pending_settlements:
                        self.execution.settle_window(trade)
                        if trade.settlement_outcome is not None:
                            self._log_settlement(trade)
                            self._log_balance("settlement", trade.asset)
                            settled.append(trade)

                    for trade in settled:
                        state.pending_settlements.remove(trade)

                    # Clear active trade once all pending settlements resolve
                    if not state.pending_settlements:
                        state.active_trade = None

                await asyncio.sleep(self.settlement_delay_seconds)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[settlement-loop] Error")
                await asyncio.sleep(self.settlement_delay_seconds)

    # -- reconciliation --------------------------------------------------------

    async def _reconciliation_loop(self):
        """Periodically compare sub-accounts against real Kalshi balance."""
        while self._running:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes

                balance = fetch_balance(self.client)
                if balance is not None:
                    result = self.account_manager.reconcile(balance)
                    logger.info(
                        "[reconcile] local=$%.2f actual=$%.2f diff=$%.2f",
                        result["local_total"],
                        result["actual_balance"],
                        result["discrepancy"],
                    )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[reconcile-loop] Error")
                await asyncio.sleep(1800)
