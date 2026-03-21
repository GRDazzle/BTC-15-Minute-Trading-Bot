"""Multi-asset Kalshi strategy — orchestrates Binance WS feeds, signal
processors, fusion engine, and the Kalshi execution adapter.

Runs concurrent loops per asset, detecting 15-minute window boundaries and
executing trades during dm 2-9 (85%+ ensemble accuracy).
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timezone, timedelta
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

try:
    from core.strategy_brain.signal_processors.ml_processor import MLProcessor
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

from data_sources.binance.websocket import BinanceWebSocketSource
from execution.kalshi_execution import KalshiExecutionAdapter, TradeRecord
from sdk.kalshi.client import KalshiClient
from sdk.kalshi.account import AccountManager
from sdk.kalshi.markets import fetch_current_market
from sdk.kalshi.orders import fetch_balance
from sdk.kalshi.ticker import series_for_asset

logger = logging.getLogger(__name__)

# -- Kalshi data collection ---------------------------------------------------

KALSHI_POLLS_DIR = Path("data/kalshi_polls")


class KalshiDataWriter:
    """Append poll/outcome records to 4-hour UTC bucket JSONL files.

    File layout matches TradingBot format:
        data/kalshi_polls/KX{ASSET}15M/YYYY-MM-DD_HHMM_UTC.jsonl
    """

    def write(self, series: str, record: dict) -> None:
        """Append one JSON line to the right bucket file."""
        now = datetime.now(timezone.utc)
        # 4-hour bucket: floor hour to nearest multiple of 4
        bucket_hour = (now.hour // 4) * 4
        bucket_name = now.strftime(f"%Y-%m-%d_{bucket_hour:02d}00_UTC.jsonl")

        out_dir = KALSHI_POLLS_DIR / series
        out_dir.mkdir(parents=True, exist_ok=True)

        path = out_dir / bucket_name
        line = json.dumps(record, default=str)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


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
    pending_settlements: list = field(default_factory=list)

    # Per-asset processor instances (no cross-contamination)
    spike: SpikeDetectionProcessor = field(default=None)
    velocity: TickVelocityProcessor = field(default=None)
    sentiment: SentimentProcessor = field(default=None)
    deribit_pcr: DeribitPCRProcessor = field(default=None)
    divergence: PriceDivergenceProcessor = field(default=None)
    kalshi_price: KalshiPriceProcessor = field(default=None)
    ml_processor: Optional[Any] = None  # MLProcessor (optional, per-asset)
    ensemble_weights: Optional[tuple] = None  # (ml_weight, threshold) from sweep

    # Circuit breaker: stop trading after 3 consecutive losses (reset on param tune)
    loss_count: int = 0
    circuit_open: bool = False

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

    SIGNAL_LOG_PATH = Path("output/signal_log.csv")
    SIGNAL_LOG_FIELDS = [
        "timestamp", "asset", "window_id", "dm",
        "ml_p", "fusion_p", "ensemble_p", "threshold",
        "direction", "kalshi_yes_ask", "kalshi_no_ask",
        "entry_price", "action", "contracts", "spot_price",
    ]

    # Entry gate: dm 2-9 (ensemble accuracy 85%+ from dm 2 onward)
    # dm 2 = window_start + 7min (elapsed 420s), mtc=8
    # dm 9 = window_start + 14min (elapsed 840s), mtc=1
    ENTRY_GATE_START_S = 420   # dm=2: 7 min into window (mtc=8)
    ENTRY_GATE_END_S = 840     # dm=9: 14 min into window (mtc=1)

    CONFIG_PATH = Path("config/trading.json")

    def __init__(
        self,
        client: KalshiClient,
        account_manager: AccountManager,
        execution_adapter: KalshiExecutionAdapter,
        assets: list[str],
        dry_run: bool = True,
        settlement_delay_seconds: int = 60,
        model_dir: Optional[Path] = None,
        ml_confidence_threshold: float = 0.60,
        warmup_seconds: int = 90,
    ):
        self.client = client
        self.account_manager = account_manager
        self.execution = execution_adapter
        self.assets = [a.upper() for a in assets]
        self.dry_run = dry_run
        self.settlement_delay_seconds = settlement_delay_seconds
        self.warmup_seconds = warmup_seconds
        self._started_at: Optional[datetime] = None

        self.fusion_engine = SignalFusionEngine()
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._ml_confidence_threshold = ml_confidence_threshold
        self._csv_lock = threading.Lock()

        # Resolve model directory for ML processors
        if model_dir is None:
            model_dir = Path("models")
        self._model_dir = model_dir

        # Load blackout windows from config
        self.blackout_windows = self._load_blackout_windows()

        # File modification tracking for hot-reload
        self._config_mtime: float = self._get_mtime(self.CONFIG_PATH)
        self._model_mtimes: dict[str, float] = {}
        self._last_reload_window: Optional[str] = None  # dedup reloads per window

        # Load ensemble params from config
        ensemble_config = self._load_ensemble_config()

        # Build per-asset state
        self.states: dict[str, AssetState] = {}
        ml_loaded = []
        ensemble_loaded = []
        for asset in self.assets:
            symbol = BINANCE_SYMBOLS.get(asset)
            if not symbol:
                raise ValueError(f"No Binance symbol mapped for asset '{asset}'")
            try:
                series = series_for_asset(asset)
            except KeyError as e:
                raise ValueError(str(e)) from e
            state = AssetState(
                asset=asset, series=series, binance_symbol=symbol,
            )

            # Try loading ML model for this asset (fallback to fusion if not found)
            if _ML_AVAILABLE:
                try:
                    state.ml_processor = MLProcessor(
                        asset=asset,
                        model_dir=model_dir,
                        confidence_threshold=ml_confidence_threshold,
                        tickvel_proc=state.velocity,
                    )
                    ml_loaded.append(asset)
                    model_path = model_dir / f"{asset}_xgb.json"
                    self._model_mtimes[asset] = self._get_mtime(model_path)
                except FileNotFoundError:
                    logger.info("No ML model for %s, using fusion fallback", asset)

            # Load ensemble weights if configured and ML model available
            if asset in ensemble_config and state.ml_processor is not None:
                ens = ensemble_config[asset]
                state.ensemble_weights = (ens["ml_weight"], ens["threshold"])
                ensemble_loaded.append(asset)
                logger.info(
                    "Ensemble mode for %s: ml_weight=%.2f threshold=%.2f",
                    asset, ens["ml_weight"], ens["threshold"],
                )

            self.states[asset] = state

        if ml_loaded:
            logger.info("ML models loaded for: %s", ml_loaded)
        if ensemble_loaded:
            logger.info("Ensemble mode active for: %s", ensemble_loaded)

        logger.info(
            "KalshiMultiAssetStrategy initialized: assets=%s dry_run=%s blackout_windows=%d ml=%s",
            self.assets, self.dry_run, len(self.blackout_windows), ml_loaded or "none",
        )

    def _load_blackout_windows(self) -> list[dict]:
        """Load blackout windows from config/trading.json.

        Returns list of {"start": time, "end": time, "reason": str}.
        """
        windows = []
        try:
            with open(self.CONFIG_PATH, "r") as f:
                config = json.load(f)
            for bw in config.get("blackout_windows", []):
                h_s, m_s = bw["start_utc"].split(":")
                h_e, m_e = bw["end_utc"].split(":")
                windows.append({
                    "start": dt_time(int(h_s), int(m_s)),
                    "end": dt_time(int(h_e), int(m_e)),
                    "reason": bw.get("reason", ""),
                })
            if windows:
                for w in windows:
                    logger.info(
                        "Blackout window: %s-%s UTC (%s)",
                        w["start"].strftime("%H:%M"),
                        w["end"].strftime("%H:%M"),
                        w["reason"],
                    )
        except FileNotFoundError:
            logger.warning("Config file %s not found, no blackout windows", self.CONFIG_PATH)
        except (KeyError, ValueError) as e:
            logger.warning("Error parsing blackout windows: %s", e)
        return windows

    def _load_ensemble_config(self) -> dict[str, dict]:
        """Load per-asset ensemble params from config/trading.json.

        Returns dict like {"BTC": {"ml_weight": 0.65, "threshold": 0.70}, ...}.
        """
        result = {}
        try:
            with open(self.CONFIG_PATH, "r") as f:
                config = json.load(f)
            for asset, asset_cfg in config.get("assets", {}).items():
                ens = asset_cfg.get("ensemble")
                if ens and "ml_weight" in ens and "threshold" in ens:
                    result[asset] = ens
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Could not load ensemble config: %s", e)
        return result

    # -- hot-reload ---------------------------------------------------------------

    @staticmethod
    def _get_mtime(path: Path) -> float:
        """Get file modification time, or 0.0 if file doesn't exist."""
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    def _check_and_reload(self) -> None:
        """Check if config or model files changed on disk; reload if so."""
        config_changed = False
        models_changed: list[str] = []

        # Check config/trading.json
        new_config_mtime = self._get_mtime(self.CONFIG_PATH)
        if new_config_mtime != self._config_mtime and new_config_mtime > 0:
            config_changed = True

        # Check models/{ASSET}_xgb.json for each asset
        for asset in self.assets:
            model_path = self._model_dir / f"{asset}_xgb.json"
            new_mtime = self._get_mtime(model_path)
            old_mtime = self._model_mtimes.get(asset, 0.0)
            if new_mtime != old_mtime and new_mtime > 0:
                models_changed.append(asset)

        if not config_changed and not models_changed:
            return

        logger.info(
            "[hot-reload] Changes detected: config=%s models=%s",
            config_changed, models_changed or "none",
        )

        # Reload config (ensemble weights + price bands)
        if config_changed:
            self._reload_config()
            self._config_mtime = new_config_mtime

        # Reload changed ML models
        for asset in models_changed:
            self._reload_model(asset)
            model_path = self._model_dir / f"{asset}_xgb.json"
            self._model_mtimes[asset] = self._get_mtime(model_path)

    def _reload_config(self) -> None:
        """Re-read config/trading.json and update ensemble weights + price bands."""
        try:
            with open(self.CONFIG_PATH, "r") as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error("[hot-reload] Failed to read config: %s", e)
            return

        defaults = config.get("defaults", {})
        asset_configs = config.get("assets", {})

        for asset, state in self.states.items():
            asset_cfg = {**defaults, **asset_configs.get(asset, {})}

            # Update ensemble weights
            ens = asset_cfg.get("ensemble")
            if ens and "ml_weight" in ens and "threshold" in ens and state.ml_processor is not None:
                old = state.ensemble_weights
                new = (ens["ml_weight"], ens["threshold"])
                if old != new:
                    state.ensemble_weights = new
                    logger.info(
                        "[hot-reload] %s ensemble: %s -> ml_weight=%.2f threshold=%.2f",
                        asset, old, new[0], new[1],
                    )

            # Update execution adapter price bounds
            max_p = asset_cfg.get("max_price_cents")
            min_p = asset_cfg.get("min_price_cents")
            max_c = asset_cfg.get("max_contracts_per_trade")
            if max_p is not None:
                self.execution._max_price[asset] = max_p
            if min_p is not None:
                self.execution._min_price[asset] = min_p
            if max_c is not None:
                self.execution._max_contracts[asset] = max_c

        # Reload blackout windows
        self.blackout_windows = self._load_blackout_windows()

        logger.info("[hot-reload] Config reloaded successfully")

    def _reload_model(self, asset: str) -> None:
        """Re-load a single asset's XGBoost model from disk."""
        if not _ML_AVAILABLE:
            return

        state = self.states.get(asset)
        if state is None:
            return

        try:
            new_processor = MLProcessor(
                asset=asset,
                model_dir=self._model_dir,
                confidence_threshold=self._ml_confidence_threshold,
                tickvel_proc=state.velocity,
            )
            state.ml_processor = new_processor
            logger.info("[hot-reload] Reloaded ML model for %s", asset)

            # If ensemble config exists but wasn't loaded before (new model), load it
            if state.ensemble_weights is None:
                ensemble_config = self._load_ensemble_config()
                if asset in ensemble_config:
                    ens = ensemble_config[asset]
                    state.ensemble_weights = (ens["ml_weight"], ens["threshold"])
                    logger.info(
                        "[hot-reload] Activated ensemble for %s: ml_weight=%.2f threshold=%.2f",
                        asset, ens["ml_weight"], ens["threshold"],
                    )
        except FileNotFoundError:
            logger.warning("[hot-reload] Model file disappeared for %s", asset)
        except Exception:
            logger.exception("[hot-reload] Failed to reload ML model for %s", asset)

    def _in_blackout(self, utc_time: Optional[dt_time] = None) -> bool:
        """Check if the given UTC time falls within any blackout window."""
        if utc_time is None:
            utc_time = datetime.now(timezone.utc).time()
        for bw in self.blackout_windows:
            if bw["start"] <= utc_time < bw["end"]:
                return True
        return False

    # -- lifecycle -------------------------------------------------------------

    async def start(self):
        """Launch all concurrent loops."""
        self._running = True
        self._started_at = datetime.now(timezone.utc)
        logger.info(
            "Starting Kalshi multi-asset strategy (warmup=%ds)...",
            self.warmup_seconds,
        )

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

        # 5. Kalshi data collection poller
        self._tasks.append(
            asyncio.create_task(self._kalshi_polling_loop(), name="kalshi-poller"),
        )

        # 6. Rolling parameter tuning (every 2 hours)
        self._tasks.append(
            asyncio.create_task(self._param_tuning_loop(), name="param-tune"),
        )

        logger.info("All loops launched (%d tasks)", len(self._tasks))

    # -- Kalshi data collection --------------------------------------------------

    async def _kalshi_polling_loop(self):
        """Poll Kalshi markets every ~5 seconds and write data to JSONL.

        Runs independently of the trading window loop. Collects bid/ask
        snapshots continuously so PnL backtesting has dense price data.
        Also fetches settlement outcomes when event tickers change.
        """
        writer = KalshiDataWriter()
        # Track current event_ticker per asset to detect window rollovers
        current_events: dict[str, str] = {}
        # Avoid re-fetching outcomes we already have
        settled_events: set[str] = set()

        while self._running:
            try:
                for asset, state in self.states.items():
                    try:
                        market_info = fetch_current_market(self.client, state.series)
                    except Exception:
                        logger.debug("[poller-%s] fetch_current_market error", asset, exc_info=True)
                        continue

                    if market_info is None:
                        continue

                    now_iso = datetime.now(timezone.utc).isoformat()

                    # Write poll record
                    poll_record = {
                        "type": "poll",
                        "ts": now_iso,
                        "series": state.series,
                        "event_ticker": market_info.get("event_ticker", ""),
                        "market_ticker": market_info.get("market_ticker", ""),
                        "close_time": market_info.get("close_time", ""),
                        "yes_bid": market_info.get("yes_bid", 0),
                        "yes_ask": market_info.get("yes_ask", 0),
                        "no_bid": market_info.get("no_bid", 0),
                        "no_ask": market_info.get("no_ask", 0),
                        "volume": str(market_info.get("volume", "")),
                        "oi": str(market_info.get("oi", "")),
                        "outcome": "",
                        "mins_to_close": market_info.get("mins_to_close", 0),
                    }
                    writer.write(state.series, poll_record)

                    # Detect event ticker change -> fetch outcome for old event
                    new_event = market_info.get("event_ticker", "")
                    old_event = current_events.get(asset, "")

                    if old_event and new_event != old_event and old_event not in settled_events:
                        # Schedule outcome fetch after a short delay (market needs time to settle)
                        asyncio.create_task(
                            self._fetch_and_write_outcome(
                                writer, state.series, old_event, settled_events,
                            )
                        )

                    if new_event:
                        current_events[asset] = new_event

                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[poller] Error in polling loop")
                await asyncio.sleep(5)

    async def _fetch_and_write_outcome(
        self,
        writer: KalshiDataWriter,
        series: str,
        event_ticker: str,
        settled_events: set[str],
    ):
        """Wait 30s then fetch and write settlement outcome for an event."""
        try:
            await asyncio.sleep(30)
            from sdk.kalshi.markets import fetch_event_outcome
            outcome = fetch_event_outcome(self.client, event_ticker)
            if outcome is not None:
                now_iso = datetime.now(timezone.utc).isoformat()
                outcome_record = {
                    "type": "outcome",
                    "ts": now_iso,
                    "series": series,
                    "event_ticker": event_ticker,
                    "outcome": outcome,
                    "outcome_source": "poller",
                }
                writer.write(series, outcome_record)
                settled_events.add(event_ticker)
                # Cap set size
                if len(settled_events) > 200:
                    # Remove oldest entries (arbitrary, just prevent unbounded growth)
                    to_remove = list(settled_events)[:50]
                    for e in to_remove:
                        settled_events.discard(e)
                logger.debug("[poller] Wrote outcome for %s: %s", event_ticker, outcome)
            else:
                logger.debug("[poller] No outcome yet for %s", event_ticker)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("[poller] Error fetching outcome for %s", event_ticker, exc_info=True)

    # -- rolling parameter tuning ------------------------------------------------

    async def _param_tuning_loop(self):
        """Run param_tune.py every 2 hours to adapt ensemble parameters.

        First run after 2 hours (needs signal data to accumulate).
        On failure, logs warning and continues (non-fatal).
        """
        tune_interval = 7200  # 2 hours
        tune_script = Path(__file__).resolve().parent.parent / "scripts" / "param_tune.py"

        # Wait 2 hours before first run
        logger.info("[param-tune] First run in %ds", tune_interval)
        await asyncio.sleep(tune_interval)

        while self._running:
            try:
                logger.info("[param-tune] Starting rolling parameter tune...")

                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(tune_script),
                    "--hours", "12",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode == 0:
                    output = stdout.decode("utf-8", errors="replace").strip()
                    logger.info("[param-tune] Completed successfully:\n%s", output)

                    # Reset circuit breakers for all assets
                    for a, st in self.states.items():
                        if st.circuit_open:
                            logger.info("[param-tune] Resetting circuit breaker for %s (was %d losses)", a, st.loss_count)
                        st.loss_count = 0
                        st.circuit_open = False
                else:
                    err = stderr.decode("utf-8", errors="replace").strip()
                    out = stdout.decode("utf-8", errors="replace").strip()
                    logger.warning(
                        "[param-tune] Exited with code %d:\nstdout: %s\nstderr: %s",
                        proc.returncode, out, err,
                    )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[param-tune] Error running param tune")

            await asyncio.sleep(tune_interval)

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

        Entry gate: dm 2-9 (85%+ ensemble accuracy).
        At each new window boundary, checks for config/model changes on disk
        and hot-reloads if needed.
        """
        while self._running:
            try:
                boundary = self._current_window_boundary()
                window_id = boundary.strftime("%Y%m%d_%H%M")
                now = datetime.now(timezone.utc)
                elapsed = (now - boundary).total_seconds()

                # Hot-reload: check once per window, before first trade
                if window_id != self._last_reload_window and elapsed < self.ENTRY_GATE_START_S + 10:
                    self._last_reload_window = window_id
                    self._check_and_reload()

                if self.ENTRY_GATE_START_S <= elapsed <= self.ENTRY_GATE_END_S:
                    dm = int((elapsed - 300) / 60)  # decision_minute equivalent
                    mtc = 10 - dm                    # mins_to_close equivalent
                    for asset, state in self.states.items():
                        await self._process_asset_window(
                            asset, state, window_id, dm=dm, mtc=mtc,
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
        dm: int = 0, mtc: int = 10,
    ):
        """Process a single asset for the current window.

        Args:
            dm: decision_minute equivalent (4-9)
            mtc: mins_to_close equivalent
        """
        # Warmup: skip trading until enough tick data has been collected
        if self._started_at is not None:
            elapsed = (datetime.now(timezone.utc) - self._started_at).total_seconds()
            if elapsed < self.warmup_seconds:
                logger.debug(
                    "[%s] Warmup: %.0fs / %ds elapsed, skipping",
                    asset, elapsed, self.warmup_seconds,
                )
                return

        # Circuit breaker: stop trading this asset after 3 losses since last param tune
        if state.circuit_open:
            logger.warning("[%s] Circuit breaker open (losses=%d), skipping", asset, state.loss_count)
            return

        # Blackout window check: skip trading during configured UTC hours
        if self._in_blackout():
            logger.debug("[%s] In blackout window, skipping", asset)
            return

        # Deduplication: don't trade the same window twice
        if window_id in state.traded_windows:
            return

        # Need enough price history for signals
        if len(state.price_history) < 20:
            logger.debug("[%s] Not enough history (%d)", asset, len(state.price_history))
            return

        if state.current_price is None:
            return

        # Fetch Kalshi market
        series = state.series
        market_info = fetch_current_market(self.client, series)
        if market_info is None:
            logger.warning("[%s] No open Kalshi market for %s", asset, series)
            return

        metadata = self._build_metadata(state)
        if market_info is not None:
            metadata["kalshi_market"] = market_info
        metadata["decision_minute"] = dm

        # Kalshi prices for signal logging
        kalshi_yes_ask = market_info.get("yes_ask", 0) or 0
        kalshi_no_ask = market_info.get("no_ask", 0) or 0
        spot_price = float(state.current_price) if state.current_price else 0.0

        # Ensemble path: blend ML + fusion probabilities
        if state.ensemble_weights is not None and state.ml_processor is not None:
            ml_w, ens_threshold = state.ensemble_weights
            fusion_w = 1.0 - ml_w

            # Get ML raw probability
            ml_p = state.ml_processor.predict_proba(
                state.current_price, list(state.price_history), metadata,
            )
            if ml_p is None:
                logger.debug("[%s] Ensemble: ML returned None for window %s", asset, window_id)
                return

            # Get fusion probability
            fusion_p = self._get_fusion_probability(state, metadata)

            # Blend
            ensemble_p = ml_w * ml_p + fusion_w * fusion_p

            # Decision
            if ensemble_p >= ens_threshold:
                direction = "BULLISH"
                confidence = ensemble_p
                entry_price = kalshi_yes_ask
            elif ensemble_p <= 1.0 - ens_threshold:
                direction = "BEARISH"
                confidence = 1.0 - ensemble_p
                entry_price = kalshi_no_ask
            else:
                logger.info(
                    "[%s] Ensemble skip: p=%.3f (ml=%.3f fus=%.3f) below threshold %.2f",
                    asset, ensemble_p, ml_p, fusion_p, ens_threshold,
                )
                self._log_signal(
                    asset, window_id, dm, ml_p, fusion_p, ensemble_p,
                    ens_threshold, "NONE", kalshi_yes_ask, kalshi_no_ask,
                    0, "skip_threshold", 0, spot_price,
                )
                return

            score = confidence * 100
            trade = self.execution.execute_trade(
                asset=asset,
                direction=direction,
                confidence=confidence,
                score=score,
                market_info=market_info,
            )

            # Log signal with outcome of execute_trade
            if trade is not None:
                self._log_signal(
                    asset, window_id, dm, ml_p, fusion_p, ensemble_p,
                    ens_threshold, direction, kalshi_yes_ask, kalshi_no_ask,
                    entry_price, "trade", trade.count, spot_price,
                )
            else:
                # Determine skip reason from price vs Kelly bands
                max_p = self.execution._get_max_price(asset)
                min_p = self.execution._get_min_price(asset)
                if entry_price > max_p or entry_price < min_p:
                    skip_action = "skip_kelly"
                elif entry_price <= 0:
                    skip_action = "skip_no_ask"
                else:
                    skip_action = "skip_funds"
                self._log_signal(
                    asset, window_id, dm, ml_p, fusion_p, ensemble_p,
                    ens_threshold, direction, kalshi_yes_ask, kalshi_no_ask,
                    entry_price, skip_action, 0, spot_price,
                )

        # ML-only path: use ML processor as sole decision maker
        elif state.ml_processor is not None:
            ml_signal = state.ml_processor.process(
                state.current_price, list(state.price_history), metadata,
            )
            if ml_signal is None:
                logger.debug("[%s] ML no signal for window %s", asset, window_id)
                return

            direction_str = str(ml_signal.direction).upper()
            if "BULLISH" in direction_str:
                direction = "BULLISH"
            elif "BEARISH" in direction_str:
                direction = "BEARISH"
            else:
                return

            trade = self.execution.execute_trade(
                asset=asset,
                direction=direction,
                confidence=ml_signal.confidence,
                score=ml_signal.score,
                market_info=market_info,
            )
        else:
            # Fusion fallback path
            # Run signal processors (with Kalshi data in metadata)
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
            state.pending_settlements.append(trade)
            state.current_window_id = window_id
            self._log_trade_entry(trade, dm, mtc)
            self._log_balance("trade", asset)
            logger.info(
                "[%s] Trade for window %s: dm=%d mtc=%d side=%s @ %dc x%d "
                "(confidence=%.2f score=%.1f)",
                asset, window_id, dm, mtc,
                trade.side, trade.price_cents, trade.count,
                trade.confidence, trade.score,
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

    def _get_fusion_probability(self, state: AssetState, metadata: dict) -> float:
        """Run fusion processors and return P(BULLISH) in [0, 1]."""
        price = state.current_price
        history = list(state.price_history)

        signals: list[TradingSignal] = []
        # Exclude KalshiPriceProcessor from ensemble fusion -- it creates
        # circular logic (using Kalshi price to decide whether to buy at
        # that price) and doesn't match the backtest which only uses
        # SpikeDetection + TickVelocity.
        processors = [
            state.spike, state.velocity, state.sentiment,
            state.deribit_pcr, state.divergence,
        ]
        for proc in processors:
            if not proc.is_enabled:
                continue
            try:
                sig = proc.process(price, history, metadata)
                if sig is not None:
                    signals.append(sig)
            except Exception:
                logger.exception("[%s] Fusion processor %s error", state.asset, proc.name)

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
        with self._csv_lock:
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
        with self._csv_lock:
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

    # -- signal CSV logging ----------------------------------------------------

    def _ensure_signal_log(self):
        """Create the signal log CSV with headers if it doesn't exist."""
        if self.SIGNAL_LOG_PATH.exists():
            return
        self.SIGNAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.SIGNAL_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(self.SIGNAL_LOG_FIELDS)

    def _log_signal(
        self,
        asset: str,
        window_id: str,
        dm: int,
        ml_p: float,
        fusion_p: float,
        ensemble_p: float,
        threshold: float,
        direction: str,
        kalshi_yes_ask: int,
        kalshi_no_ask: int,
        entry_price: int,
        action: str,
        contracts: int,
        spot_price: float,
    ):
        """Append one row to the signal log CSV."""
        self._ensure_signal_log()
        now = datetime.now(timezone.utc).isoformat()
        with open(self.SIGNAL_LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                now,
                asset,
                window_id,
                dm,
                f"{ml_p:.4f}",
                f"{fusion_p:.4f}",
                f"{ensemble_p:.4f}",
                f"{threshold:.2f}",
                direction,
                kalshi_yes_ask,
                kalshi_no_ask,
                entry_price,
                action,
                contracts,
                f"{spot_price:.2f}",
            ])

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

                            # Circuit breaker: track losses
                            won = trade.side == trade.settlement_outcome
                            if not won:
                                state.loss_count += 1
                                if state.loss_count >= 3 and not state.circuit_open:
                                    state.circuit_open = True
                                    logger.warning(
                                        "[%s] Circuit breaker TRIPPED: %d losses since last param tune",
                                        trade.asset, state.loss_count,
                                    )

                    for trade in settled:
                        state.pending_settlements.remove(trade)

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
