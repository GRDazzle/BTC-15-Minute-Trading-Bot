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
    from core.strategy_brain.signal_processors.lstm_processor import LSTMProcessor
    from data_sources.coinbase.websocket import CoinbaseWebSocket
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


AGGTRADES_DIR = Path("data/aggtrades_coinbase")


class BinanceTradeWriter:
    """Write live aggTrade data to daily CSVs (same format as Data Vision).

    File layout: data/aggtrades/{ASSET}/{SYMBOL}-aggTrades-{YYYY-MM-DD}.csv
    Columns (no header): agg_trade_id, price, qty, first_id, last_id,
                          timestamp_us, is_buyer_maker, best_price_match
    Timestamps stored in microseconds to match Data Vision format.
    """

    def __init__(self, data_dir: Path = AGGTRADES_DIR):
        self._data_dir = data_dir
        # Open file handles: "ASSET/YYYY-MM-DD" -> (file_handle, csv_writer)
        self._files: dict[str, tuple] = {}
        self._current_date: str = ""

    def write(self, asset: str, symbol: str, trade: dict[str, Any]) -> None:
        """Append one aggTrade to the appropriate daily CSV.

        Args:
            asset: Asset name (e.g. "BTC")
            symbol: Binance symbol (e.g. "btcusdt")
            trade: Trade dict from WS with raw fields (agg_trade_id, etc.)
        """
        ts_ms = trade.get("timestamp_ms")
        if ts_ms is None:
            return  # Missing raw fields, skip

        trade_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        date_str = trade_dt.strftime("%Y-%m-%d")

        # Close old handles on day rollover
        if date_str != self._current_date:
            self._close_all()
            self._current_date = date_str

        key = f"{asset}/{date_str}"
        if key not in self._files:
            asset_dir = self._data_dir / asset.upper()
            asset_dir.mkdir(parents=True, exist_ok=True)
            csv_path = asset_dir / f"{symbol.upper()}-aggTrades-{date_str}.csv"
            fh = open(csv_path, "a", newline="", encoding="utf-8")
            self._files[key] = (fh, csv.writer(fh))

        _, writer = self._files[key]
        ts_us = ts_ms * 1000  # ms -> us
        writer.writerow([
            trade["agg_trade_id"],
            trade["price"],
            trade["quantity"],
            trade["first_id"],
            trade["last_id"],
            ts_us,
            trade["is_buyer_maker"],
            trade.get("best_price_match", True),
        ])
        # Flush after each write; volume is low (~30 trades/hr on Binance.US)
        self._files[key][0].flush()

    def _close_all(self) -> None:
        for fh, _ in self._files.values():
            try:
                fh.close()
            except Exception:
                pass
        self._files.clear()

    def close(self) -> None:
        """Close all open file handles."""
        self._close_all()


# Binance symbols for each supported asset
BINANCE_SYMBOLS: dict[str, str] = {
    "BTC": "btcusdt",
    "ETH": "ethusdt",
    "SOL": "solusdt",
    "XRP": "xrpusdt",
    "HYPE": "hypeusdt",
    "BNB": "bnbusdt",
    "DOGE": "dogeusdt",
}


@dataclass
class AssetState:
    """Per-asset runtime state — each asset gets independent processor instances."""
    asset: str
    series: str
    binance_symbol: str

    price_history: deque = field(default_factory=lambda: deque(maxlen=200))
    tick_buffer: deque = field(default_factory=lambda: deque(maxlen=300))
    raw_tick_buffer: deque = field(default_factory=lambda: deque(maxlen=5000))
    current_price: Optional[Decimal] = None

    current_window_id: Optional[str] = None
    window_open_price: Optional[float] = None  # Price at minute 0 of current window
    traded_windows: set = field(default_factory=set)
    pending_settlements: list = field(default_factory=list)
    pending_verifications: list = field(default_factory=list)  # live trades awaiting Kalshi verification
    min_dm: int = 2  # Per-asset min decision minute (from walk-forward sweep)

    # Live Kalshi prices (updated by WebSocket or REST poller)
    kalshi_market_ticker: Optional[str] = None
    kalshi_event_ticker: Optional[str] = None
    kalshi_yes_ask: int = 0
    kalshi_no_ask: int = 0
    kalshi_yes_bid: int = 0
    kalshi_no_bid: int = 0
    kalshi_close_time: Optional[str] = None
    kalshi_last_update: Optional[datetime] = None

    # Per-asset processor instances (no cross-contamination)
    spike: SpikeDetectionProcessor = field(default=None)
    velocity: TickVelocityProcessor = field(default=None)
    sentiment: SentimentProcessor = field(default=None)
    deribit_pcr: DeribitPCRProcessor = field(default=None)
    divergence: PriceDivergenceProcessor = field(default=None)
    kalshi_price: KalshiPriceProcessor = field(default=None)
    ml_processor: Optional[Any] = None  # MLProcessor (optional, per-asset)
    lstm_processor: Optional[Any] = None  # LSTMProcessor (optional, per-asset)
    ensemble_weights: Optional[tuple] = None  # (ml_weight, threshold) from sweep
    early_ml_processor: Optional[Any] = None  # MLProcessor for dm 2-3 (optional)
    early_ensemble_weights: Optional[tuple] = None  # (ml_weight, threshold) for dm 2-3
    weekday_ml_processor: Optional[Any] = None  # MLProcessor for weekdays
    weekday_lstm_processor: Optional[Any] = None  # LSTMProcessor for weekdays
    weekday_ensemble_weights: Optional[tuple] = None
    weekend_ml_processor: Optional[Any] = None  # MLProcessor for weekends
    weekend_lstm_processor: Optional[Any] = None  # LSTMProcessor for weekends
    weekend_ensemble_weights: Optional[tuple] = None

    # Circuit breaker: pause if asset loses 40% of balance (auto-reset after 1 hour)
    circuit_open: bool = False
    circuit_tripped_at: Optional[datetime] = None
    session_peak_balance: Optional[float] = None

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

    # Live trade log -- richer schema with execution audit trail
    TRADE_LOG_LIVE_PATH = Path("output/trades_live.csv")
    TRADE_LOG_LIVE_FIELDS = [
        "timestamp", "asset", "window_id", "market_ticker", "event_ticker",
        "direction", "side",
        "intended_price_cents", "fill_price_cents",
        "intended_count", "filled_count",
        "cost", "fees",
        "kalshi_order_id",
        "dm", "mtc", "confidence", "score",
        "outcome",
        "expected_revenue", "verified_revenue",
        "settlement_verified", "verified_at",
        "pnl", "balance_after",
    ]

    BALANCE_LOG_PATH = Path("output/balance.csv")
    BALANCE_LOG_LIVE_PATH = Path("output/balance_live.csv")
    BALANCE_LOG_FIELDS = ["timestamp", "event", "asset", "balance", "pnl", "kalshi_balance"]

    SIGNAL_LOG_PATH = Path("output/signal_log.csv")
    SIGNAL_LOG_FIELDS = [
        "timestamp", "asset", "window_id", "dm",
        "ml_p", "fusion_p", "ensemble_p", "threshold",
        "direction", "kalshi_yes_ask", "kalshi_no_ask",
        "entry_price", "action", "contracts", "spot_price",
    ]

    # Entry gate: dm 2-9 (ensemble accuracy 82%+ from dm 2 onward in backtest)
    # dm 2 = window_start + 7min (elapsed 420s), mtc=8
    # dm 9 = window_start + 14min (elapsed 840s), mtc=1
    ENTRY_GATE_START_S = 420   # dm=2: 7 min into window (mtc=8)
    ENTRY_GATE_END_S = 840     # dm=9: 14 min into window (mtc=1)
    BLOCKED_HOURS = set()      # No blocked hours (v8 SMA features handle market open)

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
        tune_delay_seconds: int = 7200,
    ):
        self.client = client
        self.account_manager = account_manager
        self.execution = execution_adapter
        self.assets = [a.upper() for a in assets]
        self.dry_run = dry_run
        self.settlement_delay_seconds = settlement_delay_seconds
        self.warmup_seconds = warmup_seconds
        self._tune_delay_seconds = tune_delay_seconds
        self._started_at: Optional[datetime] = None

        self.fusion_engine = SignalFusionEngine()
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._ml_confidence_threshold = ml_confidence_threshold
        self._csv_lock = threading.Lock()

        # Live aggTrade writer (persists WS data for backtesting/tuning)
        self._trade_writer = BinanceTradeWriter()

        # Runtime stats published to data/runtime_state.json for the manager UI.
        # Updated by the verification and reconciliation loops.
        self._runtime_state: dict = {
            "last_updated": None,
            "reconciliation": {
                "last_run_at": None,
                "kalshi_balance": None,       # last fetched
                "live_local_total": None,
                "drift": None,
                "live_assets": [],
            },
            "verification": {
                "pending_count": 0,
                "stale_count": 0,
                "total_corrections": 0,       # cumulative since bot start
                "total_drift_dollars": 0.0,   # cumulative since bot start
            },
        }

        # SMA cache per asset (recomputed daily)
        self._asset_smas: dict[str, dict] = {}
        self._sma_date: str = ""

        # Consensus confirm: require 2 consecutive checkpoints to agree before entry
        self._require_confirm = False
        # Blocked decision minutes (skip entries at these dms)
        self._blocked_dms: set[int] = set()
        try:
            with open(self.CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            defaults = cfg.get("defaults", {})
            self._require_confirm = defaults.get("require_confirm", False)
            self._blocked_dms = set(defaults.get("blocked_dms", []))
            if self._require_confirm:
                logger.info("Consensus confirm mode ENABLED")
            if self._blocked_dms:
                logger.info("Blocked dms: %s", sorted(self._blocked_dms))
        except Exception:
            pass

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
        early_ensemble_config = self._load_early_ensemble_config()
        weekday_ensemble_config = self._load_day_ensemble_config("ensemble_weekday")
        weekend_ensemble_config = self._load_day_ensemble_config("ensemble_weekend")

        # Build per-asset state
        self.states: dict[str, AssetState] = {}
        ml_loaded = []
        early_ml_loaded = []
        ensemble_loaded = []
        early_ensemble_loaded = []
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

                # Try loading early model (dm 2-3 specialist)
                try:
                    state.early_ml_processor = MLProcessor(
                        asset=asset,
                        model_dir=model_dir,
                        confidence_threshold=ml_confidence_threshold,
                        tickvel_proc=state.velocity,
                        model_suffix="_early",
                    )
                    early_ml_loaded.append(asset)
                    early_model_path = model_dir / f"{asset}_early_xgb.json"
                    self._model_mtimes[f"{asset}_early"] = self._get_mtime(early_model_path)
                except FileNotFoundError:
                    logger.info("No early ML model for %s, standard model covers all dms", asset)

                # Try loading LSTM model
                try:
                    state.lstm_processor = LSTMProcessor(
                        asset=asset,
                        model_dir=model_dir,
                    )
                    lstm_path = model_dir / f"{asset}_lstm.pt"
                    self._model_mtimes[f"{asset}_lstm"] = self._get_mtime(lstm_path)
                    logger.info("LSTM model loaded for %s", asset)
                except FileNotFoundError:
                    logger.info("No LSTM model for %s", asset)
                except Exception as e:
                    logger.warning("Failed to load LSTM for %s: %s", asset, e)

            # Load ensemble weights if configured and ML model available
            if asset in ensemble_config and state.ml_processor is not None:
                ens = ensemble_config[asset]
                state.ensemble_weights = (ens["ml_weight"], ens["threshold"])
                # Per-asset min_dm from walk-forward sweep
                if "min_dm" in ens:
                    state.min_dm = int(ens["min_dm"])
                ensemble_loaded.append(asset)
                logger.info(
                    "Ensemble mode for %s: threshold=%.2f min_dm=%d",
                    asset, ens["threshold"], state.min_dm,
                )
                # Apply ensemble max_price_cents if present
                if "max_price_cents" in ens:
                    execution_adapter._max_price[asset] = ens["max_price_cents"]
                    logger.info(
                        "Ensemble max_price for %s: %dc",
                        asset, ens["max_price_cents"],
                    )

            # Load early ensemble weights if configured and early model available
            if asset in early_ensemble_config and state.early_ml_processor is not None:
                ens = early_ensemble_config[asset]
                state.early_ensemble_weights = (ens["ml_weight"], ens["threshold"])
                early_ensemble_loaded.append(asset)
                logger.info(
                    "Early ensemble mode for %s (dm %d-%d): ml_weight=%.2f threshold=%.2f",
                    asset, ens.get("min_dm", 2), ens.get("max_dm", 3),
                    ens["ml_weight"], ens["threshold"],
                )
                # Apply early ensemble max_price_cents if more restrictive
                if "max_price_cents" in ens:
                    current = execution_adapter._max_price.get(asset, 95)
                    if ens["max_price_cents"] < current:
                        execution_adapter._max_price[asset] = ens["max_price_cents"]
                        logger.info(
                            "Early ensemble max_price for %s: %dc (more restrictive)",
                            asset, ens["max_price_cents"],
                        )

            # Try loading weekday/weekend model variants (fallback to standard)
            if _ML_AVAILABLE:
                for variant in ("weekday", "weekend"):
                    suffix = f"_{variant}"
                    try:
                        ml_proc = MLProcessor(
                            asset=asset, model_dir=model_dir,
                            confidence_threshold=ml_confidence_threshold,
                            tickvel_proc=state.velocity, model_suffix=suffix,
                        )
                        setattr(state, f"{variant}_ml_processor", ml_proc)
                        model_path = model_dir / f"{asset}{suffix}_xgb.json"
                        self._model_mtimes[f"{asset}{suffix}"] = self._get_mtime(model_path)
                        logger.info("%s ML model loaded for %s", variant.capitalize(), asset)
                    except FileNotFoundError:
                        pass

                    try:
                        lstm_proc = LSTMProcessor(
                            asset=asset, model_dir=model_dir, model_suffix=suffix,
                        )
                        setattr(state, f"{variant}_lstm_processor", lstm_proc)
                        lstm_path = model_dir / f"{asset}{suffix}_lstm.pt"
                        self._model_mtimes[f"{asset}{suffix}_lstm"] = self._get_mtime(lstm_path)
                        logger.info("%s LSTM model loaded for %s", variant.capitalize(), asset)
                    except (FileNotFoundError, Exception):
                        pass

                    # Load variant ensemble weights
                    var_config = weekday_ensemble_config if variant == "weekday" else weekend_ensemble_config
                    if asset in var_config and getattr(state, f"{variant}_ml_processor") is not None:
                        ens = var_config[asset]
                        setattr(state, f"{variant}_ensemble_weights", (ens["ml_weight"], ens["threshold"]))
                        logger.info(
                            "%s ensemble for %s: ml_weight=%.2f threshold=%.2f",
                            variant.capitalize(), asset, ens["ml_weight"], ens["threshold"],
                        )

            self.states[asset] = state

        if ml_loaded:
            logger.info("ML models loaded for: %s", ml_loaded)
        if early_ml_loaded:
            logger.info("Early ML models loaded for: %s", early_ml_loaded)
        if ensemble_loaded:
            logger.info("Ensemble mode active for: %s", ensemble_loaded)
        if early_ensemble_loaded:
            logger.info("Early ensemble mode active for: %s", early_ensemble_loaded)

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

    def _load_early_ensemble_config(self) -> dict[str, dict]:
        """Load per-asset early ensemble params from config/trading.json.

        Returns dict like {"BTC": {"ml_weight": 0.5, "threshold": 0.60, "min_dm": 2, "max_dm": 3}, ...}.
        """
        result = {}
        try:
            with open(self.CONFIG_PATH, "r") as f:
                config = json.load(f)
            for asset, asset_cfg in config.get("assets", {}).items():
                ens = asset_cfg.get("ensemble_early")
                if ens and "ml_weight" in ens and "threshold" in ens:
                    result[asset] = ens
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Could not load early ensemble config: %s", e)
        return result

    def _load_day_ensemble_config(self, config_key: str) -> dict[str, dict]:
        """Load per-asset weekday/weekend ensemble params from config/trading.json."""
        result = {}
        try:
            with open(self.CONFIG_PATH, "r") as f:
                config = json.load(f)
            for asset, asset_cfg in config.get("assets", {}).items():
                ens = asset_cfg.get(config_key)
                if ens and "ml_weight" in ens and "threshold" in ens:
                    result[asset] = ens
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Could not load %s config: %s", config_key, e)
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

        # Check models/{ASSET}_xgb.json and {ASSET}_early_xgb.json for each asset
        for asset in self.assets:
            model_path = self._model_dir / f"{asset}_xgb.json"
            new_mtime = self._get_mtime(model_path)
            old_mtime = self._model_mtimes.get(asset, 0.0)
            if new_mtime != old_mtime and new_mtime > 0:
                models_changed.append(asset)

            early_model_path = self._model_dir / f"{asset}_early_xgb.json"
            early_key = f"{asset}_early"
            new_early_mtime = self._get_mtime(early_model_path)
            old_early_mtime = self._model_mtimes.get(early_key, 0.0)
            if new_early_mtime != old_early_mtime and new_early_mtime > 0:
                if early_key not in models_changed:
                    models_changed.append(early_key)

            # Check weekday/weekend model files
            for variant in ("weekday", "weekend"):
                var_key = f"{asset}_{variant}"
                var_path = self._model_dir / f"{asset}_{variant}_xgb.json"
                new_var_mtime = self._get_mtime(var_path)
                old_var_mtime = self._model_mtimes.get(var_key, 0.0)
                if new_var_mtime != old_var_mtime and new_var_mtime > 0:
                    if var_key not in models_changed:
                        models_changed.append(var_key)

            # Check LSTM model files (standard + weekday + weekend)
            for lstm_suffix in ("", "_weekday", "_weekend"):
                lstm_key = f"{asset}{lstm_suffix}_lstm"
                lstm_path = self._model_dir / f"{asset}{lstm_suffix}_lstm.pt"
                new_lstm_mtime = self._get_mtime(lstm_path)
                old_lstm_mtime = self._model_mtimes.get(lstm_key, 0.0)
                if new_lstm_mtime != old_lstm_mtime and new_lstm_mtime > 0:
                    if lstm_key not in models_changed:
                        models_changed.append(lstm_key)

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
        for key in models_changed:
            if key.endswith("_lstm"):
                # LSTM model reload
                self._reload_lstm(key)
                lstm_path = self._model_dir / f"{key}.pt"
                self._model_mtimes[key] = self._get_mtime(lstm_path)
            elif key.endswith("_early"):
                asset = key.replace("_early", "")
                self._reload_early_model(asset)
                early_model_path = self._model_dir / f"{asset}_early_xgb.json"
                self._model_mtimes[key] = self._get_mtime(early_model_path)
            elif key.endswith("_weekday") or key.endswith("_weekend"):
                variant = "weekday" if key.endswith("_weekday") else "weekend"
                asset = key.replace(f"_{variant}", "")
                self._reload_variant_model(asset, variant)
                var_path = self._model_dir / f"{asset}_{variant}_xgb.json"
                self._model_mtimes[key] = self._get_mtime(var_path)
            else:
                self._reload_model(key)
                model_path = self._model_dir / f"{key}_xgb.json"
                self._model_mtimes[key] = self._get_mtime(model_path)

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

            # Update ensemble weights (standard model)
            ens = asset_cfg.get("ensemble")
            if ens and "ml_weight" in ens and "threshold" in ens and state.ml_processor is not None:
                old = state.ensemble_weights
                new = (ens["ml_weight"], ens["threshold"])
                if old != new:
                    state.ensemble_weights = new
                    logger.info(
                        "[hot-reload] %s ensemble: %s -> threshold=%.2f",
                        asset, old, new[1],
                    )
                # Hot-reload min_dm
                new_min_dm = int(ens.get("min_dm", state.min_dm))
                if new_min_dm != state.min_dm:
                    logger.info(
                        "[hot-reload] %s min_dm: %d -> %d",
                        asset, state.min_dm, new_min_dm,
                    )
                    state.min_dm = new_min_dm

            # Update early ensemble weights (dm 2-3)
            ens_early = asset_cfg.get("ensemble_early")
            if ens_early and "ml_weight" in ens_early and "threshold" in ens_early and state.early_ml_processor is not None:
                old = state.early_ensemble_weights
                new = (ens_early["ml_weight"], ens_early["threshold"])
                if old != new:
                    state.early_ensemble_weights = new
                    logger.info(
                        "[hot-reload] %s early ensemble: %s -> ml_weight=%.2f threshold=%.2f",
                        asset, old, new[0], new[1],
                    )

            # Update weekday/weekend ensemble weights
            for variant in ("weekday", "weekend"):
                var_ens = asset_cfg.get(f"ensemble_{variant}")
                ml_attr = f"{variant}_ml_processor"
                weights_attr = f"{variant}_ensemble_weights"
                if var_ens and "ml_weight" in var_ens and "threshold" in var_ens and getattr(state, ml_attr) is not None:
                    old = getattr(state, weights_attr)
                    new = (var_ens["ml_weight"], var_ens["threshold"])
                    if old != new:
                        setattr(state, weights_attr, new)
                        logger.info(
                            "[hot-reload] %s %s ensemble: %s -> ml_weight=%.2f threshold=%.2f",
                            asset, variant, old, new[0], new[1],
                        )

            # Update execution adapter price bounds
            # Ensemble max_price_cents overrides asset-level (more specific)
            max_p = asset_cfg.get("max_price_cents")
            ens_max_p = (ens or {}).get("max_price_cents")
            # Use ensemble max_price if available, else asset-level
            max_p_candidates = [p for p in [ens_max_p, max_p] if p is not None]
            effective_max_p = min(max_p_candidates) if max_p_candidates else None

            min_p = asset_cfg.get("min_price_cents")
            max_c = asset_cfg.get("max_contracts_per_trade")
            if effective_max_p is not None:
                old_max = self.execution._max_price.get(asset)
                if old_max != effective_max_p:
                    logger.info(
                        "[hot-reload] %s max_price: %s -> %dc (source: %s)",
                        asset, old_max, effective_max_p,
                        "ensemble" if ens_max_p else "asset",
                    )
                self.execution._max_price[asset] = effective_max_p
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

    def _reload_early_model(self, asset: str) -> None:
        """Re-load a single asset's early XGBoost model from disk."""
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
                model_suffix="_early",
            )
            state.early_ml_processor = new_processor
            logger.info("[hot-reload] Reloaded early ML model for %s", asset)

            # If early ensemble config exists but wasn't loaded before, load it
            if state.early_ensemble_weights is None:
                early_config = self._load_early_ensemble_config()
                if asset in early_config:
                    ens = early_config[asset]
                    state.early_ensemble_weights = (ens["ml_weight"], ens["threshold"])
                    logger.info(
                        "[hot-reload] Activated early ensemble for %s: ml_weight=%.2f threshold=%.2f",
                        asset, ens["ml_weight"], ens["threshold"],
                    )
        except FileNotFoundError:
            logger.warning("[hot-reload] Early model file disappeared for %s", asset)
            state.early_ml_processor = None
            state.early_ensemble_weights = None
        except Exception:
            logger.exception("[hot-reload] Failed to reload early ML model for %s", asset)

    def _reload_variant_model(self, asset: str, variant: str) -> None:
        """Re-load a weekday/weekend XGBoost model from disk."""
        if not _ML_AVAILABLE:
            return

        state = self.states.get(asset)
        if state is None:
            return

        suffix = f"_{variant}"
        ml_attr = f"{variant}_ml_processor"
        weights_attr = f"{variant}_ensemble_weights"
        config_key = f"ensemble_{variant}"

        try:
            new_processor = MLProcessor(
                asset=asset,
                model_dir=self._model_dir,
                confidence_threshold=self._ml_confidence_threshold,
                tickvel_proc=state.velocity,
                model_suffix=suffix,
            )
            setattr(state, ml_attr, new_processor)
            logger.info("[hot-reload] Reloaded %s ML model for %s", variant, asset)

            if getattr(state, weights_attr) is None:
                var_config = self._load_day_ensemble_config(config_key)
                if asset in var_config:
                    ens = var_config[asset]
                    setattr(state, weights_attr, (ens["ml_weight"], ens["threshold"]))
                    logger.info(
                        "[hot-reload] Activated %s ensemble for %s: ml_weight=%.2f threshold=%.2f",
                        variant, asset, ens["ml_weight"], ens["threshold"],
                    )
        except FileNotFoundError:
            logger.warning("[hot-reload] %s model file disappeared for %s", variant.capitalize(), asset)
            setattr(state, ml_attr, None)
            setattr(state, weights_attr, None)
        except Exception:
            logger.exception("[hot-reload] Failed to reload %s ML model for %s", variant, asset)

    def _reload_lstm(self, key: str) -> None:
        """Re-load an LSTM model from disk. Key format: {ASSET}{suffix}_lstm."""
        # Parse key: e.g. "BTC_lstm", "ETH_weekday_lstm", "SOL_weekend_lstm"
        parts = key.replace("_lstm", "").split("_", 1)
        asset = parts[0]
        suffix = f"_{parts[1]}" if len(parts) > 1 else ""

        state = self.states.get(asset)
        if state is None:
            return

        try:
            new_lstm = LSTMProcessor(
                asset=asset, model_dir=self._model_dir, model_suffix=suffix,
            )
            # Determine which state field to update
            if suffix == "_weekday":
                state.weekday_lstm_processor = new_lstm
            elif suffix == "_weekend":
                state.weekend_lstm_processor = new_lstm
            else:
                state.lstm_processor = new_lstm
            logger.info("[hot-reload] Reloaded LSTM%s model for %s", suffix, asset)
        except FileNotFoundError:
            logger.warning("[hot-reload] LSTM%s model disappeared for %s", suffix, asset)
        except Exception:
            logger.exception("[hot-reload] Failed to reload LSTM%s for %s", suffix, asset)

    def _process_deposits(self) -> None:
        """Apply queued deposits from data/deposits.json to in-memory accounts."""
        deposits_path = Path("data/deposits.json")
        if not deposits_path.exists():
            return
        try:
            with open(deposits_path, "r") as f:
                queue = json.load(f)
            pending = queue.get("pending", [])
            if not pending:
                return
            applied = []
            for dep in pending:
                series = dep.get("series")
                amount = dep.get("amount", 0)
                asset = dep.get("asset", "?")
                try:
                    self.account_manager.deposit(amount, target=series)
                    # Reset circuit breaker peak for this asset (fresh slate)
                    state = self.states.get(asset)
                    if state is not None:
                        state.session_peak_balance = None
                        if state.circuit_open:
                            state.circuit_open = False
                            state.circuit_tripped_at = None
                            logger.info("[deposit] %s circuit breaker reset", asset)
                    logger.info(
                        "[deposit] Credited %s (%s) +$%.2f", asset, series, amount,
                    )
                    self._log_deposit_applied(asset, series, amount)
                    applied.append(dep)
                except Exception as e:
                    logger.error("[deposit] Failed for %s: %s", asset, e)
            # Remove applied deposits and write back
            queue["pending"] = [d for d in pending if d not in applied]
            with open(deposits_path, "w") as f:
                json.dump(queue, f, indent=2)
        except Exception as e:
            logger.warning("[deposit] Failed to process deposits queue: %s", e)

    def _log_deposit_applied(self, asset: str, series: str, amount: float) -> None:
        """Append an applied deposit to data/deposits_log.csv for ledger reconcile."""
        log_path = Path("data/deposits_log.csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not log_path.exists()
        try:
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(["timestamp", "asset", "series", "amount_dollars"])
                writer.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    asset,
                    series,
                    f"{amount:.4f}",
                ])
        except Exception as e:
            logger.warning("[deposit] Failed to log deposit: %s", e)

    def _update_smas(self) -> None:
        """Recompute SMA 5/15/30 for all assets (once per day)."""
        from ml.features import load_daily_closes, compute_daily_smas
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today == self._sma_date:
            return  # Already computed today
        for asset in self.assets:
            daily_closes = load_daily_closes(AGGTRADES_DIR, asset)
            sma5, sma15, sma30 = compute_daily_smas(daily_closes, today)
            self._asset_smas[asset] = {"sma5": sma5, "sma15": sma15, "sma30": sma30}
            logger.info(
                "SMA updated for %s: sma5=%s sma15=%s sma30=%s",
                asset,
                f"{sma5:.2f}" if sma5 else "N/A",
                f"{sma15:.2f}" if sma15 else "N/A",
                f"{sma30:.2f}" if sma30 else "N/A",
            )
        self._sma_date = today

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

        # 1. Coinbase WS streams for dense tick data (one per asset)
        for asset, state in self.states.items():
            task = asyncio.create_task(
                self._coinbase_stream(asset, state),
                name=f"coinbase-ws-{asset}",
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

        # 4b. Settlement verification loop (live trades only -- no-op if all dry-run)
        self._tasks.append(
            asyncio.create_task(self._verification_loop(), name="verification-loop"),
        )

        # 5. Kalshi WebSocket for real-time prices
        self._tasks.append(
            asyncio.create_task(self._kalshi_ws_stream(), name="kalshi-ws"),
        )

        # 6. Kalshi data collection poller (writes JSONL for backtesting)
        self._tasks.append(
            asyncio.create_task(self._kalshi_polling_loop(), name="kalshi-poller"),
        )

        # 6. Rolling parameter tuning -- DISABLED, using weekly retrain only
        # self._tasks.append(
        #     asyncio.create_task(self._param_tuning_loop(), name="param-tune"),
        # )

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

                await asyncio.sleep(2)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[poller] Error in polling loop")
                await asyncio.sleep(2)

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
        """Run pnl_sweep.py every 2 hours to adapt ensemble parameters.

        Single sweep across dm 2-8 using the standard model. This gives more
        trades per combo than splitting into two narrow dm ranges, producing
        more statistically meaningful results for the 12h rolling window.
        Initial delay is configurable via tune_delay_seconds (default 7200s).
        On failure, logs warning and continues (non-fatal).
        """
        tune_interval = 7200  # 2 hours
        tune_script = Path(__file__).resolve().parent.parent / "scripts" / "pnl_sweep.py"
        assets_arg = ",".join(self.assets)

        # Configurable initial delay before first run
        if self._tune_delay_seconds > 0:
            logger.info("[param-tune] First run in %ds", self._tune_delay_seconds)
            await asyncio.sleep(self._tune_delay_seconds)
        else:
            logger.info("[param-tune] Running immediately (tune_delay=0)")

        while self._running:
            try:
                # Single sweep across dm 2-8 (standard model, 12h rolling window)
                logger.info("[param-tune] Starting PnL sweep (dm 2-8, last 12h)...")
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(tune_script),
                    "--asset", assets_arg,
                    "--min-dm", "2", "--max-dm", "8",
                    "--hours", "12",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode == 0:
                    output = stdout.decode("utf-8", errors="replace").strip()
                    logger.info("[param-tune] PnL sweep completed:\n%s", output)
                else:
                    err = stderr.decode("utf-8", errors="replace").strip()
                    out = stdout.decode("utf-8", errors="replace").strip()
                    logger.warning(
                        "[param-tune] PnL sweep exited with code %d:\nstdout: %s\nstderr: %s",
                        proc.returncode, out, err,
                    )

                # Reset circuit breakers for all assets
                for a, st in self.states.items():
                    if st.circuit_open:
                        logger.info("[param-tune] Resetting circuit breaker for %s", a)
                    st.circuit_open = False
                    st.circuit_tripped_at = None
                    st.session_peak_balance = None

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[param-tune] Error running param tune")

            await asyncio.sleep(tune_interval)

    async def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping Kalshi multi-asset strategy...")
        self._running = False

        # Close aggTrade writer
        self._trade_writer.close()

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
        """Stream Binance ticker + aggTrade data via combined stream.

        Ticker updates populate price_history and tick_buffer (existing).
        aggTrade updates populate raw_tick_buffer with qty/is_buyer fields
        needed for volume and aggressor-ratio ML features.
        """
        ws = state.ws_source

        async def _on_price(ticker: dict[str, Any]):
            price = ticker["price"]
            ts = ticker["timestamp"]
            state.current_price = price
            state.price_history.append(price)
            state.tick_buffer.append({"ts": ts, "price": float(price)})

        async def _on_trade(trade: dict[str, Any]):
            state.raw_tick_buffer.append({
                "ts": trade["timestamp"],
                "price": float(trade["price"]),
                "qty": float(trade["quantity"]),
                "is_buyer": trade["side"] == "buy",
            })
            # Persist to daily CSV for backtesting/tuning
            self._trade_writer.write(asset, state.binance_symbol, trade)

        ws.on_price_update = _on_price
        ws.on_trade = _on_trade

        while self._running:
            try:
                logger.info("[ws-%s] Connecting to Binance combined stream...", asset)
                await ws.stream_combined()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[ws-%s] Stream error, reconnecting in 5s", asset)
                await asyncio.sleep(5)

    # -- Coinbase streaming ----------------------------------------------------

    async def _coinbase_stream(self, asset: str, state: AssetState):
        """Stream Coinbase trade matches for dense tick data.

        Replaces Binance WS for live data — Coinbase is not geo-blocked
        and provides 100-1000x more tick density than Binance.US.
        Feeds price_history, tick_buffer, and raw_tick_buffer.
        """
        from data_sources.coinbase.websocket import CoinbaseWebSocket
        cb_ws = CoinbaseWebSocket(asset)

        def on_trade(trade: dict[str, Any]):
            price = Decimal(str(trade["price"]))
            ts = trade["timestamp"]

            # Update price state (replaces Binance ticker)
            state.current_price = price
            state.price_history.append(price)
            state.tick_buffer.append({"ts": ts, "price": float(price)})

            # Update raw tick buffer (replaces Binance aggTrade)
            state.raw_tick_buffer.append({
                "ts": ts,
                "price": float(trade["price"]),
                "qty": float(trade["quantity"]),
                "is_buyer": trade["side"] == "buy",
            })

            # Persist to daily CSV for backtesting/tuning
            ts_ms = int(ts.timestamp() * 1000)
            compat_trade = {
                "agg_trade_id": 0,
                "price": float(trade["price"]),
                "quantity": float(trade["quantity"]),
                "first_id": 0,
                "last_id": 0,
                "timestamp_ms": ts_ms,
                "is_buyer_maker": trade["side"] != "buy",
                "best_price_match": True,
            }
            self._trade_writer.write(asset, state.binance_symbol, compat_trade)

        while self._running:
            try:
                logger.info("[ws-%s] Connecting to Coinbase stream...", asset)
                await cb_ws.stream_trades(on_trade)
            except asyncio.CancelledError:
                cb_ws.stop()
                break
            except Exception:
                logger.exception("[ws-%s] Coinbase stream error, reconnecting in 5s", asset)
                await asyncio.sleep(5)

    # -- Kalshi WebSocket streaming --------------------------------------------

    @staticmethod
    def _dollars_to_cents(val: str) -> int:
        """Convert dollar string (e.g. '0.68') to cents (68)."""
        try:
            return int(round(float(val) * 100))
        except (ValueError, TypeError):
            return 0

    async def _kalshi_ws_stream(self):
        """Stream Kalshi ticker prices via WebSocket.

        Subscribes to current market tickers and updates AssetState
        with live yes_ask/no_ask prices. Re-subscribes every 15 minutes
        when market tickers change.
        """
        from sdk.kalshi.websocket import KalshiWebSocket

        kalshi_ws = KalshiWebSocket(self.client.cfg)

        def on_ticker(msg: dict):
            """Update AssetState with live Kalshi prices."""
            market_ticker = msg.get("market_ticker", "")
            yes_ask_str = msg.get("yes_ask_dollars", "")
            yes_bid_str = msg.get("yes_bid_dollars", "")

            # Find which asset this ticker belongs to
            for asset, state in self.states.items():
                if state.kalshi_market_ticker == market_ticker:
                    if yes_ask_str:
                        yes_ask = KalshiMultiAssetStrategy._dollars_to_cents(yes_ask_str)
                        state.kalshi_yes_ask = yes_ask
                        # no_bid = 100 - yes_ask (Kalshi binary market)
                        state.kalshi_no_bid = 100 - yes_ask
                    if yes_bid_str:
                        yes_bid = KalshiMultiAssetStrategy._dollars_to_cents(yes_bid_str)
                        state.kalshi_yes_bid = yes_bid
                        # no_ask = 100 - yes_bid
                        state.kalshi_no_ask = 100 - yes_bid
                    state.kalshi_last_update = datetime.now(timezone.utc)
                    break

        def on_error(msg: dict):
            logger.warning("[kalshi-ws] Error: %s", msg)

        kalshi_ws.on_ticker = on_ticker
        kalshi_ws.on_error = on_error

        # Start connection in background
        ws_task = asyncio.create_task(kalshi_ws.connect_and_stream())

        # Wait for connection to establish
        await asyncio.sleep(3)

        last_subscribed_tickers: list[str] = []

        while self._running:
            try:
                # Resolve current market tickers for all assets
                new_tickers = []
                for asset, state in self.states.items():
                    try:
                        market_info = fetch_current_market(self.client, state.series)
                        if market_info:
                            mt = market_info.get("market_ticker", "")
                            et = market_info.get("event_ticker", "")
                            if mt:
                                state.kalshi_market_ticker = mt
                                state.kalshi_event_ticker = et
                                state.kalshi_close_time = market_info.get("close_time")
                                new_tickers.append(mt)
                                # Seed initial prices from REST
                                state.kalshi_yes_ask = market_info.get("yes_ask", 0) or 0
                                state.kalshi_no_ask = market_info.get("no_ask", 0) or 0
                                state.kalshi_yes_bid = market_info.get("yes_bid", 0) or 0
                                state.kalshi_no_bid = market_info.get("no_bid", 0) or 0
                    except Exception:
                        logger.debug("[kalshi-ws] Failed to resolve market for %s", asset)

                # Subscribe to new tickers (or update existing subscription)
                if new_tickers and new_tickers != last_subscribed_tickers:
                    if last_subscribed_tickers:
                        # Unsubscribe old, subscribe new
                        await kalshi_ws.unsubscribe_all()
                        await asyncio.sleep(0.5)
                    await kalshi_ws.subscribe_ticker(new_tickers)
                    logger.info("[kalshi-ws] Subscribed to %d tickers: %s",
                                len(new_tickers), ", ".join(new_tickers))
                    last_subscribed_tickers = new_tickers

                # Sleep until next window boundary (re-subscribe with new tickers)
                next_boundary = self._next_window_boundary()
                sleep_s = max(5, (next_boundary - datetime.now(timezone.utc)).total_seconds() + 2)
                await asyncio.sleep(min(sleep_s, 60))  # Check at least every 60s

            except asyncio.CancelledError:
                kalshi_ws.stop()
                break
            except Exception:
                logger.exception("[kalshi-ws] Error in subscription loop")
                await asyncio.sleep(10)

        ws_task.cancel()

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
                    self._update_smas()
                    self._process_deposits()
                    # Log window transition
                    prices = {a: f"${float(s.current_price):.2f}" for a, s in self.states.items() if s.current_price}
                    logger.info("[window] %s | prices: %s", window_id, prices)
                    # Reset window open price and Kalshi prices for new window
                    for state in self.states.values():
                        state.window_open_price = None
                        state.kalshi_yes_ask = 0
                        state.kalshi_no_ask = 0
                        state.kalshi_yes_bid = 0
                        state.kalshi_no_bid = 0

                # Capture window open price at minute 0 (for price_vs_open feature)
                if elapsed < self.ENTRY_GATE_START_S:
                    for asset, state in self.states.items():
                        if state.window_open_price is None and state.current_price is not None:
                            state.window_open_price = float(state.current_price)

                if self.ENTRY_GATE_START_S <= elapsed <= self.ENTRY_GATE_END_S:
                    current_hour = datetime.now(timezone.utc).hour
                    blocked = current_hour in self.BLOCKED_HOURS
                    if blocked:
                        if not hasattr(self, '_last_blocked_log') or self._last_blocked_log != window_id:
                            self._last_blocked_log = window_id
                            logger.info("[window] Hour %d UTC blocked, signals only (no trades)", current_hour)
                    dm = int((elapsed - 300) / 60)
                    mtc = 10 - dm
                    # Skip blocked decision minutes (configurable)
                    if dm in self._blocked_dms:
                        await asyncio.sleep(2)
                        continue
                    for asset, state in self.states.items():
                        # Per-asset min_dm gate (from walk-forward sweep)
                        if dm < state.min_dm:
                            continue
                        await self._process_asset_window(
                            asset, state, window_id, dm=dm, mtc=mtc,
                            blocked=blocked,
                        )
                elif elapsed < self.ENTRY_GATE_START_S:
                    secs_to_gate = self.ENTRY_GATE_START_S - elapsed
                    if secs_to_gate > 30:
                        logger.debug(
                            "[window-loop] Waiting for gate (%.0fs to go)", secs_to_gate,
                        )

                # Sleep until next check (every 2 seconds)
                await asyncio.sleep(2)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[window-loop] Error")
                await asyncio.sleep(2)

    async def _process_asset_window(
        self, asset: str, state: AssetState, window_id: str,
        dm: int = 0, mtc: int = 10, blocked: bool = False,
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

        # Circuit breaker: pause asset if balance drops 40% from session peak, auto-reset after 1 hour
        if state.circuit_open:
            elapsed_since_trip = (datetime.now(tz=timezone.utc) - state.circuit_tripped_at).total_seconds()
            if elapsed_since_trip >= 3600:
                logger.info("[%s] Circuit breaker auto-reset after 1 hour", asset)
                state.circuit_open = False
                state.circuit_tripped_at = None
                # Peak was already reset when breaker tripped
            else:
                mins_left = (3600 - elapsed_since_trip) / 60
                if not hasattr(self, '_last_cb_log') or self._last_cb_log.get(asset) != window_id:
                    if not hasattr(self, '_last_cb_log'):
                        self._last_cb_log = {}
                    self._last_cb_log[asset] = window_id
                    logger.warning("[%s] Circuit breaker open (40%% drawdown), %.0f min until reset", asset, mins_left)
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

        # Use cached Kalshi prices (updated by WebSocket or poller)
        kalshi_yes_ask = state.kalshi_yes_ask
        kalshi_no_ask = state.kalshi_no_ask
        if kalshi_yes_ask == 0 and kalshi_no_ask == 0:
            # Fallback to REST if WebSocket hasn't provided prices yet
            series = state.series
            market_info = fetch_current_market(self.client, series)
            if market_info is None:
                logger.warning("[%s] No open Kalshi market for %s", asset, series)
                return
            kalshi_yes_ask = market_info.get("yes_ask", 0) or 0
            kalshi_no_ask = market_info.get("no_ask", 0) or 0
            state.kalshi_yes_ask = kalshi_yes_ask
            state.kalshi_no_ask = kalshi_no_ask
            state.kalshi_market_ticker = market_info.get("market_ticker")
            state.kalshi_event_ticker = market_info.get("event_ticker")

        # Build market_info from cached state for execution adapter
        market_info = {
            "market_ticker": state.kalshi_market_ticker,
            "event_ticker": state.kalshi_event_ticker,
            "close_time": state.kalshi_close_time,
            "yes_ask": kalshi_yes_ask,
            "no_ask": kalshi_no_ask,
            "yes_bid": state.kalshi_yes_bid,
            "no_bid": state.kalshi_no_bid,
        }

        metadata = self._build_metadata(state)
        metadata["kalshi_market"] = market_info
        metadata["decision_minute"] = dm
        # v10 Kalshi features for ML processor
        metadata["kalshi_yes_ask"] = kalshi_yes_ask
        metadata["kalshi_yes_bid"] = state.kalshi_yes_bid
        metadata["kalshi_no_ask"] = kalshi_no_ask
        metadata["kalshi_mins_to_close"] = state.kalshi_last_update and (
            (datetime.fromisoformat(state.kalshi_close_time) - datetime.now(timezone.utc)).total_seconds() / 60.0
            if state.kalshi_close_time else None
        )
        spot_price = float(state.current_price) if state.current_price else 0.0

        # Select model variant based on day of week (fallback to standard)
        dow = datetime.now(timezone.utc).weekday()
        if dow >= 5 and state.weekend_ml_processor is not None and state.weekend_ensemble_weights is not None:
            active_ml = state.weekend_ml_processor
            active_weights = state.weekend_ensemble_weights
            active_lstm = state.weekend_lstm_processor
            model_label = "weekend"
        elif dow < 5 and state.weekday_ml_processor is not None and state.weekday_ensemble_weights is not None:
            active_ml = state.weekday_ml_processor
            active_weights = state.weekday_ensemble_weights
            active_lstm = state.weekday_lstm_processor
            model_label = "weekday"
        else:
            active_ml = state.ml_processor
            active_weights = state.ensemble_weights
            active_lstm = state.lstm_processor
            model_label = "standard"

        # Stacked ensemble: LSTM → XGB (lstm_p is an XGB feature)
        if active_weights is not None and active_ml is not None:
            _ml_w, ens_threshold = active_weights

            # Step 1: Get LSTM probability first (feeds into XGB)
            lstm_p = None
            if active_lstm is not None:
                lstm_p = active_lstm.predict_proba(
                    state.current_price, list(state.price_history), metadata,
                )
            metadata["lstm_p"] = lstm_p if lstm_p is not None else 0.5

            # Step 2: Run XGB with lstm_p as a stacked feature
            # XGB was trained with lstm_p baked in, so its output IS the
            # final probability. No dynamic weighting or fusion needed.
            ml_p = active_ml.predict_proba(
                state.current_price, list(state.price_history), metadata,
            )
            if ml_p is None:
                logger.debug("[%s] Ensemble (%s): XGB returned None for window %s", asset, model_label, window_id)
                return

            ensemble_p = ml_p
            fusion_p = 0.5  # kept for logging only, not used in decision

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
                lstm_str = f" lstm={lstm_p:.3f}" if lstm_p is not None else ""
                logger.info(
                    "[%s] Ensemble skip (%s): p=%.3f (xgb=%.3f%s fus=%.3f) below threshold %.2f",
                    asset, model_label, ensemble_p, ml_p, lstm_str, fusion_p, ens_threshold,
                )
                self._log_signal(
                    asset, window_id, dm, ml_p, fusion_p, ensemble_p,
                    ens_threshold, "NONE", kalshi_yes_ask, kalshi_no_ask,
                    0, "skip_threshold", 0, spot_price,
                )
                return

            score = confidence * 100
            lstm_str = f" lstm={lstm_p:.3f}" if lstm_p is not None else ""

            # Consensus confirm: require 2 consecutive checkpoints to agree
            if self._require_confirm:
                confirm_key = f"{asset}_{window_id}"
                if not hasattr(self, '_last_signal'):
                    self._last_signal = {}
                prev = self._last_signal.get(confirm_key)
                self._last_signal[confirm_key] = direction
                if prev != direction:
                    logger.info(
                        "[%s] Ensemble %s (%s): p=%.3f price=%dc [AWAITING CONFIRM]",
                        asset, direction, model_label, ensemble_p, entry_price,
                    )
                    return

            logger.info(
                "[%s] Ensemble %s (%s): p=%.3f (xgb=%.3f%s fus=%.3f) price=%dc%s",
                asset, direction, model_label, ensemble_p, ml_p, lstm_str, fusion_p, entry_price,
                " [BLOCKED]" if blocked else "",
            )

            if blocked:
                self._log_signal(
                    asset, window_id, dm, ml_p, fusion_p, ensemble_p,
                    ens_threshold, direction, kalshi_yes_ask, kalshi_no_ask,
                    entry_price, "skip_blocked", 0, spot_price,
                )
                return

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
        """Run fusion processors and return P(BULLISH) in [0, 1].

        Uses contribution-based probability: the net directional contribution
        is normalized by the total weight capacity of the fusion engine.
        This produces values proportional to signal strength instead of the
        old avg_confidence mapping which jumped from 0.50 to 0.63+ on any
        bullish signal (causing 98% bullish bias).
        """
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

        # Contribution-based probability mapping
        bullish_c = fused.metadata["bullish_contrib"]
        bearish_c = fused.metadata["bearish_contrib"]
        net = bullish_c - bearish_c

        # Normalize by sum of fusion weights (theoretical max contribution
        # if all processors fired at max strength & confidence = 1.0)
        weight_sum = sum(self.fusion_engine.weights.values())
        if weight_sum < 0.001:
            return 0.5

        # P(BULLISH) = 0.5 + net/weight_sum * 0.5
        # Two WEAK bullish signals -> ~0.56  (was 0.63)
        # Two MODERATE bullish     -> ~0.64  (was 0.65)
        # Strong multi-processor   -> ~0.80+ (was 0.73)
        fusion_p = 0.5 + (net / weight_sum) * 0.5
        return max(0.01, min(0.99, fusion_p))

    def _build_metadata(self, state: AssetState) -> dict[str, Any]:
        """Build metadata dict consumed by signal processors."""
        meta: dict[str, Any] = {
            "tick_buffer": list(state.tick_buffer),
        }

        # Pass raw_tick_buffer (with qty/is_buyer) for ML volume features
        if state.raw_tick_buffer:
            meta["raw_tick_buffer"] = list(state.raw_tick_buffer)

        # Spot price for divergence processor
        if state.current_price is not None:
            meta["spot_price"] = float(state.current_price)

        # Window open price (minute 0) for price_vs_open feature
        if state.window_open_price is not None:
            meta["window_open_price"] = state.window_open_price

        # Simple momentum: 5-period ROC
        if len(state.price_history) >= 5:
            recent = list(state.price_history)
            old = float(recent[-5])
            cur = float(recent[-1])
            meta["momentum"] = (cur - old) / old if old else 0.0

        # Sentiment score — placeholder; in production, wire to an actual
        # sentiment feed. Using 50 (neutral) for now.
        meta["sentiment_score"] = 50.0

        # SMA features (precomputed per asset)
        smas = self._asset_smas.get(state.asset, {})
        meta["sma5"] = smas.get("sma5")
        meta["sma15"] = smas.get("sma15")
        meta["sma30"] = smas.get("sma30")

        return meta

    # -- trade CSV logging -----------------------------------------------------

    def _ensure_trade_log(self):
        """Create the dry CSV file with headers if it doesn't exist."""
        if self.TRADE_LOG_PATH.exists():
            return
        self.TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.TRADE_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(self.TRADE_LOG_FIELDS)

    def _ensure_trade_log_live(self):
        """Create the live CSV file with headers if it doesn't exist."""
        if self.TRADE_LOG_LIVE_PATH.exists():
            return
        self.TRADE_LOG_LIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.TRADE_LOG_LIVE_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.TRADE_LOG_LIVE_FIELDS).writeheader()

    def _log_trade_entry(self, trade: TradeRecord, dm: int, mtc: int):
        """Append a row when a trade is placed. Routes to live or dry CSV."""
        if trade.dry_run:
            self._log_trade_entry_dry(trade, dm, mtc)
        else:
            self._log_trade_entry_live(trade, dm, mtc)

    def _log_trade_entry_dry(self, trade: TradeRecord, dm: int, mtc: int):
        """Append a row to trades.csv (dry schema)."""
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

    def _log_trade_entry_live(self, trade: TradeRecord, dm: int, mtc: int):
        """Append a row to trades_live.csv (live schema with execution audit)."""
        with self._csv_lock:
            self._ensure_trade_log_live()
            row = {
                "timestamp": trade.placed_at.isoformat() if trade.placed_at else "",
                "asset": trade.asset,
                "window_id": trade.window_id,
                "market_ticker": trade.market_ticker,
                "event_ticker": trade.event_ticker,
                "direction": trade.direction,
                "side": trade.side,
                "intended_price_cents": trade.price_cents,
                "fill_price_cents": "",   # populated at settlement (avg fill from cost/filled)
                "intended_count": trade.count,
                "filled_count": trade.filled,
                "cost": f"{trade.cost_dollars:.4f}",
                "fees": f"{trade.fees_dollars:.4f}",
                "kalshi_order_id": trade.order_id or "",
                "dm": dm,
                "mtc": mtc,
                "confidence": f"{trade.confidence:.4f}",
                "score": f"{trade.score:.2f}",
                "outcome": "",
                "expected_revenue": "",
                "verified_revenue": "",
                "settlement_verified": "False",
                "verified_at": "",
                "pnl": "",
                "balance_after": "",
            }
            with open(self.TRADE_LOG_LIVE_PATH, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.TRADE_LOG_LIVE_FIELDS).writerow(row)

    def _log_settlement(self, trade: TradeRecord):
        """Update the trade's row with outcome and PnL after settlement.

        Routes to dry or live CSV based on trade.dry_run.
        """
        if trade.dry_run:
            self._log_settlement_dry(trade)
        else:
            self._log_settlement_live(trade)

    def _log_settlement_dry(self, trade: TradeRecord):
        """Update the dry trade's row with outcome and PnL."""
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

    def _log_settlement_live(self, trade: TradeRecord):
        """Update the live trade's row with outcome + expected_revenue.

        Verified fields (verified_revenue, settlement_verified, verified_at,
        balance_after, final pnl) are filled in later by _log_verification.
        """
        with self._csv_lock:
            if not self.TRADE_LOG_LIVE_PATH.exists():
                return
            self._update_live_row(
                trade,
                outcome=trade.settlement_outcome or "",
                expected_revenue=f"{(trade.expected_revenue_dollars or 0.0):.4f}",
            )

    def _log_verification(self, trade: TradeRecord):
        """Final update to a live trade row after Kalshi verification.

        Sets verified_revenue, settlement_verified, verified_at, fill_price,
        final pnl, balance_after.
        """
        if trade.dry_run:
            return
        with self._csv_lock:
            if not self.TRADE_LOG_LIVE_PATH.exists():
                return
            try:
                acct = self.account_manager.get_account(trade.series)
                balance = f"{acct.balance_dollars:.2f}"
            except KeyError:
                balance = ""
            verified_rev = trade.verified_revenue_dollars
            if verified_rev is None:
                verified_rev = trade.expected_revenue_dollars or 0.0
            pnl = verified_rev - (trade.cost_dollars + trade.fees_dollars)
            # Compute average fill price from cost / filled
            fill_px = ""
            if trade.filled and trade.filled > 0:
                fill_px = f"{int(round(trade.cost_dollars / trade.filled * 100))}"
            self._update_live_row(
                trade,
                fill_price_cents=fill_px,
                verified_revenue=f"{verified_rev:.4f}",
                settlement_verified="True",
                verified_at=trade.verified_at.isoformat() if trade.verified_at else "",
                pnl=f"{pnl:+.4f}",
                balance_after=balance,
            )

    def _update_live_row(self, trade: TradeRecord, **fields_to_update):
        """In-place update of a live trade row matched by market_ticker + event_ticker.

        Reads, updates, rewrites the entire CSV. Same approach as the dry
        settlement updater.
        """
        rows: list[dict] = []
        try:
            with open(self.TRADE_LOG_LIVE_PATH, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (row.get("market_ticker") == trade.market_ticker
                            and row.get("event_ticker") == trade.event_ticker):
                        for k, v in fields_to_update.items():
                            row[k] = v
                    rows.append(row)
        except Exception as e:
            logger.warning("[trade-log] failed to read live CSV: %s", e)
            return

        try:
            with open(self.TRADE_LOG_LIVE_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.TRADE_LOG_LIVE_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            logger.warning("[trade-log] failed to write live CSV: %s", e)

    # -- balance CSV logging ---------------------------------------------------

    def _ensure_balance_log(self, live: bool = False):
        """Create the balance CSV with headers if it doesn't exist."""
        path = self.BALANCE_LOG_LIVE_PATH if live else self.BALANCE_LOG_PATH
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(self.BALANCE_LOG_FIELDS)

    def _balance_log_path_for(self, asset: str) -> Path:
        """Return live balance CSV for live assets, dry one otherwise."""
        if self.execution.is_dry_run(asset):
            return self.BALANCE_LOG_PATH
        return self.BALANCE_LOG_LIVE_PATH

    def _log_balance(self, event: str, asset: str = ""):
        """Append a balance snapshot row. Routes per-asset to live or dry CSV."""
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
                path = self._balance_log_path_for(asset)
                self._ensure_balance_log(live=(path == self.BALANCE_LOG_LIVE_PATH))
                with open(path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        now, event, asset,
                        f"{acct.balance_dollars:.2f}",
                        f"{acct.pnl_dollars:.2f}",
                        kalshi_bal,
                    ])
            except KeyError:
                pass
        else:
            # Log all assets -- each routed independently
            for a in self.assets:
                series = series_for_asset(a)
                try:
                    acct = self.account_manager.get_account(series)
                    path = self._balance_log_path_for(a)
                    self._ensure_balance_log(live=(path == self.BALANCE_LOG_LIVE_PATH))
                    with open(path, "a", newline="") as f:
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

                            # Circuit breaker: check if balance dropped 40% from session peak
                            try:
                                series = self.states[trade.asset].series
                                acct = self.account_manager.get_account(series)
                                bal = acct.balance_dollars
                            except (KeyError, AttributeError):
                                bal = None
                            if bal is not None:
                                if state.session_peak_balance is None or bal > state.session_peak_balance:
                                    state.session_peak_balance = bal
                                if state.session_peak_balance > 0:
                                    drawdown_pct = (state.session_peak_balance - bal) / state.session_peak_balance
                                    if drawdown_pct >= 0.40 and not state.circuit_open:
                                        state.circuit_open = True
                                        state.circuit_tripped_at = datetime.now(tz=timezone.utc)
                                        logger.warning(
                                            "[%s] Circuit breaker TRIPPED: %.0f%% drawdown ($%.2f -> $%.2f), pausing 1 hour",
                                            trade.asset, drawdown_pct * 100,
                                            state.session_peak_balance, bal,
                                        )

                    for trade in settled:
                        state.pending_settlements.remove(trade)
                        # Live trades go to verification queue (Kalshi confirms revenue)
                        if not trade.dry_run and not trade.settlement_verified:
                            state.pending_verifications.append(trade)

                await asyncio.sleep(self.settlement_delay_seconds)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[settlement-loop] Error")
                await asyncio.sleep(self.settlement_delay_seconds)

    # -- settlement verification (live trades only) ----------------------------

    async def _verification_loop(self):
        """Confirm live trade payouts against Kalshi /portfolio/settlements.

        Runs every 60s. For each live trade in pending_verifications:
          - Wait at least 60s after settlement (Kalshi posting delay)
          - Call execution.verify_settlement(trade) which fetches Kalshi's
            actual revenue and applies any delta
          - On success: remove from queue
          - On failure (settlement not yet in Kalshi history): leave in queue
            and retry next iteration. After 5 minutes, log STALE warning once
            but keep retrying.

        Also publishes pending/stale counts to data/runtime_state.json so
        the manager UI can render the reconciliation panel.
        """
        while self._running:
            try:
                await asyncio.sleep(60)
                now = datetime.now(timezone.utc)
                pending_total = 0
                stale_total = 0
                for state in self.states.values():
                    if not state.pending_verifications:
                        continue
                    verified = []
                    for trade in state.pending_verifications:
                        if trade.settled_at is None:
                            continue
                        elapsed = (now - trade.settled_at).total_seconds()
                        if elapsed < 60:
                            continue
                        expected_before = trade.expected_revenue_dollars or 0.0
                        try:
                            ok = self.execution.verify_settlement(trade)
                        except Exception:
                            logger.exception(
                                "[verify] error verifying %s", trade.market_ticker,
                            )
                            ok = False
                        if ok:
                            verified.append(trade)
                            actual = trade.verified_revenue_dollars or 0.0
                            delta = actual - expected_before
                            if abs(delta) > 0.005:
                                self._runtime_state["verification"]["total_corrections"] += 1
                                self._runtime_state["verification"]["total_drift_dollars"] += delta
                            # Persist the final verified row to trades_live.csv
                            self._log_verification(trade)
                        elif elapsed > 300:
                            stale_total += 1
                            if not trade.verification_warned:
                                logger.warning(
                                    "[verify] STALE %s -- not in Kalshi history after %.0fs, will keep retrying",
                                    trade.market_ticker, elapsed,
                                )
                                trade.verification_warned = True
                    for trade in verified:
                        state.pending_verifications.remove(trade)
                    pending_total += len(state.pending_verifications)

                # Publish current verification stats to runtime state
                self._runtime_state["verification"]["pending_count"] = pending_total
                self._runtime_state["verification"]["stale_count"] = stale_total
                self._write_runtime_state()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[verification-loop] Error")
                await asyncio.sleep(60)

    def _write_runtime_state(self) -> None:
        """Persist runtime state for the manager UI to read.

        Best-effort, never raises. Manager polls this file once per second
        so the file should be small and the write should be quick.
        """
        try:
            self._runtime_state["last_updated"] = datetime.now(timezone.utc).isoformat()
            path = Path("data/runtime_state.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._runtime_state, indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception as e:
            logger.debug("[runtime-state] write failed: %s", e)

    # -- reconciliation --------------------------------------------------------

    def _get_ensemble_param(self, asset: str, model_label: str, key: str, default: float) -> float:
        """Read a dynamic weight param from config/trading.json ensemble block.

        Looks up the correct ensemble block based on model_label (standard,
        weekday, weekend), then falls back to the standard block, then to
        the hardcoded default.
        """
        try:
            with open(self.CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            asset_cfg = cfg.get("assets", {}).get(asset, {})

            # Determine which ensemble block to read
            if model_label == "weekday":
                ens = asset_cfg.get("ensemble_weekday", {})
            elif model_label == "weekend":
                ens = asset_cfg.get("ensemble_weekend", {})
            else:
                ens = asset_cfg.get("ensemble", {})

            val = ens.get(key)
            if val is not None:
                return float(val)

            # Fall back to standard ensemble block
            val = asset_cfg.get("ensemble", {}).get(key)
            if val is not None:
                return float(val)
        except Exception:
            pass
        return default

    def _live_assets(self) -> list[str]:
        """List of assets currently running in live mode (dry_run=False)."""
        return [a for a in self.assets if not self.execution.is_dry_run(a)]

    async def _reconciliation_loop(self):
        """Periodically compare LIVE sub-accounts against real Kalshi balance.

        Only sums sub-accounts for assets where dry_run=False, since the
        Kalshi balance only reflects actual on-exchange holdings. Dry-run
        sub-accounts have no Kalshi counterpart and would skew the diff.

        If no assets are live, the loop logs once and skips the API call
        but still publishes a fresh runtime_state snapshot so the manager
        UI can render an empty live row instead of stale data.
        """
        first_pass_no_live_logged = False
        # Publish initial state immediately so manager has something to read
        self._update_reconciliation_state(live_assets=[], kalshi_balance=None,
                                          local_live_total=None, drift=None)
        self._write_runtime_state()
        while self._running:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes

                live_assets = self._live_assets()
                if not live_assets:
                    if not first_pass_no_live_logged:
                        logger.info(
                            "[reconcile] No live assets -- skipping Kalshi balance check"
                        )
                        first_pass_no_live_logged = True
                    self._update_reconciliation_state(
                        live_assets=[], kalshi_balance=None,
                        local_live_total=None, drift=None,
                    )
                    self._write_runtime_state()
                    continue
                first_pass_no_live_logged = False  # reset if assets become live later

                balance = fetch_balance(self.client)
                if balance is None:
                    continue

                # Sum only LIVE sub-accounts
                local_live_total = 0.0
                for asset in live_assets:
                    try:
                        acct = self.account_manager.get_account(
                            self.states[asset].series
                        )
                        local_live_total += acct.balance_dollars
                    except (KeyError, AttributeError):
                        continue

                discrepancy = balance - local_live_total
                logger.info(
                    "[reconcile] live local=$%.2f kalshi=$%.2f diff=$%+.2f (assets: %s)",
                    local_live_total, balance, discrepancy, ",".join(live_assets),
                )
                if abs(discrepancy) > 1.0:
                    logger.warning(
                        "[reconcile] LIVE BALANCE DRIFT > $1.00 -- run scripts/reconcile.py --kalshi for details"
                    )

                self._update_reconciliation_state(
                    live_assets=live_assets,
                    kalshi_balance=balance,
                    local_live_total=local_live_total,
                    drift=discrepancy,
                )
                self._write_runtime_state()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[reconcile-loop] Error")
                await asyncio.sleep(1800)

    def _update_reconciliation_state(
        self,
        *,
        live_assets: list[str],
        kalshi_balance: Optional[float],
        local_live_total: Optional[float],
        drift: Optional[float],
    ) -> None:
        """Update the in-memory reconciliation snapshot for runtime_state.json."""
        self._runtime_state["reconciliation"] = {
            "last_run_at": datetime.now(timezone.utc).isoformat(),
            "kalshi_balance": kalshi_balance,
            "live_local_total": local_live_total,
            "drift": drift,
            "live_assets": live_assets,
        }
