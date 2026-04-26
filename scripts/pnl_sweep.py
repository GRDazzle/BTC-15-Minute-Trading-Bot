"""
PnL-based ensemble sweep: find optimal (ml_weight, threshold) using real Kalshi prices.

Unlike the accuracy sweep (ensemble_sweep.py) which scores by signal correctness,
this sweep scores by dollar PnL -- factoring in entry prices, Kelly bands, position
sizing, fees, and settlement outcomes from actual Kalshi polling data.

Three phases:
  1. Collect ML + fusion probabilities in one backtest pass (reuses simulator)
  2. Enrich checkpoints with Kalshi bid/ask prices and outcomes
  3. Sweep weight/threshold combos scoring by total PnL

Optimizations:
  - Pre-filters Binance windows to only those with Kalshi data (~5x Phase 1 speedup)
  - Limits aggTrades loading to Kalshi date range (less I/O)
  - Parallel asset processing via ProcessPoolExecutor (~4x on multi-core)

Requires Kalshi polling data in data/kalshi_polls/KX{ASSET}15M/.

Usage:
  python scripts/pnl_sweep.py --asset BTC
  python scripts/pnl_sweep.py --asset BTC,ETH,SOL,XRP --min-dm 2
  python scripts/pnl_sweep.py --asset BTC --min-dm 2 --max-dm 3 --model-suffix _early
"""
import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader import load_fear_greed
from backtester.data_loader_ticks import load_aggtrades_multi, generate_tick_windows
from core.tick_window_slicer import TickWindowSlicer
from backtester.data_loader_kalshi import load_kalshi_windows, get_kalshi_prices
from backtester.simulator import BacktestSimulator

from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
KRAKEN_DIR = PROJECT_ROOT / "data" / "aggtrades_kraken"
KALSHI_DATA_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
FG_CSV = PROJECT_ROOT / "data" / "historical" / "fear_greed.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pnl_sweep"
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"

# Sweep grid: 2-way XGB+LSTM blend, each model weighted by its own confidence^k.
# ensemble_p = (xgb_conf^k_xgb * ml_p + lstm_conf^k_lstm * lstm_p)
#              / (xgb_conf^k_xgb + lstm_conf^k_lstm)
# Whichever model is more confident gets more say. No ml_weight needed.
THRESHOLDS = [0.55, 0.58, 0.60, 0.62, 0.65, 0.67, 0.70, 0.72, 0.75, 0.78]  # 10
MAX_PRICES = [80, 85, 90]  # 3
MIN_PRICES = [55, 60, 65]  # 3 — min_price_cents floor; entry rejected below this
DYNAMIC_K_XGB_VALUES = [3.0, 4.5, 6.0, 8.0]  # 4
DYNAMIC_K_LSTM_VALUES = [3.0, 4.5, 6.0, 8.0]  # 4
MIN_DM_VALUES = [2, 3, 4, 5, 6, 7]  # 6
# Total combos per asset: 10 * 3 * 3 * 4 * 4 * 6 = 8640

# Ensemble sweep candidate directory
ENSEMBLE_CANDIDATE_DIR = PROJECT_ROOT / "output" / "ensemble_sweep"

# Kalshi fee per contract in cents
KALSHI_FEE_CENTS = 2

# Minimum days of Kalshi data required to run PnL sweep
MIN_KALSHI_DAYS = 3


def load_candidates(asset: str, model_suffix: str = "") -> list[tuple[float, float]] | None:
    """Load ensemble sweep candidate set for an asset.

    Returns list of (ml_weight, threshold) tuples, or None if no candidate file.
    """
    cand_path = ENSEMBLE_CANDIDATE_DIR / f"{asset}{model_suffix}_candidates.json"
    if not cand_path.exists():
        return None

    try:
        with open(cand_path, "r") as f:
            data = json.load(f)
        if not data:
            return None
        combos = [(c["ml_weight"], c["threshold"]) for c in data]
        return combos
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Warning: Failed to parse {cand_path}: {e}")
        return None


def build_processors() -> list:
    """Instantiate signal processors with sweep-validated parameters."""
    return [
        SpikeDetectionProcessor(
            spike_threshold=0.003,
            velocity_threshold=0.0015,
            lookback_periods=20,
            min_confidence=0.55,
        ),
        TickVelocityProcessor(
            velocity_threshold_60s=0.001,
            velocity_threshold_30s=0.0007,
            min_ticks=5,
            min_confidence=0.55,
        ),
    ]


def load_config(asset: str) -> dict:
    """Load per-asset config from trading.json."""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        return {
            "initial_balance": 25.0,
            "max_contracts_per_trade": 10,
            "max_price_cents": 85,
            "min_price_cents": 15,
        }

    defaults = config.get("defaults", {})
    asset_cfg = config.get("assets", {}).get(asset.upper(), {})

    return {
        "initial_balance": asset_cfg.get("initial_balance", defaults.get("initial_balance", 25.0)),
        "max_contracts_per_trade": asset_cfg.get("max_contracts_per_trade", defaults.get("max_contracts_per_trade", 10)),
        "max_price_cents": asset_cfg.get("max_price_cents", defaults.get("max_price_cents", 85)),
        "min_price_cents": asset_cfg.get("min_price_cents", defaults.get("min_price_cents", 15)),
    }


def calculate_contracts(
    balance: float,
    price_cents: int,
    confidence: float,
    max_contracts: int,
) -> int:
    """Determine how many contracts to buy (matches live execution)."""
    cost_per = (price_cents + KALSHI_FEE_CENTS) / 100.0
    if cost_per <= 0 or balance <= 0:
        return 0
    max_by_balance = int(balance / cost_per)
    # Live: scale = confidence * (score / 100), and score = confidence * 100
    # So scale = confidence * confidence = confidence^2
    scale = min(1.0, confidence * confidence)
    desired = max(1, int(max_by_balance * scale))
    return min(desired, max_contracts)


def sweep_combo_pnl(
    threshold: float,
    k_xgb: float,
    k_lstm: float,
    min_dm: int,
    enriched_windows: list[dict],
    config: dict,
    max_dm: int | None = None,
) -> dict:
    """Evaluate one (threshold, k_xgb, k_lstm, min_dm) combo by dollar PnL.

    Formula: ensemble_p = (xgb_conf^k_xgb * ml_p + lstm_conf^k_lstm * lstm_p)
                          / (xgb_conf^k_xgb + lstm_conf^k_lstm)

    Both models weighted by their own confidence (deviation from 0.5); whichever
    is more confident gets more say. No fusion term.
    """
    balance = config["initial_balance"]
    max_contracts = config["max_contracts_per_trade"]
    max_price = config["max_price_cents"]
    min_price = config["min_price_cents"]

    total_windows = len(enriched_windows)
    traded_count = 0
    win_count = 0
    total_pnl = 0.0
    skipped_kelly = 0
    skipped_no_ask = 0

    for win in enriched_windows:
        outcome = win["outcome"]
        if not outcome:
            continue

        predicted = None
        confidence = 0.0
        entry_price = 0
        side = ""
        had_signal = False

        for cp in win["checkpoints"]:
            if cp.get("kalshi") is None:
                continue

            # Skip checkpoints outside dm gate
            if cp["dm"] < min_dm:
                continue
            if max_dm is not None and cp["dm"] > max_dm:
                continue

            # 2-way blend: each model weighted by its own confidence^k.
            ml_p_val = cp["ml_p"]
            lstm_p_val = cp.get("lstm_p", 0.5)
            xgb_conf = abs(ml_p_val - 0.5) * 2.0
            lstm_conf = abs(lstm_p_val - 0.5) * 2.0
            xgb_w_raw = xgb_conf ** k_xgb
            lstm_w_raw = lstm_conf ** k_lstm
            total_w = xgb_w_raw + lstm_w_raw
            if total_w < 1e-9:
                ensemble_p = 0.5  # both models neutral, no conviction
            else:
                ensemble_p = (xgb_w_raw * ml_p_val + lstm_w_raw * lstm_p_val) / total_w

            if ensemble_p >= threshold:
                p = cp["kalshi"]["yes_ask"]
                if p <= 0 or p >= 100:
                    had_signal = True
                    continue
                if p < min_price or p > max_price:
                    had_signal = True
                    continue
                predicted = "BULLISH"
                confidence = ensemble_p
                side = "yes"
                entry_price = p
                break
            elif ensemble_p <= 1.0 - threshold:
                p = cp["kalshi"]["no_ask"]
                if p <= 0 or p >= 100:
                    had_signal = True
                    continue
                if p < min_price or p > max_price:
                    had_signal = True
                    continue
                predicted = "BEARISH"
                confidence = 1.0 - ensemble_p
                side = "no"
                entry_price = p
                break

        if predicted is None:
            if had_signal:
                skipped_kelly += 1
            continue

        # Position sizing
        contracts = calculate_contracts(balance, entry_price, confidence, max_contracts)
        if contracts < 1:
            continue

        # Cost and fees
        cost = (entry_price / 100.0) * contracts
        fees = (KALSHI_FEE_CENTS / 100.0) * contracts
        balance -= (cost + fees)

        # Settlement
        won = (side == outcome)
        revenue = contracts * 1.00 if won else 0.0
        pnl = revenue - cost - fees
        balance += revenue

        traded_count += 1
        total_pnl += pnl
        if won:
            win_count += 1

    win_rate = win_count / traded_count * 100 if traded_count else 0.0
    avg_pnl = total_pnl / traded_count if traded_count else 0.0
    traded_pct = traded_count / total_windows * 100 if total_windows else 0.0

    return {
        "threshold": threshold,
        "k_xgb": k_xgb,
        "k_lstm": k_lstm,
        "min_dm": min_dm,
        "max_price_cents": max_price,
        "min_price_cents": min_price,
        "total_pnl": round(total_pnl, 4),
        "win_rate": round(win_rate, 1),
        "traded_count": traded_count,
        "traded_pct": round(traded_pct, 1),
        "win_count": win_count,
        "loss_count": traded_count - win_count,
        "avg_pnl_per_trade": round(avg_pnl, 4),
        "final_balance": round(balance, 2),
        "total_windows": total_windows,
        "skipped_kelly": skipped_kelly,
        "skipped_no_ask": skipped_no_ask,
    }


def run_asset(asset: str, min_dm: int, hours: int | None = None,
              max_dm: int | None = None, model_suffix: str = "",
              from_date=None, to_date=None, output_subdir: str | None = None,
              balance_override: float | None = None) -> dict | None:
    """Run PnL sweep for one asset. Returns best combo dict or None."""
    # Configure loguru for worker processes (suppress DEBUG noise)
    logger.remove()
    model_label = f"{asset}{model_suffix}" if model_suffix else asset
    log_path = PROJECT_ROOT / "logs" / f"pnl_sweep_{model_label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), mode="w", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(sys.stderr, level="WARNING")

    from core.strategy_brain.signal_processors.ml_processor import MLProcessor
    from core.strategy_brain.signal_processors.lstm_processor import LSTMProcessor

    config = load_config(asset)

    # Balance override — walk-forward sweeps use a large balance so combos don't
    # get starved after a handful of trades at live's $25/asset cap.
    if balance_override is not None:
        config["initial_balance"] = float(balance_override)

    # Load Kalshi data
    print(f"\nLoading Kalshi polling data for {asset}...")
    kalshi_windows = load_kalshi_windows(KALSHI_DATA_DIR, asset)
    if not kalshi_windows:
        print(f"No Kalshi data for {asset}. Start the bot to collect data.")
        return None

    # Filter to recent N hours OR to an explicit [from_date, to_date] range.
    # The 4-fold procedure uses date-range so each fold gets a clean slice.
    from datetime import datetime as dt, date, timedelta as td, timezone
    if hours is not None:
        cutoff = dt.now(tz=timezone.utc) - td(hours=hours)
        before = len(kalshi_windows)
        kalshi_windows = {k: v for k, v in kalshi_windows.items() if k >= cutoff}
        print(f"  Filtered to last {hours}h: {before} -> {len(kalshi_windows)} Kalshi windows")
        if not kalshi_windows:
            print(f"No Kalshi data in the last {hours}h for {asset}.")
            return None
    elif from_date is not None or to_date is not None:
        # Inclusive on both ends by window close_time (UTC).
        lo = dt.combine(from_date, dt.min.time(), tzinfo=timezone.utc) if from_date else dt.min.replace(tzinfo=timezone.utc)
        hi = dt.combine(to_date + td(days=1), dt.min.time(), tzinfo=timezone.utc) if to_date else dt.max.replace(tzinfo=timezone.utc)
        before = len(kalshi_windows)
        kalshi_windows = {k: v for k, v in kalshi_windows.items() if lo <= k < hi}
        print(f"  Filtered to [{from_date}..{to_date}]: {before} -> {len(kalshi_windows)} Kalshi windows")
        if not kalshi_windows:
            print(f"No Kalshi data in [{from_date}..{to_date}] for {asset}.")
            return None

    # Check minimum windows (relax MIN_KALSHI_DAYS when --hours is set)
    kalshi_dates = sorted(set(ct.date() for ct in kalshi_windows.keys()))
    if hours is None and len(kalshi_dates) < MIN_KALSHI_DAYS:
        print(f"Only {len(kalshi_dates)} days of Kalshi data for {asset} "
              f"(need {MIN_KALSHI_DAYS}). Skipping PnL sweep.")
        return None
    min_windows = 20
    if len(kalshi_windows) < min_windows:
        print(f"Only {len(kalshi_windows)} Kalshi windows for {asset} "
              f"(need {min_windows}). Skipping PnL sweep.")
        return None

    # Determine Kalshi date range to limit aggTrades loading
    kalshi_min_date = kalshi_dates[0]
    today = date.today()
    days_back = (today - kalshi_min_date).days + 2  # +2 for buffer (warmup ticks)

    # Load tick data (limited to Kalshi date coverage)
    print(f"Loading aggTrades data for {asset} (last {days_back} days to match Kalshi)...")
    ticks = load_aggtrades_multi(DATA_DIR, asset, days=days_back)
    if not ticks:
        print(f"No aggTrades data for {asset}. Run download_binance_aggtrades.py first.")
        return None

    # Lower tick thresholds for Binance.US data (~30 trades/hr vs thousands on .com)
    windows = generate_tick_windows(ticks, min_warmup_ticks=1, min_during_ticks=1)
    if not windows:
        print(f"No valid tick windows for {asset}")
        return None

    # Also load Kraken ticks so the simulator's merged-buffer path matches live's
    # metadata["raw_tick_buffer"] (CB+KR merged+sorted, kalshi_strategy.py:2005-2009).
    # Without this, sweep's LSTM/XGB see CB-only — different distribution than live.
    kraken_ticks = load_aggtrades_multi(KRAKEN_DIR, asset, days=days_back)
    print(f"  Loaded {len(ticks)} Coinbase + {len(kraken_ticks)} Kraken ticks")

    # Build shared slicer — authoritative tick primitive for all metadata tick
    # buffers during sweep. check_ts -> get_merged_window(ts, 900, [cb, kr]).
    slicer = TickWindowSlicer()
    if ticks:
        slicer.extend("coinbase", [{
            "ts": t.ts, "price": t.price, "qty": t.qty, "is_buyer": t.is_buyer,
        } for t in ticks])
    if kraken_ticks:
        slicer.extend("kraken", [{
            "ts": t.ts, "price": t.price, "qty": t.qty, "is_buyer": t.is_buyer,
        } for t in kraken_ticks])

    # Pre-filter: only keep windows whose window_end matches a Kalshi close_time
    kalshi_close_times = set(kalshi_windows.keys())
    all_window_count = len(windows)
    windows = [w for w in windows if w.window_end in kalshi_close_times]
    print(f"  Pre-filtered: {all_window_count} -> {len(windows)} windows (Kalshi-matched only)")

    if not windows:
        print(f"No overlapping windows between Binance and Kalshi data for {asset}")
        return None

    fg_scores = load_fear_greed(FG_CSV) if FG_CSV.exists() else {}

    # Load ML model (with optional suffix for early model)
    print(f"Loading ML model for {model_label}...")
    try:
        ml_processor = MLProcessor(
            asset=asset,
            model_dir=MODEL_DIR,
            confidence_threshold=0.60,
            model_suffix=model_suffix,
        )
    except FileNotFoundError as e:
        print(f"ML model not found for {model_label}: {e} -- skipping")
        return None

    # Load LSTM model — required so the sweep's ensemble formula matches live's
    # XGB+LSTM blend (fusion term removed).
    print(f"Loading LSTM model for {model_label}...")
    try:
        lstm_processor = LSTMProcessor(
            asset=asset,
            model_dir=MODEL_DIR,
            model_suffix=model_suffix,
        )
    except FileNotFoundError as e:
        print(f"LSTM model not found for {model_label}: {e}")
        print("Sweep will blend XGB with lstm_p=0.5 default — results won't match live.")
        lstm_processor = None

    dm_range = f"dm {min_dm}-{max_dm}" if max_dm is not None else f"dm {min_dm}+"

    # Full 6-D sweep grid. (threshold x max_price x min_price x k_xgb x k_lstm x min_dm_inner)
    # MIN_DM_VALUES entries below the CLI floor are filtered during sweep, so the
    # banner count reflects effective combos.
    _eff_min_dm = [dm for dm in MIN_DM_VALUES if dm >= min_dm]
    total_combos = (
        len(THRESHOLDS) * len(MAX_PRICES) * len(MIN_PRICES) *
        len(DYNAMIC_K_XGB_VALUES) * len(DYNAMIC_K_LSTM_VALUES) * len(_eff_min_dm)
    )
    combo_desc = (f"{len(THRESHOLDS)} thr x {len(MAX_PRICES)} maxP x "
                  f"{len(MIN_PRICES)} minP x "
                  f"{len(DYNAMIC_K_XGB_VALUES)} k_xgb x "
                  f"{len(DYNAMIC_K_LSTM_VALUES)} k_lstm x "
                  f"{len(_eff_min_dm)} dm")

    print(f"\n{'='*70}")
    print(f"  PnL Sweep: {model_label}")
    print(f"  {total_combos} combos ({combo_desc})")
    print(f"  {len(windows)} windows (Kalshi-matched), {len(kalshi_windows)} Kalshi windows")
    print(f"  Kalshi dates: {kalshi_dates[0]} -> {kalshi_dates[-1]} ({len(kalshi_dates)} days)")
    print(f"  {dm_range}")
    print(f"  Config: balance=${config['initial_balance']:.2f} "
          f"max_contracts={config['max_contracts_per_trade']} "
          f"Kelly=[{config['min_price_cents']}c,{config['max_price_cents']}c]")
    print(f"{'='*70}")

    # Phase 1: Collect probabilities
    print(f"\nPhase 1: Collecting XGB + LSTM probabilities (2s grain)...")
    t0 = time.time()

    processors = build_processors()
    fusion_engine = SignalFusionEngine()
    simulator = BacktestSimulator(
        processors, fusion_engine,
        ml_processor=ml_processor,
        lstm_processor=lstm_processor,
        min_dm=min_dm,
        check_interval_seconds=2,  # Match live's 2s WS cadence (and 2s training grain)
        skip_fusion=True,          # formula is XGB+LSTM only, fusion unused
        batch_lstm=True,           # batched GPU inference; ~5-10x faster than per-call
        slicer=slicer,             # authoritative CB+KR merged buffer (live parity)
        slicer_lookback_seconds=900,
    )
    window_data = simulator.run_ticks_collect_probabilities(windows, fg_scores)

    t_collect = time.time() - t0
    total_checkpoints = sum(len(w["checkpoints"]) for w in window_data)
    print(f"  Collected {total_checkpoints} checkpoints across {len(window_data)} windows "
          f"in {t_collect:.1f}s")

    # Phase 2: Enrich with Kalshi data
    print(f"\nPhase 2: Enriching checkpoints with Kalshi prices...")
    t1 = time.time()

    enriched_windows = []
    matched_count = 0
    unmatched_count = 0

    for win in window_data:
        window_end = win.get("window_end")
        if window_end is None:
            unmatched_count += 1
            continue

        kw = kalshi_windows.get(window_end)
        if kw is None:
            unmatched_count += 1
            continue

        matched_count += 1

        # Enrich each checkpoint with Kalshi prices at signal_ts
        for cp in win["checkpoints"]:
            signal_ts = cp.get("signal_ts")
            if signal_ts is not None:
                kalshi_prices = get_kalshi_prices(kw, signal_ts)
                cp["kalshi"] = kalshi_prices
            else:
                cp["kalshi"] = None

        enriched_windows.append({
            "window_start": win["window_start"],
            "window_end": window_end,
            "actual_direction": win["actual_direction"],
            "outcome": kw.outcome,
            "checkpoints": win["checkpoints"],
        })

    t_enrich = time.time() - t1
    print(f"  Matched {matched_count} windows, {unmatched_count} unmatched "
          f"in {t_enrich:.2f}s")

    if matched_count == 0:
        print("No overlapping windows between Binance and Kalshi data!")
        return None

    # Phase 3: Sweep combos. Grid for XGB+LSTM confidence-weighted blend:
    # threshold x max_price x min_price x k_xgb x k_lstm x min_dm_inner
    # Note: min_dm from CLI is the floor; we ALSO sweep min_dm as a dimension.
    from itertools import product as _product
    combo_list = list(_product(
        THRESHOLDS, MAX_PRICES, MIN_PRICES, DYNAMIC_K_XGB_VALUES,
        DYNAMIC_K_LSTM_VALUES, MIN_DM_VALUES,
    ))
    total_combos = len(combo_list)
    print(f"\nPhase 3: Sweeping {total_combos} combos (threshold x maxP x minP x k_xgb x k_lstm x min_dm) by PnL...")
    t2 = time.time()

    all_results = []
    for thresh, max_p, min_p, k_xgb_val, k_lstm_val, combo_min_dm in combo_list:
        if combo_min_dm < min_dm:
            continue  # respect CLI --min-dm floor
        if min_p >= max_p:
            continue  # min must be below max; skip degenerate bands
        combo_config = {**config, "max_price_cents": max_p, "min_price_cents": min_p}
        result = sweep_combo_pnl(
            thresh, k_xgb_val, k_lstm_val, combo_min_dm,
            enriched_windows, combo_config, max_dm=max_dm,
        )
        all_results.append(result)

    t_sweep = time.time() - t2
    print(f"  Sweep completed in {t_sweep:.2f}s")
    print(f"\nTotal time: {t_collect + t_enrich + t_sweep:.1f}s")

    # Filter: need at least some trades
    MIN_TRADE_RATE = 2.0
    qualifying = [r for r in all_results if r["traded_pct"] >= MIN_TRADE_RATE and r["traded_count"] >= 5]

    if not qualifying:
        print(f"\nNo combos met the {MIN_TRADE_RATE:.0f}% trade rate filter!")
        qualifying = sorted(all_results, key=lambda r: r["total_pnl"], reverse=True)[:20]
    else:
        qualifying.sort(key=lambda r: r["total_pnl"], reverse=True)

    # Print top 20 by PnL
    print(f"\n{'='*140}")
    print(f"  {asset} -- Top 20 by TOTAL PnL -- {matched_count} Kalshi windows")
    print(f"{'='*140}")
    header = (
        f"{'Rank':>4} | {'PnL':>9} | {'AvgPnL':>8} | {'WinRate':>7} | "
        f"{'Traded':>6} | {'Trd%':>5} | {'Wins':>4} | {'Loss':>4} | "
        f"{'Final$':>8} | {'Thresh':>6} | {'MinP':>4} | {'MaxP':>4} | {'k_xgb':>5} | {'k_lstm':>6} | {'mDm':>3}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(qualifying[:20], 1):
        pnl_sign = "+" if r["total_pnl"] >= 0 else ""
        avg_sign = "+" if r["avg_pnl_per_trade"] >= 0 else ""
        print(
            f"{i:4d} | {pnl_sign}${r['total_pnl']:7.2f} | {avg_sign}${r['avg_pnl_per_trade']:6.3f} | "
            f"{r['win_rate']:6.1f}% | {r['traded_count']:6d} | {r['traded_pct']:4.1f}% | "
            f"{r['win_count']:4d} | {r['loss_count']:4d} | "
            f"${r['final_balance']:7.2f} | {r['threshold']:6.2f} | "
            f"{r.get('min_price_cents', ''):>4} | {r.get('max_price_cents', ''):>4} | "
            f"{r['k_xgb']:5.2f} | {r['k_lstm']:6.2f} | {r['min_dm']:>3}"
        )

    # Also show top 10 by win rate for reference (positive-PnL only — no losing combos)
    by_wr = sorted(
        [r for r in all_results if r["traded_count"] >= 10 and r["total_pnl"] > 0],
        key=lambda r: (-r["win_rate"], -r["total_pnl"]),
    )
    if by_wr:
        print(f"\n{'='*140}")
        print(f"  {asset} -- Top 10 by WIN RATE (positive PnL only, n >= 10)")
        print(f"{'='*140}")
        print(header)
        print("-" * len(header))
        for i, r in enumerate(by_wr[:10], 1):
            pnl_sign = "+" if r["total_pnl"] >= 0 else ""
            avg_sign = "+" if r["avg_pnl_per_trade"] >= 0 else ""
            print(
                f"{i:4d} | {pnl_sign}${r['total_pnl']:7.2f} | {avg_sign}${r['avg_pnl_per_trade']:6.3f} | "
                f"{r['win_rate']:6.1f}% | {r['traded_count']:6d} | {r['traded_pct']:4.1f}% | "
                f"{r['win_count']:4d} | {r['loss_count']:4d} | "
                f"${r['final_balance']:7.2f} | {r['threshold']:6.2f} | "
                f"{r.get('min_price_cents', ''):>4} | {r.get('max_price_cents', ''):>4} | "
                f"{r['k_xgb']:5.2f} | {r['k_lstm']:6.2f} | {r['min_dm']:>3}"
            )

    # Export CSV. output_subdir lets fold orchestrators write to
    # output/pnl_sweep/<subdir>/ instead of the shared dir.
    out_dir = OUTPUT_DIR / output_subdir if output_subdir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{asset}{model_suffix}_pnl_sweep.csv"
    fieldnames = [
        "threshold", "k_xgb", "k_lstm", "min_dm",
        "min_price_cents", "max_price_cents",
        "total_pnl", "win_rate", "traded_count", "traded_pct",
        "win_count", "loss_count", "avg_pnl_per_trade", "final_balance",
        "total_windows", "skipped_kelly", "skipped_no_ask",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: x["total_pnl"], reverse=True):
            writer.writerow(r)

    print(f"\nExported {len(all_results)} rows to {csv_path}")

    # Select best: highest WR among positive-PnL combos with meaningful sample (n >= 10)
    if not by_wr:
        print("\nNo positive-PnL combos with n >= 10; falling back to top PnL.")
        if not qualifying:
            return None
        best = qualifying[0]
    else:
        best = by_wr[0]

    print(f"\nSelected best (WR-first, positive PnL only): "
          f"threshold={best['threshold']:.2f} k_xgb={best['k_xgb']:.2f} "
          f"k_lstm={best['k_lstm']:.2f} min_dm={best['min_dm']} "
          f"minP={best.get('min_price_cents', '?')}c "
          f"maxP={best.get('max_price_cents', '?')}c "
          f"PnL=${best['total_pnl']:+.2f} "
          f"win_rate={best['win_rate']:.1f}% ({best['traded_count']} trades)")
    return best


def write_pnl_config(best_per_asset: dict[str, dict], min_dm: int, max_dm: int | None = None, config_key: str = "ensemble") -> None:
    """Write PnL-optimized ensemble params to config/trading.json.

    Only updates the specified config key; preserves all other config.
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    else:
        config = {"defaults": {}, "assets": {}}

    for asset, best in best_per_asset.items():
        if asset not in config.get("assets", {}):
            config.setdefault("assets", {})[asset] = {}

        ens_data = {
            "threshold": best["threshold"],
            "dynamic_k_xgb": best["k_xgb"],
            "dynamic_k_lstm": best["k_lstm"],
            "min_dm": best["min_dm"],
            "max_price_cents": best.get("max_price_cents"),
            "min_price_cents": best.get("min_price_cents"),
            "accuracy": round(best["win_rate"], 1),
            "net_correct": best["win_count"] - best["loss_count"],
            "traded_pct": best["traded_pct"],
            "pnl_sweep_total_pnl": best["total_pnl"],
            "pnl_sweep_source": "pnl_sweep",
            "pnl_sweep_formula": "2way_xgb_lstm_conf_weighted",
        }
        if max_dm is not None:
            ens_data["max_dm"] = max_dm

        config["assets"][asset][config_key] = ens_data

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    dm_label = f"dm {min_dm}-{max_dm}" if max_dm is not None else f"dm {min_dm}+"
    print(f"\n{'='*70}")
    print(f"  PnL sweep params ({config_key}, {dm_label}) written to {CONFIG_PATH}")
    print(f"{'='*70}")
    print(f"  {'Asset':<5} | {'Thresh':>6} | {'k_xgb':>5} | {'k_lstm':>6} | {'mDm':>3} | "
          f"{'PnL':>9} | {'WinRate':>7} | {'Trades':>6} | {'MinP':>4} | {'MaxP':>4}")
    print(f"  {'-'*83}")
    for asset, best in best_per_asset.items():
        pnl_sign = "+" if best["total_pnl"] >= 0 else ""
        max_p = best.get("max_price_cents", "")
        min_p = best.get("min_price_cents", "")
        print(f"  {asset:<5} | {best['threshold']:6.2f} | "
              f"{best.get('k_xgb', 0):5.2f} | {best.get('k_lstm', 0):6.2f} | "
              f"{best.get('min_dm', 0):>3} | "
              f"{pnl_sign}${best['total_pnl']:7.2f} | {best['win_rate']:6.1f}% | "
              f"{best['traded_count']:6d} | {min_p:>4} | {max_p:>4}")
    print()


def main():
    # Configure loguru
    logger.remove()
    log_path = PROJECT_ROOT / "logs" / "pnl_sweep.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), mode="w", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="PnL-based ensemble sweep using Kalshi prices")
    parser.add_argument(
        "--asset", required=True,
        help="Asset(s) to sweep, comma-separated (e.g. BTC or BTC,ETH,SOL,XRP)",
    )
    parser.add_argument("--min-dm", type=int, default=2, help="Minimum decision minute (default: 2)")
    parser.add_argument("--max-dm", type=int, default=None,
                        help="Maximum decision minute (e.g. 3 for dm 2-3 early model)")
    parser.add_argument("--model-suffix", type=str, default="",
                        help="Model filename suffix (e.g. '_early' -> {ASSET}_early_xgb.json)")
    parser.add_argument("--hours", type=int, default=None,
                        help="Only use the most recent N hours of Kalshi data (default: all)")
    parser.add_argument("--from-date", type=str, default=None,
                        help="Start date YYYY-MM-DD (inclusive). Use with --to-date for fold-based sweeps.")
    parser.add_argument("--to-date", type=str, default=None,
                        help="End date YYYY-MM-DD (inclusive).")
    parser.add_argument("--sequential", action="store_true",
                        help="Run assets sequentially instead of in parallel")
    parser.add_argument("--no-config-write", action="store_true",
                        help="Don't touch config/trading.json — just produce the per-asset "
                             "pnl_sweep CSVs. Used by fold-based orchestrators.")
    parser.add_argument("--output-subdir", type=str, default=None,
                        help="Write sweep CSVs to output/pnl_sweep/<subdir>/ instead of "
                             "output/pnl_sweep/ so fold runs don't overwrite each other.")
    parser.add_argument("--balance", type=float, default=None,
                        help="Override initial_balance from config. Fold orchestrators "
                             "pass a large value (e.g. 10000) so position sizing isn't "
                             "starved at live's $25/asset — otherwise combos deplete "
                             "balance in 3-4 trades and subsequent sizes drop to zero.")
    args = parser.parse_args()


    from datetime import date as _date
    from_date = _date.fromisoformat(args.from_date) if args.from_date else None
    to_date = _date.fromisoformat(args.to_date) if args.to_date else None

    # Determine config key: suffix maps to "ensemble{suffix}" so staging runs
    # like --model-suffix _align2s write to "ensemble_align2s" not "ensemble".
    config_key = f"ensemble{args.model_suffix}" if args.model_suffix else "ensemble"

    assets = [a.strip().upper() for a in args.asset.split(",")]
    best_per_asset: dict[str, dict] = {}

    if len(assets) > 1 and not args.sequential:
        # Parallel execution: one process per asset
        print(f"\nRunning {len(assets)} assets in parallel...")
        if args.hours:
            print(f"  Using last {args.hours}h of Kalshi data")
        t_total = time.time()
        with ProcessPoolExecutor(max_workers=len(assets)) as executor:
            futures = {
                executor.submit(
                    run_asset, asset, args.min_dm, args.hours,
                    args.max_dm, args.model_suffix, from_date, to_date,
                    args.output_subdir, args.balance,
                ): asset
                for asset in assets
            }
            for future in as_completed(futures):
                asset = futures[future]
                try:
                    best = future.result()
                    if best is not None:
                        best_per_asset[asset] = best
                except Exception as e:
                    print(f"\n[ERROR] {asset} sweep failed: {e}")
        print(f"\nAll assets completed in {time.time() - t_total:.1f}s")
    else:
        # Sequential execution (single asset or --sequential flag)
        for asset in assets:
            best = run_asset(
                asset, args.min_dm, args.hours, max_dm=args.max_dm,
                model_suffix=args.model_suffix, from_date=from_date, to_date=to_date,
                output_subdir=args.output_subdir, balance_override=args.balance,
            )
            if best is not None:
                best_per_asset[asset] = best

    # Write config only if we have results AND caller didn't suppress
    if args.no_config_write:
        print("\n--no-config-write: skipping config/trading.json update.")
    elif best_per_asset:
        write_pnl_config(best_per_asset, args.min_dm, max_dm=args.max_dm, config_key=config_key)
    else:
        print("\nNo assets had sufficient Kalshi data for PnL sweep.")
        print("Config NOT updated. Run the bot to collect more data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
