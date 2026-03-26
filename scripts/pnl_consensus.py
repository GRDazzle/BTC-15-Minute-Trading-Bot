"""
PnL backtest for the ML consensus signal using real Kalshi prices.

Instead of using a single ML prediction, collects ML P(BULLISH) over a rolling
window (e.g. 180s, 300s) and uses the mean as the signal. Evaluates dollar PnL
using actual Kalshi bid/ask prices and settlement outcomes.

Sweeps:
  - Consensus windows: 180s, 300s (configurable)
  - Thresholds: 0.55 to 0.75
  - Max prices: 60-95 cents

Usage:
  python scripts/pnl_consensus.py --asset BTC
  python scripts/pnl_consensus.py --asset BTC,ETH,SOL,XRP --min-dm 2
  python scripts/pnl_consensus.py --asset BTC --hours 12
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader import load_fear_greed
from backtester.data_loader_ticks import load_aggtrades_multi, generate_tick_windows
from backtester.data_loader_kalshi import load_kalshi_windows, get_kalshi_prices
from backtester.simulator import BacktestSimulator

from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades"
KALSHI_DATA_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
FG_CSV = PROJECT_ROOT / "data" / "historical" / "fear_greed.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pnl_consensus"
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"

# Consensus windows to sweep (seconds)
CONSENSUS_WINDOWS = [180, 300]

# Thresholds to sweep
THRESHOLDS = [0.55, 0.57, 0.60, 0.63, 0.65, 0.70, 0.75]

# Max price sweep values (cents)
MAX_PRICES = [60, 65, 70, 75, 80, 85, 90, 95]

# Checkpoint interval (matches simulator)
CHECKPOINT_INTERVAL_S = 10

# Kalshi fee per contract in cents
KALSHI_FEE_CENTS = 2

MIN_KALSHI_DAYS = 3


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
    scale = min(1.0, confidence * confidence)
    desired = max(1, int(max_by_balance * scale))
    return min(desired, max_contracts)


def sweep_consensus_pnl(
    consensus_seconds: int,
    threshold: float,
    max_price_cents: int,
    enriched_windows: list[dict],
    config: dict,
    min_dm: int = 0,
    max_dm: int = 9,
) -> dict:
    """Evaluate one (consensus_seconds, threshold, max_price) combo by dollar PnL.

    Collects ML predictions over a rolling window, computes mean P(BULLISH),
    and trades when it crosses the threshold. Uses Kalshi prices at the moment
    of the trade signal.
    """
    n_checkpoints = max(1, consensus_seconds // CHECKPOINT_INTERVAL_S)

    balance = config["initial_balance"]
    max_contracts = config["max_contracts_per_trade"]
    min_price = config["min_price_cents"]

    total_windows = len(enriched_windows)
    traded_count = 0
    win_count = 0
    total_pnl = 0.0
    skipped_kelly = 0
    trade_dms = []

    for win in enriched_windows:
        outcome = win["outcome"]
        if not outcome:
            continue

        # Filter and collect checkpoints in dm range
        valid_cps = [
            cp for cp in win["checkpoints"]
            if min_dm <= cp["dm"] <= max_dm and cp.get("kalshi") is not None
        ]
        if not valid_cps:
            continue

        ml_p_history = []
        traded_this = False

        for cp in valid_cps:
            ml_p_history.append(cp["ml_p"])

            if len(ml_p_history) < n_checkpoints:
                continue

            # Compute consensus
            recent = ml_p_history[-n_checkpoints:]
            consensus_p = sum(recent) / len(recent)

            # Decision
            if consensus_p >= threshold:
                direction = "BULLISH"
                confidence = consensus_p
                side = "yes"
                entry_price = cp["kalshi"]["yes_ask"]
            elif consensus_p <= 1.0 - threshold:
                direction = "BEARISH"
                confidence = 1.0 - consensus_p
                side = "no"
                entry_price = cp["kalshi"]["no_ask"]
            else:
                continue

            # Price filters
            if entry_price <= 0 or entry_price >= 100:
                continue
            if entry_price > max_price_cents or entry_price < min_price:
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
            trade_dms.append(cp["dm"])
            if won:
                win_count += 1
            traded_this = True
            break  # First signal wins

    win_rate = win_count / traded_count * 100 if traded_count else 0.0
    avg_pnl = total_pnl / traded_count if traded_count else 0.0
    traded_pct = traded_count / total_windows * 100 if total_windows else 0.0
    avg_dm = sum(trade_dms) / len(trade_dms) if trade_dms else 0.0

    return {
        "consensus_seconds": consensus_seconds,
        "threshold": threshold,
        "max_price_cents": max_price_cents,
        "total_pnl": round(total_pnl, 4),
        "win_rate": round(win_rate, 1),
        "traded_count": traded_count,
        "traded_pct": round(traded_pct, 1),
        "win_count": win_count,
        "loss_count": traded_count - win_count,
        "avg_pnl_per_trade": round(avg_pnl, 4),
        "final_balance": round(balance, 2),
        "avg_dm": round(avg_dm, 1),
        "total_windows": total_windows,
        "skipped_kelly": skipped_kelly,
    }


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


def run_asset(
    asset: str,
    min_dm: int,
    hours: int | None = None,
    max_dm: int = 9,
    model_suffix: str = "",
) -> dict | None:
    """Run consensus PnL backtest for one asset."""
    logger.remove()
    model_label = f"{asset}{model_suffix}" if model_suffix else asset
    log_path = PROJECT_ROOT / "logs" / f"pnl_consensus_{model_label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), mode="w", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(sys.stderr, level="WARNING")

    from core.strategy_brain.signal_processors.ml_processor import MLProcessor

    config = load_config(asset)

    # Load Kalshi data
    print(f"\nLoading Kalshi polling data for {asset}...")
    kalshi_windows = load_kalshi_windows(KALSHI_DATA_DIR, asset)
    if not kalshi_windows:
        print(f"No Kalshi data for {asset}.")
        return None

    from datetime import datetime as dt, date, timedelta as td, timezone
    if hours is not None:
        cutoff = dt.now(tz=timezone.utc) - td(hours=hours)
        before = len(kalshi_windows)
        kalshi_windows = {k: v for k, v in kalshi_windows.items() if k >= cutoff}
        print(f"  Filtered to last {hours}h: {before} -> {len(kalshi_windows)} Kalshi windows")
        if not kalshi_windows:
            print(f"No Kalshi data in the last {hours}h for {asset}.")
            return None

    kalshi_dates = sorted(set(ct.date() for ct in kalshi_windows.keys()))
    if hours is None and len(kalshi_dates) < MIN_KALSHI_DAYS:
        print(f"Only {len(kalshi_dates)} days of Kalshi data for {asset} "
              f"(need {MIN_KALSHI_DAYS}). Skipping.")
        return None
    if len(kalshi_windows) < 20:
        print(f"Only {len(kalshi_windows)} Kalshi windows for {asset} (need 20). Skipping.")
        return None

    kalshi_min_date = kalshi_dates[0]
    today = date.today()
    days_back = (today - kalshi_min_date).days + 2

    # Load tick data
    print(f"Loading aggTrades for {asset} (last {days_back} days)...")
    ticks = load_aggtrades_multi(DATA_DIR, asset, days=days_back)
    if not ticks:
        print(f"No aggTrades data for {asset}.")
        return None

    windows = generate_tick_windows(ticks, min_warmup_ticks=1, min_during_ticks=1)
    if not windows:
        print(f"No tick windows for {asset}")
        return None

    # Pre-filter to Kalshi-matched windows
    kalshi_close_times = set(kalshi_windows.keys())
    all_count = len(windows)
    windows = [w for w in windows if w.window_end in kalshi_close_times]
    print(f"  Pre-filtered: {all_count} -> {len(windows)} windows (Kalshi-matched)")

    if not windows:
        print(f"No overlapping Binance/Kalshi data for {asset}")
        return None

    fg_scores = load_fear_greed(FG_CSV) if FG_CSV.exists() else {}

    # Load ML model
    print(f"Loading ML model for {model_label}...")
    try:
        ml_processor = MLProcessor(
            asset=asset, model_dir=MODEL_DIR,
            confidence_threshold=0.60, model_suffix=model_suffix,
        )
    except FileNotFoundError as e:
        print(f"ML model not found for {model_label}: {e}")
        return None

    total_combos = len(CONSENSUS_WINDOWS) * len(THRESHOLDS) * len(MAX_PRICES)
    dm_range = f"dm {min_dm}-{max_dm}"

    print(f"\n{'='*70}")
    print(f"  Consensus PnL Backtest: {model_label}")
    print(f"  {total_combos} combos ({len(CONSENSUS_WINDOWS)} windows x "
          f"{len(THRESHOLDS)} thresholds x {len(MAX_PRICES)} max_prices)")
    print(f"  {len(windows)} Kalshi-matched windows")
    print(f"  Kalshi dates: {kalshi_dates[0]} -> {kalshi_dates[-1]} ({len(kalshi_dates)} days)")
    print(f"  {dm_range}")
    print(f"{'='*70}")

    # Phase 1: Collect probabilities (one pass)
    print(f"\nPhase 1: Collecting ML + fusion probabilities...")
    t0 = time.time()

    processors = build_processors()
    fusion_engine = SignalFusionEngine()
    simulator = BacktestSimulator(
        processors, fusion_engine,
        ml_processor=ml_processor,
        min_dm=min_dm,
    )
    window_data = simulator.run_ticks_collect_probabilities(windows, fg_scores)

    t_collect = time.time() - t0
    total_checkpoints = sum(len(w["checkpoints"]) for w in window_data)
    print(f"  Collected {total_checkpoints} checkpoints in {t_collect:.1f}s")

    # Phase 2: Enrich with Kalshi data
    print(f"\nPhase 2: Enriching with Kalshi prices...")
    t1 = time.time()

    enriched_windows = []
    matched = 0

    for win in window_data:
        window_end = win.get("window_end")
        if window_end is None:
            continue
        kw = kalshi_windows.get(window_end)
        if kw is None:
            continue

        matched += 1
        for cp in win["checkpoints"]:
            signal_ts = cp.get("signal_ts")
            if signal_ts is not None:
                cp["kalshi"] = get_kalshi_prices(kw, signal_ts)
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
    print(f"  Matched {matched} windows in {t_enrich:.2f}s")

    if matched == 0:
        print("No overlapping windows!")
        return None

    # Phase 3: Sweep consensus combos
    print(f"\nPhase 3: Sweeping {total_combos} combos...")
    t2 = time.time()

    all_results = []
    for cons_s in CONSENSUS_WINDOWS:
        for thresh in THRESHOLDS:
            for max_p in MAX_PRICES:
                result = sweep_consensus_pnl(
                    cons_s, thresh, max_p,
                    enriched_windows, config,
                    min_dm=min_dm, max_dm=max_dm,
                )
                all_results.append(result)

    t_sweep = time.time() - t2
    print(f"  Sweep completed in {t_sweep:.2f}s")
    print(f"\nTotal time: {t_collect + t_enrich + t_sweep:.1f}s")

    # Display results per consensus window
    for cons_s in CONSENSUS_WINDOWS:
        subset = [r for r in all_results if r["consensus_seconds"] == cons_s]
        qualifying = sorted(
            [r for r in subset if r["traded_count"] >= 5],
            key=lambda r: r["total_pnl"], reverse=True,
        )

        print(f"\n{'='*130}")
        print(f"  {asset} -- Consensus {cons_s}s -- Top 20 by PnL -- {matched} windows")
        print(f"{'='*130}")
        header = (
            f"{'Rank':>4} | {'PnL':>9} | {'AvgPnL':>8} | {'WinRate':>7} | "
            f"{'Traded':>6} | {'Trd%':>5} | {'W':>4} | {'L':>4} | "
            f"{'AvgDM':>5} | {'MaxP':>4} | {'Thresh':>6}"
        )
        print(header)
        print("-" * len(header))

        for i, r in enumerate(qualifying[:20], 1):
            pnl_s = "+" if r["total_pnl"] >= 0 else ""
            avg_s = "+" if r["avg_pnl_per_trade"] >= 0 else ""
            print(
                f"{i:4d} | {pnl_s}${r['total_pnl']:7.2f} | {avg_s}${r['avg_pnl_per_trade']:6.3f} | "
                f"{r['win_rate']:6.1f}% | {r['traded_count']:6d} | {r['traded_pct']:4.1f}% | "
                f"{r['win_count']:4d} | {r['loss_count']:4d} | "
                f"{r['avg_dm']:5.1f} | {r['max_price_cents']:4d} | {r['threshold']:6.2f}"
            )

    # Overall best
    all_qualifying = [r for r in all_results if r["traded_count"] >= 5]
    if all_qualifying:
        best_pnl = max(all_qualifying, key=lambda r: r["total_pnl"])
        best_wr = max(all_qualifying, key=lambda r: r["win_rate"])

        print(f"\n--- Overall Best by PnL ---")
        print(f"  consensus={best_pnl['consensus_seconds']}s thresh={best_pnl['threshold']:.2f} "
              f"max_price={best_pnl['max_price_cents']}c "
              f"PnL=${best_pnl['total_pnl']:+.2f} win_rate={best_pnl['win_rate']:.1f}% "
              f"trades={best_pnl['traded_count']} avg_dm={best_pnl['avg_dm']:.1f}")

        print(f"\n--- Overall Best by Win Rate (>= 5 trades) ---")
        print(f"  consensus={best_wr['consensus_seconds']}s thresh={best_wr['threshold']:.2f} "
              f"max_price={best_wr['max_price_cents']}c "
              f"PnL=${best_wr['total_pnl']:+.2f} win_rate={best_wr['win_rate']:.1f}% "
              f"trades={best_wr['traded_count']} avg_dm={best_wr['avg_dm']:.1f}")

    # Export CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{asset}{model_suffix}_pnl_consensus.csv"
    fieldnames = [
        "consensus_seconds", "threshold", "max_price_cents",
        "total_pnl", "win_rate", "traded_count", "traded_pct",
        "win_count", "loss_count", "avg_pnl_per_trade",
        "final_balance", "avg_dm", "total_windows", "skipped_kelly",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: x["total_pnl"], reverse=True):
            writer.writerow(r)
    print(f"\nExported {len(all_results)} rows to {csv_path}")

    return best_pnl if all_qualifying else None


def main():
    logger.remove()
    log_path = PROJECT_ROOT / "logs" / "pnl_consensus.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), mode="w", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="Consensus PnL backtest with Kalshi data")
    parser.add_argument("--asset", required=True, help="Asset(s), comma-separated")
    parser.add_argument("--min-dm", type=int, default=0, help="Min decision minute (default: 0)")
    parser.add_argument("--max-dm", type=int, default=9, help="Max decision minute (default: 9)")
    parser.add_argument("--model-suffix", default="", help="Model suffix (e.g. _early)")
    parser.add_argument("--hours", type=int, default=None, help="Limit to last N hours of Kalshi data")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    for asset in assets:
        run_asset(
            asset, args.min_dm,
            hours=args.hours,
            max_dm=args.max_dm,
            model_suffix=args.model_suffix,
        )


if __name__ == "__main__":
    main()
