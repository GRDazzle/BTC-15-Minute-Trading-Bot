"""
PnL-based ensemble sweep: find optimal (ml_weight, threshold) using real Kalshi prices.

Unlike the accuracy sweep (ensemble_sweep.py) which scores by signal correctness,
this sweep scores by dollar PnL -- factoring in entry prices, Kelly bands, position
sizing, fees, and settlement outcomes from actual Kalshi polling data.

Three phases:
  1. Collect ML + fusion probabilities in one backtest pass (reuses simulator)
  2. Enrich checkpoints with Kalshi bid/ask prices and outcomes
  3. Sweep weight/threshold combos scoring by total PnL

Requires Kalshi polling data in data/kalshi_polls/KX{ASSET}15M/.

Usage:
  python scripts/pnl_sweep.py --asset BTC
  python scripts/pnl_sweep.py --asset BTC,ETH,SOL,XRP --min-dm 2
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
OUTPUT_DIR = PROJECT_ROOT / "output" / "pnl_sweep"
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"

# Sweep grid: finer threshold resolution than accuracy sweep
ML_WEIGHTS = [round(w * 0.05, 2) for w in range(21)]  # 0.0, 0.05, ..., 1.0
THRESHOLDS = [round(0.55 + i * 0.01, 2) for i in range(16)]  # 0.55 to 0.70

# Kalshi fee per contract in cents
KALSHI_FEE_CENTS = 2

# Minimum days of Kalshi data required to run PnL sweep
MIN_KALSHI_DAYS = 3


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
    """Determine how many contracts to buy (matches backtest_kalshi_pnl.py)."""
    cost_per = (price_cents + KALSHI_FEE_CENTS) / 100.0
    if cost_per <= 0 or balance <= 0:
        return 0
    max_by_balance = int(balance / cost_per)
    scale = min(1.0, confidence)
    desired = max(1, int(max_by_balance * scale))
    return min(desired, max_contracts)


def sweep_combo_pnl(
    ml_weight: float,
    threshold: float,
    enriched_windows: list[dict],
    config: dict,
) -> dict:
    """Evaluate one (ml_weight, threshold) combo by dollar PnL.

    For each window, find the first checkpoint crossing the threshold
    (first-signal-wins), then simulate PnL using Kalshi prices.
    """
    fusion_weight = 1.0 - ml_weight
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

        for cp in win["checkpoints"]:
            if cp.get("kalshi") is None:
                continue

            ensemble_p = ml_weight * cp["ml_p"] + fusion_weight * cp["fusion_p"]

            if ensemble_p >= threshold:
                predicted = "BULLISH"
                confidence = ensemble_p
                side = "yes"
                entry_price = cp["kalshi"]["yes_ask"]
                break
            elif ensemble_p <= 1.0 - threshold:
                predicted = "BEARISH"
                confidence = 1.0 - ensemble_p
                side = "no"
                entry_price = cp["kalshi"]["no_ask"]
                break

        if predicted is None:
            continue

        # Validate entry price
        if entry_price <= 0 or entry_price >= 100:
            skipped_no_ask += 1
            continue

        # Kelly band filter
        if entry_price > max_price or entry_price < min_price:
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
        "ml_weight": ml_weight,
        "threshold": threshold,
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


def run_asset(asset: str, min_dm: int) -> dict | None:
    """Run PnL sweep for one asset. Returns best combo dict or None."""
    from core.strategy_brain.signal_processors.ml_processor import MLProcessor

    config = load_config(asset)

    # Load Kalshi data
    print(f"\nLoading Kalshi polling data for {asset}...")
    kalshi_windows = load_kalshi_windows(KALSHI_DATA_DIR, asset)
    if not kalshi_windows:
        print(f"No Kalshi data for {asset}. Start the bot to collect data.")
        return None

    # Check minimum days of data
    kalshi_dates = sorted(set(ct.date() for ct in kalshi_windows.keys()))
    if len(kalshi_dates) < MIN_KALSHI_DAYS:
        print(f"Only {len(kalshi_dates)} days of Kalshi data for {asset} "
              f"(need {MIN_KALSHI_DAYS}). Skipping PnL sweep.")
        return None

    # Load tick data
    print(f"Loading aggTrades data for {asset}...")
    ticks = load_aggtrades_multi(DATA_DIR, asset)
    if not ticks:
        print(f"No aggTrades data for {asset}. Run download_binance_aggtrades.py first.")
        return None

    windows = generate_tick_windows(ticks)
    if not windows:
        print(f"No valid tick windows for {asset}")
        return None

    fg_scores = load_fear_greed(FG_CSV) if FG_CSV.exists() else {}

    # Load ML model
    print(f"Loading ML model for {asset}...")
    try:
        ml_processor = MLProcessor(
            asset=asset,
            model_dir=MODEL_DIR,
            confidence_threshold=0.60,
        )
    except FileNotFoundError as e:
        print(f"ML model not found for {asset}: {e} -- skipping")
        return None

    total_combos = len(ML_WEIGHTS) * len(THRESHOLDS)
    print(f"\n{'='*70}")
    print(f"  PnL Sweep: {asset}")
    print(f"  {total_combos} combos ({len(ML_WEIGHTS)} weights x {len(THRESHOLDS)} thresholds)")
    print(f"  {len(windows)} Binance windows, {len(kalshi_windows)} Kalshi windows")
    print(f"  Kalshi dates: {kalshi_dates[0]} -> {kalshi_dates[-1]} ({len(kalshi_dates)} days)")
    print(f"  Config: balance=${config['initial_balance']:.2f} "
          f"max_contracts={config['max_contracts_per_trade']} "
          f"Kelly=[{config['min_price_cents']}c,{config['max_price_cents']}c]")
    print(f"{'='*70}")

    # Phase 1: Collect probabilities
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

    # Phase 3: Sweep combos with PnL scoring
    print(f"\nPhase 3: Sweeping {total_combos} combos by PnL...")
    t2 = time.time()

    all_results = []
    for ml_w in ML_WEIGHTS:
        for thresh in THRESHOLDS:
            result = sweep_combo_pnl(ml_w, thresh, enriched_windows, config)
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
    print(f"\n{'='*130}")
    print(f"  {asset} -- Top 20 by TOTAL PnL -- {matched_count} Kalshi windows")
    print(f"{'='*130}")
    header = (
        f"{'Rank':>4} | {'PnL':>9} | {'AvgPnL':>8} | {'WinRate':>7} | "
        f"{'Traded':>6} | {'Trd%':>5} | {'Wins':>4} | {'Loss':>4} | "
        f"{'Final$':>8} | {'ML_W':>5} | {'Thresh':>6}"
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
            f"${r['final_balance']:7.2f} | {r['ml_weight']:5.2f} | {r['threshold']:6.2f}"
        )

    # Also show top 10 by win rate for reference
    by_wr = sorted(
        [r for r in all_results if r["traded_count"] >= 5],
        key=lambda r: r["win_rate"], reverse=True,
    )
    if by_wr:
        print(f"\n{'='*130}")
        print(f"  {asset} -- Top 10 by WIN RATE (for reference)")
        print(f"{'='*130}")
        print(header)
        print("-" * len(header))
        for i, r in enumerate(by_wr[:10], 1):
            pnl_sign = "+" if r["total_pnl"] >= 0 else ""
            avg_sign = "+" if r["avg_pnl_per_trade"] >= 0 else ""
            print(
                f"{i:4d} | {pnl_sign}${r['total_pnl']:7.2f} | {avg_sign}${r['avg_pnl_per_trade']:6.3f} | "
                f"{r['win_rate']:6.1f}% | {r['traded_count']:6d} | {r['traded_pct']:4.1f}% | "
                f"{r['win_count']:4d} | {r['loss_count']:4d} | "
                f"${r['final_balance']:7.2f} | {r['ml_weight']:5.2f} | {r['threshold']:6.2f}"
            )

    # Export CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{asset}_pnl_sweep.csv"
    fieldnames = [
        "ml_weight", "threshold", "total_pnl", "win_rate",
        "traded_count", "traded_pct", "win_count", "loss_count",
        "avg_pnl_per_trade", "final_balance", "total_windows",
        "skipped_kelly", "skipped_no_ask",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: x["total_pnl"], reverse=True):
            writer.writerow(r)

    print(f"\nExported {len(all_results)} rows to {csv_path}")

    # Select best: highest total_pnl, tie-break on lower threshold
    if not qualifying:
        return None

    top_pnl = qualifying[0]["total_pnl"]
    # 5% tolerance from best PnL
    tolerance = max(0.50, abs(top_pnl) * 0.05)
    near_top = [r for r in qualifying if r["total_pnl"] >= top_pnl - tolerance]
    # Among near-top, pick the lowest threshold (more participation)
    near_top.sort(key=lambda r: (r["threshold"], -r["total_pnl"]))
    best = near_top[0]

    print(f"\nSelected best: threshold={best['threshold']:.2f} "
          f"ml_weight={best['ml_weight']:.2f} PnL=${best['total_pnl']:+.2f} "
          f"win_rate={best['win_rate']:.1f}% ({best['traded_count']} trades)")
    return best


def write_pnl_config(best_per_asset: dict[str, dict], min_dm: int) -> None:
    """Write PnL-optimized ensemble params to config/trading.json.

    Only updates ensemble section; preserves all other config.
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    else:
        config = {"defaults": {}, "assets": {}}

    for asset, best in best_per_asset.items():
        if asset not in config.get("assets", {}):
            config.setdefault("assets", {})[asset] = {}

        config["assets"][asset]["ensemble"] = {
            "ml_weight": best["ml_weight"],
            "threshold": best["threshold"],
            "min_dm": min_dm,
            "accuracy": round(best["win_rate"], 1),
            "net_correct": best["win_count"] - best["loss_count"],
            "traded_pct": best["traded_pct"],
            "pnl_sweep_total_pnl": best["total_pnl"],
            "pnl_sweep_source": "pnl_sweep",
        }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"\n{'='*70}")
    print(f"  PnL sweep params written to {CONFIG_PATH}")
    print(f"{'='*70}")
    print(f"  {'Asset':<5} | {'ML_W':>5} | {'Thresh':>6} | {'PnL':>9} | "
          f"{'WinRate':>7} | {'Trades':>6}")
    print(f"  {'-'*55}")
    for asset, best in best_per_asset.items():
        pnl_sign = "+" if best["total_pnl"] >= 0 else ""
        print(f"  {asset:<5} | {best['ml_weight']:5.2f} | {best['threshold']:6.2f} | "
              f"{pnl_sign}${best['total_pnl']:7.2f} | {best['win_rate']:6.1f}% | "
              f"{best['traded_count']:6d}")
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
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    best_per_asset: dict[str, dict] = {}

    for asset in assets:
        best = run_asset(asset, args.min_dm)
        if best is not None:
            best_per_asset[asset] = best

    # Write config only if we have results
    if best_per_asset:
        write_pnl_config(best_per_asset, args.min_dm)
    else:
        print("\nNo assets had sufficient Kalshi data for PnL sweep.")
        print("Config NOT updated. Run the bot to collect more data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
