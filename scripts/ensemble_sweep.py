"""
Ensemble weight sweep: find optimal ML vs Fusion blend.

Pre-computes ML and fusion probabilities in ONE backtest pass, then sweeps
ml_weight (0.0 - 1.0, step 0.05) x threshold (6 values) = 126 combos
as pure arithmetic over the cached data. Runs in seconds, not hours.

Usage:
  python scripts/ensemble_sweep.py --asset XRP --days 30
  python scripts/ensemble_sweep.py --asset BTC,ETH,SOL,XRP --days 30
  python scripts/ensemble_sweep.py --asset BTC --days 30 --min-dm 2
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
from backtester.simulator import BacktestSimulator

from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades"
FG_CSV = PROJECT_ROOT / "data" / "historical" / "fear_greed.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ensemble_sweep"
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"

# Sweep grid
ML_WEIGHTS = [round(w * 0.05, 2) for w in range(21)]  # 0.0, 0.05, ..., 1.0
THRESHOLDS = [0.55, 0.58, 0.60, 0.62, 0.65, 0.70]


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


def sweep_combo(ml_weight: float, threshold: float, window_data: list[dict]) -> dict:
    """Evaluate one (ml_weight, threshold) combo over pre-computed probabilities.

    For each window, find the first checkpoint where the blended probability
    crosses the threshold (mimicking first-signal-wins behavior).
    """
    fusion_weight = 1.0 - ml_weight
    total = len(window_data)
    traded_count = 0
    correct_count = 0
    sum_confidence = 0.0
    sum_dm = 0.0

    for win in window_data:
        actual = win["actual_direction"]
        predicted = None
        confidence = 0.0
        dm = -1

        for cp in win["checkpoints"]:
            ensemble_p = ml_weight * cp["ml_p"] + fusion_weight * cp["fusion_p"]

            if ensemble_p >= threshold:
                predicted = "BULLISH"
                confidence = ensemble_p
                dm = cp["dm"]
                break
            elif ensemble_p <= 1.0 - threshold:
                predicted = "BEARISH"
                confidence = 1.0 - ensemble_p
                dm = cp["dm"]
                break

        if predicted is not None:
            traded_count += 1
            if predicted == actual:
                correct_count += 1
            sum_confidence += confidence
            sum_dm += dm

    wrong_count = traded_count - correct_count
    net_correct = correct_count - wrong_count
    accuracy = correct_count / traded_count * 100 if traded_count else 0.0
    traded_pct = traded_count / total * 100 if total else 0.0
    avg_confidence = sum_confidence / traded_count if traded_count else 0.0
    avg_dm = sum_dm / traded_count if traded_count else 0.0

    return {
        "ml_weight": ml_weight,
        "threshold": threshold,
        "accuracy": accuracy,
        "traded_pct": traded_pct,
        "traded_count": traded_count,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "net_correct": net_correct,
        "total_windows": total,
        "avg_confidence": avg_confidence,
        "avg_dm": avg_dm,
    }


def run_asset(asset: str, days: int | None, min_dm: int) -> dict | None:
    """Run ensemble sweep for one asset. Returns best combo dict or None."""
    from core.strategy_brain.signal_processors.ml_processor import MLProcessor

    # Load tick data
    print(f"\nLoading aggTrades data for {asset}...")
    ticks = load_aggtrades_multi(DATA_DIR, asset, days=days)
    if not ticks:
        print(f"No aggTrades data found for {asset}. Run download_binance_aggtrades.py first.")
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

    days_str = f"last {days}d" if days else "all"
    total_combos = len(ML_WEIGHTS) * len(THRESHOLDS)
    print(f"\n{'='*70}")
    print(f"  Ensemble Sweep: {asset}")
    print(f"  {total_combos} combos ({len(ML_WEIGHTS)} weights x {len(THRESHOLDS)} thresholds)")
    print(f"  {len(windows)} windows ({days_str}), min_dm={min_dm}")
    print(f"{'='*70}")

    # Phase 1: Collect probabilities in one backtest pass
    print(f"\nPhase 1: Collecting ML + fusion probabilities (one pass)...")
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

    # Phase 2: Sweep weight/threshold combos (pure arithmetic)
    print(f"\nPhase 2: Sweeping {total_combos} combos...")
    t1 = time.time()

    all_results = []
    for ml_w in ML_WEIGHTS:
        for thresh in THRESHOLDS:
            result = sweep_combo(ml_w, thresh, window_data)
            all_results.append(result)

    t_sweep = time.time() - t1
    print(f"  Sweep completed in {t_sweep:.2f}s")
    print(f"\nTotal time: {t_collect + t_sweep:.1f}s")

    # Filter and sort
    MIN_TRADE_RATE = 5.0
    qualifying = [r for r in all_results if r["traded_pct"] >= MIN_TRADE_RATE]

    if not qualifying:
        print(f"\nNo combos met the {MIN_TRADE_RATE:.0f}% trade rate filter!")
        print("Showing top 20 by net correct regardless:")
        qualifying = sorted(all_results, key=lambda r: r["net_correct"], reverse=True)[:20]
    else:
        qualifying.sort(key=lambda r: r["net_correct"], reverse=True)

    # Print top 20 by net correct
    print(f"\n{'='*120}")
    print(f"  {asset} -- Top 20 by NET CORRECT (correct - wrong) -- min {MIN_TRADE_RATE:.0f}% trade rate, "
          f"{len(qualifying)} qualifying")
    print(f"{'='*120}")
    header = (
        f"{'Rank':>4} | {'Net':>5} | {'Correct':>7} | {'Wrong':>5} | "
        f"{'Acc':>7} | {'Traded%':>7} | {'Count':>5} | "
        f"{'AvgConf':>7} | {'AvgDM':>5} | {'ML_W':>5} | {'Thresh':>6}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(qualifying[:20], 1):
        print(
            f"{i:4d} | {r['net_correct']:5d} | {r['correct_count']:7d} | {r['wrong_count']:5d} | "
            f"{r['accuracy']:6.1f}% | {r['traded_pct']:6.1f}% | "
            f"{r['traded_count']:5d} | {r['avg_confidence']:6.3f} | "
            f"{r['avg_dm']:5.1f} | {r['ml_weight']:5.2f} | {r['threshold']:6.2f}"
        )

    # Also show top 10 by accuracy for reference
    by_acc = sorted(
        [r for r in all_results if r["traded_pct"] >= MIN_TRADE_RATE],
        key=lambda r: r["accuracy"], reverse=True,
    )
    if by_acc:
        print(f"\n{'='*120}")
        print(f"  {asset} -- Top 10 by ACCURACY (for reference)")
        print(f"{'='*120}")
        print(header)
        print("-" * len(header))
        for i, r in enumerate(by_acc[:10], 1):
            print(
                f"{i:4d} | {r['net_correct']:5d} | {r['correct_count']:7d} | {r['wrong_count']:5d} | "
                f"{r['accuracy']:6.1f}% | {r['traded_pct']:6.1f}% | "
                f"{r['traded_count']:5d} | {r['avg_confidence']:6.3f} | "
                f"{r['avg_dm']:5.1f} | {r['ml_weight']:5.2f} | {r['threshold']:6.2f}"
            )

    # Sanity checks
    print(f"\n--- {asset} sanity checks ---")
    for r in all_results:
        if r["ml_weight"] == 1.0 and r["threshold"] == 0.60:
            print(f"  ml_weight=1.0, thresh=0.60 (pure ML):  "
                  f"acc={r['accuracy']:.1f}%  traded={r['traded_pct']:.1f}%  n={r['traded_count']}")
        if r["ml_weight"] == 0.0 and r["threshold"] == 0.60:
            print(f"  ml_weight=0.0, thresh=0.60 (pure fusion): "
                  f"acc={r['accuracy']:.1f}%  traded={r['traded_pct']:.1f}%  n={r['traded_count']}")

    # Export CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{asset}_ensemble_sweep.csv"
    fieldnames = [
        "ml_weight", "threshold", "accuracy", "traded_pct",
        "traded_count", "correct_count", "wrong_count", "net_correct",
        "total_windows", "avg_confidence", "avg_dm",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: x["net_correct"], reverse=True):
            writer.writerow(r)

    print(f"\nExported {len(all_results)} rows to {csv_path}")

    # Select best combo: among top results by net_correct, favor the lowest
    # threshold. Lower thresholds produce more trades (higher participation)
    # and are more robust -- validated via Kalshi PnL backtest sweep showing
    # 0.64 outperforms 0.65 on real dollar PnL.
    if not qualifying:
        return None

    top_net = qualifying[0]["net_correct"]
    # Allow 2% tolerance from the absolute best net_correct
    tolerance = max(1, int(top_net * 0.02))
    near_top = [r for r in qualifying if r["net_correct"] >= top_net - tolerance]
    # Among near-top, pick the lowest threshold; if tied, highest net_correct
    near_top.sort(key=lambda r: (r["threshold"], -r["net_correct"]))
    best = near_top[0]
    print(f"\nSelected best: threshold={best['threshold']:.2f} "
          f"ml_weight={best['ml_weight']:.2f} acc={best['accuracy']:.1f}% "
          f"net={best['net_correct']} (from {len(near_top)} near-top combos)")
    return best


KALSHI_FEE_CENTS = 2  # per-contract fee


def half_kelly_band(accuracy: float, fee_cents: int = KALSHI_FEE_CENTS) -> tuple[int, int]:
    """Compute entry price band using half-Kelly criterion.

    For a YES bet at price c (cents):
        net_odds b = (100 - c) / c
        kelly f* = p - q * c / (100 - c)
        f* > 0 when c < p * 100

    For a NO bet at price (100 - c):
        f* > 0 when (100 - c) < p * 100, i.e. c > (1 - p) * 100

    We subtract the fee from the breakeven to stay in positive-EV territory.

    Returns (min_price_cents, max_price_cents).
    """
    p = accuracy / 100.0
    # YES side breakeven: price where Kelly = 0
    max_price = int(p * 100) - fee_cents
    # NO side breakeven: minimum price where buying NO still has edge
    min_price = int((1.0 - p) * 100) + fee_cents + 1

    # Clamp to sane bounds
    max_price = max(50, min(95, max_price))
    min_price = max(5, min(50, min_price))

    return min_price, max_price


def half_kelly_fraction(accuracy: float, price_cents: int) -> float:
    """Calculate half-Kelly bet fraction for a given accuracy and entry price.

    Returns fraction of bankroll to wager (0.0 - 1.0).
    """
    p = accuracy / 100.0
    q = 1.0 - p
    c = price_cents / 100.0
    if c <= 0 or c >= 1:
        return 0.0
    kelly = p - q * c / (1.0 - c)
    return max(0.0, kelly / 2.0)


def write_ensemble_config(best_per_asset: dict[str, dict], min_dm: int) -> None:
    """Write best ensemble params + half-Kelly bands per asset into config/trading.json."""
    # Load existing config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    else:
        config = {"defaults": {}, "assets": {}}

    for asset, best in best_per_asset.items():
        if asset not in config.get("assets", {}):
            config.setdefault("assets", {})[asset] = {}

        acc = best["accuracy"]
        min_price, max_price = half_kelly_band(acc)

        config["assets"][asset]["ensemble"] = {
            "ml_weight": best["ml_weight"],
            "threshold": best["threshold"],
            "min_dm": min_dm,
            "accuracy": round(acc, 1),
            "net_correct": best["net_correct"],
            "traded_pct": round(best["traded_pct"], 1),
        }
        # Entry bands (min/max_price_cents) are set manually in trading.json
        # and validated against real Kalshi PnL data -- do NOT overwrite them here.

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"\n{'='*70}")
    print(f"  Ensemble params + half-Kelly bands written to {CONFIG_PATH}")
    print(f"{'='*70}")
    print(f"  {'Asset':<5} | {'ML_W':>5} | {'Thresh':>6} | {'Acc':>6} | "
          f"{'Net':>5} | {'Band':>10} | {'HK@30c':>6} | {'HK@50c':>6} | {'HK@70c':>6}")
    print(f"  {'-'*68}")
    for asset, best in best_per_asset.items():
        acc = best["accuracy"]
        min_p, max_p = half_kelly_band(acc)
        hk30 = half_kelly_fraction(acc, 30)
        hk50 = half_kelly_fraction(acc, 50)
        hk70 = half_kelly_fraction(acc, 70)
        print(f"  {asset:<5} | {best['ml_weight']:5.2f} | {best['threshold']:6.2f} | "
              f"{acc:5.1f}% | {best['net_correct']:5d} | "
              f"{min_p:2d}c-{max_p:2d}c | {hk30:5.1%} | {hk50:5.1%} | {hk70:5.1%}")
    print()


def main():
    # Configure loguru
    logger.remove()
    log_path = PROJECT_ROOT / "logs" / "ensemble_sweep.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), mode="w", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="Ensemble weight sweep (ML + Fusion)")
    parser.add_argument(
        "--asset", required=True,
        help="Asset(s) to sweep, comma-separated (e.g. BTC or BTC,ETH,SOL,XRP)",
    )
    parser.add_argument("--days", type=int, default=None, help="Limit to last N days of data")
    parser.add_argument("--min-dm", type=int, default=2, help="Minimum decision minute (default: 2)")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    best_per_asset: dict[str, dict] = {}
    for asset in assets:
        best = run_asset(asset, args.days, args.min_dm)
        if best is not None:
            best_per_asset[asset] = best

    # Write best params to config/trading.json
    if best_per_asset:
        write_ensemble_config(best_per_asset, args.min_dm)


if __name__ == "__main__":
    main()
