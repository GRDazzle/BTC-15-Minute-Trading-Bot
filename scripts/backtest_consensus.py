"""
Backtest the ML consensus signal: aggregate multiple ML predictions over a
rolling window and use the consensus percentage as the trading signal.

Hypothesis: collecting N predictions over 30-60s and using the consensus
direction produces a more robust signal than a single point-in-time prediction.

Approach:
  1. Replay tick windows, collecting ML P(BULLISH) at every 10s checkpoint
  2. At each checkpoint, compute rolling consensus over last N seconds
  3. Use consensus % (mean P(BULLISH)) as the signal with a threshold
  4. Report accuracy at different consensus windows and thresholds

Usage:
  python scripts/backtest_consensus.py --asset BTC
  python scripts/backtest_consensus.py --asset BTC --days 30 --min-dm 0
  python scripts/backtest_consensus.py --asset BTC,ETH,SOL,XRP --days 30
"""
import argparse
import csv
import sys
from collections import deque
from datetime import timedelta
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
MODEL_DIR = PROJECT_ROOT / "models"
FG_CSV = PROJECT_ROOT / "data" / "fear_greed.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "consensus_backtest"

# Consensus window sizes to test (seconds)
CONSENSUS_WINDOWS = [30, 60, 90, 120, 180, 300]

# Thresholds to test
THRESHOLDS = [0.52, 0.55, 0.57, 0.60, 0.63, 0.65, 0.70, 0.75]

# Checkpoint interval matches simulator (10 seconds)
CHECKPOINT_INTERVAL_S = 10


def evaluate_consensus(
    window_data: list[dict],
    consensus_seconds: int,
    threshold: float,
    min_dm: int = 0,
    max_dm: int = 9,
) -> dict:
    """Evaluate consensus signal over pre-collected probability data.

    For each window, collects ML predictions over a rolling window of
    `consensus_seconds`, computes the mean P(BULLISH), and uses it as
    the signal when it crosses the threshold.

    Uses "first signal wins" -- the first checkpoint where the consensus
    crosses the threshold determines the trade direction for that window.

    Args:
        window_data: Output from run_ticks_collect_probabilities()
        consensus_seconds: Rolling window size in seconds
        threshold: Consensus threshold for trading
        min_dm: Minimum decision minute to consider
        max_dm: Maximum decision minute to consider

    Returns:
        Dict with accuracy, traded_count, correct_count, etc.
    """
    n_checkpoints = max(1, consensus_seconds // CHECKPOINT_INTERVAL_S)

    total_windows = 0
    traded = 0
    correct = 0
    trade_dms = []

    for wd in window_data:
        checkpoints = wd["checkpoints"]
        if not checkpoints:
            continue

        total_windows += 1

        # Filter checkpoints by dm range
        valid_cps = [cp for cp in checkpoints if min_dm <= cp["dm"] <= max_dm]
        if not valid_cps:
            continue

        # Collect ml_p values and compute rolling consensus
        ml_p_history = []
        traded_this_window = False

        for cp in valid_cps:
            ml_p_history.append(cp["ml_p"])

            # Need at least n_checkpoints to form consensus
            if len(ml_p_history) < n_checkpoints:
                continue

            # Compute consensus: mean of last n_checkpoints
            recent = ml_p_history[-n_checkpoints:]
            consensus_p = sum(recent) / len(recent)

            # Decision based on consensus
            if consensus_p >= threshold:
                direction = "BULLISH"
            elif consensus_p <= 1.0 - threshold:
                direction = "BEARISH"
            else:
                continue

            # First signal wins
            traded += 1
            trade_dms.append(cp["dm"])
            if direction == wd["actual_direction"]:
                correct += 1
            traded_this_window = True
            break

    accuracy = (correct / traded * 100) if traded > 0 else 0.0
    traded_pct = (traded / total_windows * 100) if total_windows > 0 else 0.0
    avg_dm = (sum(trade_dms) / len(trade_dms)) if trade_dms else 0.0
    net_correct = correct - (traded - correct)  # wins - losses

    return {
        "consensus_seconds": consensus_seconds,
        "threshold": threshold,
        "accuracy": round(accuracy, 1),
        "traded_count": traded,
        "traded_pct": round(traded_pct, 1),
        "correct": correct,
        "wrong": traded - correct,
        "net_correct": net_correct,
        "avg_dm": round(avg_dm, 1),
        "total_windows": total_windows,
    }


def evaluate_single_prediction(
    window_data: list[dict],
    threshold: float,
    min_dm: int = 0,
    max_dm: int = 9,
) -> dict:
    """Baseline: single prediction (first checkpoint that crosses threshold).

    This is essentially the current system behavior -- no consensus.
    """
    total_windows = 0
    traded = 0
    correct = 0

    for wd in window_data:
        checkpoints = wd["checkpoints"]
        if not checkpoints:
            continue
        total_windows += 1

        for cp in checkpoints:
            if cp["dm"] < min_dm or cp["dm"] > max_dm:
                continue
            ml_p = cp["ml_p"]
            if ml_p >= threshold:
                direction = "BULLISH"
            elif ml_p <= 1.0 - threshold:
                direction = "BEARISH"
            else:
                continue

            traded += 1
            if direction == wd["actual_direction"]:
                correct += 1
            break

    accuracy = (correct / traded * 100) if traded > 0 else 0.0
    traded_pct = (traded / total_windows * 100) if total_windows > 0 else 0.0
    net_correct = correct - (traded - correct)

    return {
        "consensus_seconds": 0,
        "threshold": threshold,
        "accuracy": round(accuracy, 1),
        "traded_count": traded,
        "traded_pct": round(traded_pct, 1),
        "correct": correct,
        "wrong": traded - correct,
        "net_correct": net_correct,
        "avg_dm": 0.0,
        "total_windows": total_windows,
    }


def run_asset(
    asset: str,
    days: int | None,
    min_dm: int,
    max_dm: int = 9,
    model_suffix: str = "",
) -> None:
    """Run consensus backtest for one asset."""
    # Load tick data
    ticks = load_aggtrades_multi(DATA_DIR, asset, days=days)
    if not ticks:
        print(f"No aggTrades data for {asset}")
        return

    windows = generate_tick_windows(ticks)
    if not windows:
        print(f"No tick windows for {asset}")
        return

    fg_scores = {}
    if FG_CSV.exists():
        fg_scores = load_fear_greed(FG_CSV)

    # Load ML model
    try:
        from core.strategy_brain.signal_processors.ml_processor import MLProcessor
        ml_proc = MLProcessor(
            asset=asset,
            model_dir=MODEL_DIR,
            confidence_threshold=0.60,
            model_suffix=model_suffix,
        )
    except FileNotFoundError:
        print(f"No model found for {asset}{model_suffix}")
        return

    # Build signal processors for fusion
    spike_proc = SpikeDetectionProcessor(
        spike_threshold=0.003, lookback_periods=20, min_confidence=0.55,
    )
    tickvel_proc = TickVelocityProcessor(
        velocity_threshold_60s=0.001, velocity_threshold_30s=0.0007,
        min_ticks=5, min_confidence=0.55,
    )
    fusion = SignalFusionEngine()

    sim = BacktestSimulator(
        processors=[spike_proc, tickvel_proc],
        fusion_engine=fusion,
        ml_processor=ml_proc,
        min_dm=min_dm,
    )

    print(f"\n{'='*70}")
    print(f"  CONSENSUS BACKTEST: {asset} ({len(windows)} windows, dm {min_dm}-{max_dm})")
    print(f"{'='*70}")

    # Phase 1: Collect all ML probabilities in one pass
    print(f"\nPhase 1: Collecting ML probabilities...")
    window_data = sim.run_ticks_collect_probabilities(windows, fg_scores)
    total_checkpoints = sum(len(wd["checkpoints"]) for wd in window_data)
    print(f"  Collected {total_checkpoints} checkpoints across {len(window_data)} windows")

    # Phase 2: Evaluate consensus at different window sizes and thresholds
    print(f"\nPhase 2: Evaluating consensus signals...")

    all_results = []

    # Baseline: single prediction (no consensus)
    for thresh in THRESHOLDS:
        result = evaluate_single_prediction(window_data, thresh, min_dm=min_dm, max_dm=max_dm)
        result["type"] = "single"
        all_results.append(result)

    # Consensus at various window sizes
    for consensus_s in CONSENSUS_WINDOWS:
        for thresh in THRESHOLDS:
            result = evaluate_consensus(
                window_data, consensus_s, thresh,
                min_dm=min_dm, max_dm=max_dm,
            )
            result["type"] = "consensus"
            all_results.append(result)

    # Display results
    print(f"\n--- Baseline: Single ML Prediction (no consensus) ---")
    print(f"{'Thresh':>7} {'Acc%':>6} {'Traded':>7} {'Trd%':>6} {'W':>4} {'L':>4} {'Net':>5}")
    print(f"{'-'*7} {'-'*6} {'-'*7} {'-'*6} {'-'*4} {'-'*4} {'-'*5}")
    for r in all_results:
        if r["type"] != "single":
            continue
        print(
            f"{r['threshold']:>7.2f} {r['accuracy']:>6.1f} {r['traded_count']:>7} "
            f"{r['traded_pct']:>6.1f} {r['correct']:>4} {r['wrong']:>4} {r['net_correct']:>5}"
        )

    for consensus_s in CONSENSUS_WINDOWS:
        print(f"\n--- Consensus Window: {consensus_s}s ({consensus_s // CHECKPOINT_INTERVAL_S} checkpoints) ---")
        print(f"{'Thresh':>7} {'Acc%':>6} {'Traded':>7} {'Trd%':>6} {'W':>4} {'L':>4} {'Net':>5} {'AvgDM':>6}")
        print(f"{'-'*7} {'-'*6} {'-'*7} {'-'*6} {'-'*4} {'-'*4} {'-'*5} {'-'*6}")
        for r in all_results:
            if r["type"] != "consensus" or r["consensus_seconds"] != consensus_s:
                continue
            print(
                f"{r['threshold']:>7.2f} {r['accuracy']:>6.1f} {r['traded_count']:>7} "
                f"{r['traded_pct']:>6.1f} {r['correct']:>4} {r['wrong']:>4} "
                f"{r['net_correct']:>5} {r['avg_dm']:>6.1f}"
            )

    # Find best consensus result by net_correct (min 5% traded)
    viable = [r for r in all_results if r["traded_pct"] >= 5.0 and r["traded_count"] >= 20]
    if viable:
        best_net = max(viable, key=lambda r: r["net_correct"])
        best_acc = max(viable, key=lambda r: r["accuracy"])

        print(f"\n--- Best by Net Correct (>= 5% traded, >= 20 trades) ---")
        _print_result(best_net)

        print(f"\n--- Best by Accuracy (>= 5% traded, >= 20 trades) ---")
        _print_result(best_acc)

        # Compare best consensus vs best single
        best_single = max(
            [r for r in viable if r["type"] == "single"],
            key=lambda r: r["net_correct"],
            default=None,
        )
        best_consensus = max(
            [r for r in viable if r["type"] == "consensus"],
            key=lambda r: r["net_correct"],
            default=None,
        )
        if best_single and best_consensus:
            print(f"\n--- Consensus vs Single Prediction ---")
            print(f"  Single:    net={best_single['net_correct']:+d}, acc={best_single['accuracy']:.1f}%, "
                  f"trades={best_single['traded_count']}, thresh={best_single['threshold']:.2f}")
            print(f"  Consensus: net={best_consensus['net_correct']:+d}, acc={best_consensus['accuracy']:.1f}%, "
                  f"trades={best_consensus['traded_count']}, window={best_consensus['consensus_seconds']}s, "
                  f"thresh={best_consensus['threshold']:.2f}")
            delta = best_consensus["net_correct"] - best_single["net_correct"]
            print(f"  Delta:     {delta:+d} net correct")

    # Export CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{asset}{model_suffix}_consensus_backtest.csv"
    fieldnames = [
        "type", "consensus_seconds", "threshold", "accuracy",
        "traded_count", "traded_pct", "correct", "wrong",
        "net_correct", "avg_dm", "total_windows",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nExported {len(all_results)} results to {csv_path}")


def _print_result(r: dict) -> None:
    label = f"consensus {r['consensus_seconds']}s" if r["type"] == "consensus" else "single"
    print(
        f"  {label}: thresh={r['threshold']:.2f}, acc={r['accuracy']:.1f}%, "
        f"traded={r['traded_count']} ({r['traded_pct']:.1f}%), "
        f"W={r['correct']} L={r['wrong']} net={r['net_correct']:+d}"
    )


def main():
    logger.remove()
    log_path = PROJECT_ROOT / "logs" / "consensus_backtest.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), mode="w", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="Backtest ML consensus signal")
    parser.add_argument("--asset", required=True, help="Asset(s), comma-separated")
    parser.add_argument("--days", type=int, default=None, help="Limit to last N days")
    parser.add_argument("--min-dm", type=int, default=0, help="Min decision minute (default: 0)")
    parser.add_argument("--max-dm", type=int, default=9, help="Max decision minute (default: 9)")
    parser.add_argument("--model-suffix", default="", help="Model suffix (e.g. _early)")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    for asset in assets:
        run_asset(
            asset, args.days,
            min_dm=args.min_dm,
            max_dm=args.max_dm,
            model_suffix=args.model_suffix,
        )


if __name__ == "__main__":
    main()
