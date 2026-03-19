"""
Parameter sweep for signal processor thresholds.

Runs a grid search over spike_threshold, spike_velocity, tick_vel_60s,
lookback, and min_confidence using multiprocessing.

Usage:
  python scripts/param_sweep.py --asset BTC
  python scripts/param_sweep.py --asset BTC --days 30
  python scripts/param_sweep.py --asset BTC --workers 6
"""
import argparse
import csv
import itertools
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtester.data_loader import load_binance_klines, load_fear_greed, generate_windows

ASSET_MAP = {
    "BTC": "BTCUSD_1m.csv",
    "ETH": "ETHUSD_1m.csv",
    "SOL": "SOLUSD_1m.csv",
    "XRP": "XRPUSD_1m.csv",
}

DATA_DIR = PROJECT_ROOT / "data" / "historical"
BINANCE_DIR = DATA_DIR / "binance"
FG_CSV = DATA_DIR / "fear_greed.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "sweep"

# --- Parameter grid ---
GRID = {
    "spike_threshold":  [0.001, 0.002, 0.003, 0.005, 0.008],
    "spike_velocity":   [0.0005, 0.001, 0.0015, 0.003],
    "tick_vel_60s":     [0.0005, 0.001, 0.002, 0.003],
    "lookback":         [10, 20, 30],
    "min_confidence":   [0.50, 0.55, 0.60],
}

# --- Decision-minute weights (bell curve centered on m5-m6) ---
MINUTE_WEIGHTS = {
    0: 0.20,
    1: 0.40,
    2: 0.60,
    3: 0.75,
    4: 0.95,
    5: 1.00,
    6: 1.00,
    7: 0.75,
    8: 0.60,
    9: 0.35,
}
# Minutes 10+ get 0.20
for _m in range(10, 15):
    MINUTE_WEIGHTS[_m] = 0.20

# --- Per-worker globals (loaded once via Pool initializer) ---
_windows = None
_fg_scores = None
_counter = None


def worker_init(csv_path: str, fg_path: str, counter):
    """Load data once per worker process."""
    global _windows, _fg_scores, _counter
    _counter = counter

    from backtester.data_loader import load_binance_klines, load_fear_greed, generate_windows
    from loguru import logger
    logger.remove()  # silence loguru in workers

    klines = load_binance_klines(Path(csv_path))

    # Apply days filter if encoded in path marker
    # (handled by main before passing pre-filtered CSV — we just load all)
    _fg_scores = load_fear_greed(Path(fg_path)) if Path(fg_path).exists() else {}
    _windows = generate_windows(klines)


def run_combo(params: dict) -> dict:
    """Run one parameter combination on pre-loaded windows. Returns summary row."""
    from backtester.simulator import BacktestSimulator
    from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
    from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
    from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine

    # Derive 30s velocity threshold from 60s (same ratio as default 0.001 / 0.0007)
    tick_vel_30s = params["tick_vel_60s"] * 0.7

    processors = [
        SpikeDetectionProcessor(
            spike_threshold=params["spike_threshold"],
            velocity_threshold=params["spike_velocity"],
            lookback_periods=params["lookback"],
            min_confidence=params["min_confidence"],
        ),
        TickVelocityProcessor(
            velocity_threshold_60s=params["tick_vel_60s"],
            velocity_threshold_30s=tick_vel_30s,
            min_ticks=5,
            min_confidence=params["min_confidence"],
        ),
    ]
    fusion_engine = SignalFusionEngine()
    simulator = BacktestSimulator(processors, fusion_engine)
    results = simulator.run(_windows, _fg_scores)

    # Compute summary metrics
    total = len(results)
    traded = [r for r in results if r.predicted_direction != "NONE"]
    traded_count = len(traded)
    correct_count = sum(1 for r in traded if r.correct)
    accuracy = correct_count / traded_count * 100 if traded_count else 0.0
    traded_pct = traded_count / total * 100 if total else 0.0
    avg_minute = (
        sum(r.decision_minute for r in traded) / traded_count
        if traded_count else 0.0
    )

    # Per-decision-minute accuracy
    minute_buckets: dict[int, list] = defaultdict(list)
    for r in traded:
        minute_buckets[r.decision_minute].append(r)

    per_minute: dict[int, dict] = {}
    for m in sorted(minute_buckets.keys()):
        rs = minute_buckets[m]
        mc = sum(1 for r in rs if r.correct)
        per_minute[m] = {"accuracy": mc / len(rs) * 100, "count": len(rs)}

    # Early (min 0-3) / Mid (min 4-7) / Late (min 8+) accuracy
    def _bucket_acc(lo, hi):
        rs = [r for r in traded if lo <= r.decision_minute <= hi]
        if not rs:
            return 0.0, 0
        c = sum(1 for r in rs if r.correct)
        return c / len(rs) * 100, len(rs)

    acc_early, n_early = _bucket_acc(0, 3)
    acc_mid, n_mid = _bucket_acc(4, 7)
    acc_late, n_late = _bucket_acc(8, 13)

    # Weighted accuracy (bell curve centered on m5-m6)
    w_correct = 0.0
    w_total = 0.0
    for r in traded:
        w = MINUTE_WEIGHTS.get(r.decision_minute, 0.30)
        w_total += w
        if r.correct:
            w_correct += w
    weighted_acc = w_correct / w_total * 100 if w_total else 0.0

    # Increment shared progress counter
    with _counter.get_lock():
        _counter.value += 1
        done = _counter.value
    if done % 50 == 0 or done == 1:
        print(f"  [{done}] combos completed...", flush=True)

    return {
        "spike_threshold": params["spike_threshold"],
        "spike_velocity": params["spike_velocity"],
        "tick_vel_60s": params["tick_vel_60s"],
        "lookback": params["lookback"],
        "min_confidence": params["min_confidence"],
        "accuracy": accuracy,
        "traded_pct": traded_pct,
        "traded_count": traded_count,
        "correct_count": correct_count,
        "total_windows": total,
        "avg_minute": avg_minute,
        "acc_early": acc_early,
        "n_early": n_early,
        "acc_mid": acc_mid,
        "n_mid": n_mid,
        "acc_late": acc_late,
        "n_late": n_late,
        "weighted_acc": weighted_acc,
        "per_minute": per_minute,
    }


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for signal thresholds")
    parser.add_argument("--asset", required=True, help="Asset to sweep (e.g. BTC)")
    parser.add_argument("--days", type=int, default=None, help="Limit to last N days")
    parser.add_argument("--workers", type=int, default=9, help="Worker processes (default: 9)")
    args = parser.parse_args()

    asset = args.asset.upper()
    csv_name = ASSET_MAP.get(asset)
    if csv_name is None:
        print(f"Unknown asset: {asset}. Available: {list(ASSET_MAP.keys())}")
        sys.exit(1)

    csv_path = BINANCE_DIR / csv_name
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    # If --days, pre-filter CSV and write to temp file so workers load less data
    actual_csv_path = csv_path
    if args.days is not None:
        from backtester.data_loader import load_binance_klines as _load
        klines = _load(csv_path)
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
        # Keep extra 200 klines before cutoff for buffer initialization
        filtered = []
        buffer_start = None
        for k in klines:
            if k.timestamp >= cutoff:
                filtered.append(k)
            elif buffer_start is None or (cutoff - k.timestamp).total_seconds() <= 200 * 60:
                filtered.append(k)

        tmp_csv = OUTPUT_DIR / f".tmp_{asset}_filtered.csv"
        tmp_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for k in filtered:
                writer.writerow([
                    k.timestamp.isoformat(),
                    str(k.open), str(k.high), str(k.low), str(k.close), str(k.volume),
                ])
        actual_csv_path = tmp_csv
        print(f"Filtered to last {args.days} days: {len(filtered)} klines (inc. buffer)")

    # Build parameter grid
    grid = list(itertools.product(
        GRID["spike_threshold"],
        GRID["spike_velocity"],
        GRID["tick_vel_60s"],
        GRID["lookback"],
        GRID["min_confidence"],
    ))
    param_dicts = [
        {
            "spike_threshold": st,
            "spike_velocity": sv,
            "tick_vel_60s": tv,
            "lookback": lb,
            "min_confidence": mc,
        }
        for st, sv, tv, lb, mc in grid
    ]

    n_workers = min(args.workers, len(param_dicts))
    est_time = len(param_dicts) * 10.5 / n_workers / 60

    print(f"\n{'='*60}")
    print(f"  Parameter Sweep: {asset}")
    print(f"  {len(param_dicts)} combos, {n_workers} workers, ~{est_time:.0f} min")
    print(f"{'='*60}\n")

    # Shared progress counter
    counter = mp.Value("i", 0)

    t0 = time.time()
    with mp.Pool(
        n_workers,
        initializer=worker_init,
        initargs=(str(actual_csv_path), str(FG_CSV), counter),
    ) as pool:
        all_results = pool.map(run_combo, param_dicts)
    elapsed = time.time() - t0

    # Clean up temp file
    if args.days is not None:
        tmp_csv = OUTPUT_DIR / f".tmp_{asset}_filtered.csv"
        if tmp_csv.exists():
            tmp_csv.unlink()

    print(f"\nCompleted {len(all_results)} combos in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")

    # --- Filter & sort ---
    MIN_TRADE_RATE = 5.0  # minimum 5% trade rate
    qualifying = [r for r in all_results if r["traded_pct"] >= MIN_TRADE_RATE]

    if not qualifying:
        print("\nNo combos met the minimum 5% trade rate filter!")
        print("Showing top 20 by weighted accuracy regardless:")
        qualifying = sorted(all_results, key=lambda r: r["weighted_acc"], reverse=True)[:20]
    else:
        qualifying.sort(key=lambda r: r["weighted_acc"], reverse=True)

    # --- Print top 20 ---
    print(f"\n{'='*140}")
    print(f"  Top 20 combos by WEIGHTED accuracy (min {MIN_TRADE_RATE:.0f}% trade rate) — {len(qualifying)} qualifying")
    print(f"  Weights: m0=.20  m1=.40  m2=.60  m3=.75  m4=.95  m5=1.0  m6=1.0  m7=.75  m8=.60  m9=.35  m10+=.20")
    print(f"{'='*140}")
    header = (
        f"{'Rank':>4} | {'WtdAcc':>7} | {'RawAcc':>7} | {'Traded%':>7} | {'Count':>5} | "
        f"{'AvgMin':>6} | "
        f"{'Early':>7} | {'Mid':>7} | {'Late':>7} | "
        f"{'spike_th':>8} | {'spike_vel':>9} | "
        f"{'tick_vel':>8} | {'lb':>3} | {'conf':>5}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(qualifying[:20], 1):
        early_s = f"{r['acc_early']:5.1f}%" if r['n_early'] else "   -  "
        mid_s = f"{r['acc_mid']:5.1f}%" if r['n_mid'] else "   -  "
        late_s = f"{r['acc_late']:5.1f}%" if r['n_late'] else "   -  "
        print(
            f"{i:4d} | {r['weighted_acc']:6.1f}% | {r['accuracy']:6.1f}% | {r['traded_pct']:6.1f}% | "
            f"{r['traded_count']:5d} | {r['avg_minute']:6.1f} | "
            f"{early_s:>7} | {mid_s:>7} | {late_s:>7} | "
            f"{r['spike_threshold']:8.4f} | {r['spike_velocity']:9.4f} | "
            f"{r['tick_vel_60s']:8.4f} | {r['lookback']:3d} | "
            f"{r['min_confidence']:5.2f}"
        )

    # --- Per-minute detail for top 10 ---
    print(f"\n{'='*130}")
    print(f"  Accuracy by decision minute — Top 10 combos")
    print(f"{'='*130}")
    # Header: Rank | params summary | min0 | min1 | ... | min9
    min_cols = list(range(10))  # minutes 0-9 (where most decisions happen)
    hdr = f"{'Rank':>4} | {'Params':^40} |"
    for m in min_cols:
        hdr += f" {'m'+str(m):>6} |"
    print(hdr)
    print("-" * len(hdr))

    for i, r in enumerate(qualifying[:10], 1):
        params_str = (
            f"st={r['spike_threshold']:.3f} sv={r['spike_velocity']:.4f} "
            f"tv={r['tick_vel_60s']:.4f} lb={r['lookback']} c={r['min_confidence']:.2f}"
        )
        row = f"{i:4d} | {params_str:^40} |"
        pm = r["per_minute"]
        for m in min_cols:
            if m in pm and pm[m]["count"] >= 3:
                row += f" {pm[m]['accuracy']:5.1f}% |"
            elif m in pm:
                row += f" {pm[m]['accuracy']:4.0f}%* |"  # * = low sample
            else:
                row += f"     - |"
        print(row)

    print("  (* = fewer than 3 trades at that minute)")

    # --- Per-minute detail for top 10 by LATE accuracy ---
    late_qualifying = [r for r in qualifying if r["n_late"] >= 5]
    if late_qualifying:
        late_qualifying.sort(key=lambda r: r["acc_late"], reverse=True)
        print(f"\n{'='*130}")
        print(f"  Accuracy by decision minute — Top 10 by LATE (min 8+) accuracy")
        print(f"{'='*130}")
        hdr2 = f"{'Rank':>4} | {'Params':^40} |"
        for m in min_cols:
            hdr2 += f" {'m'+str(m):>6} |"
        print(hdr2)
        print("-" * len(hdr2))

        for i, r in enumerate(late_qualifying[:10], 1):
            params_str = (
                f"st={r['spike_threshold']:.3f} sv={r['spike_velocity']:.4f} "
                f"tv={r['tick_vel_60s']:.4f} lb={r['lookback']} c={r['min_confidence']:.2f}"
            )
            row = f"{i:4d} | {params_str:^40} |"
            pm = r["per_minute"]
            for m in min_cols:
                if m in pm and pm[m]["count"] >= 3:
                    row += f" {pm[m]['accuracy']:5.1f}% |"
                elif m in pm:
                    row += f" {pm[m]['accuracy']:4.0f}%* |"
                else:
                    row += f"     - |"
            print(row)

        print("  (* = fewer than 3 trades at that minute)")

    # --- Export full CSV ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = OUTPUT_DIR / f"{asset}_sweep_results.csv"
    with open(csv_out, "w", newline="") as f:
        fieldnames = [
            "spike_threshold", "spike_velocity", "tick_vel_60s",
            "lookback", "min_confidence",
            "accuracy", "traded_pct", "traded_count", "correct_count",
            "total_windows", "avg_minute",
            "weighted_acc",
            "acc_early", "n_early", "acc_mid", "n_mid", "acc_late", "n_late",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        # Sort by weighted accuracy descending for the CSV too
        for r in sorted(all_results, key=lambda x: x["weighted_acc"], reverse=True):
            writer.writerow(r)

    print(f"\nExported all {len(all_results)} rows to {csv_out}")

    # --- Sanity check: find current params ---
    print(f"\n--- Sanity check: current params (0.003, 0.0015, 0.001, 20, 0.55) ---")
    for r in all_results:
        if (
            r["spike_threshold"] == 0.003
            and r["spike_velocity"] == 0.0015
            and r["tick_vel_60s"] == 0.001
            and r["lookback"] == 20
            and r["min_confidence"] == 0.55
        ):
            print(
                f"  Accuracy: {r['accuracy']:.1f}% | "
                f"Traded: {r['traded_pct']:.1f}% ({r['traded_count']}/{r['total_windows']}) | "
                f"Avg min: {r['avg_minute']:.1f}"
            )
            break
    else:
        print("  Not found in grid (check if values are included)")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.freeze_support()
    main()
