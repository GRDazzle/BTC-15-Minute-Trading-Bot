"""
Generate training data for XGBoost meta-model.

Replays tick windows (same as backtester) but dumps the full feature vector
at every 10-second decision point, giving ~60x more training samples than
windows alone.

Usage:
  python scripts/generate_training_data.py --asset BTC
  python scripts/generate_training_data.py --asset BTC --days 30
  python scripts/generate_training_data.py --asset BTC,ETH,SOL,XRP --days 30
  python scripts/generate_training_data.py --asset XRP --days 30 --min-move 0.0001
"""
import argparse
import csv
import sys
from collections import deque
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader_ticks import (
    TickWindow,
    load_aggtrades_multi,
    generate_tick_windows,
    resample_ticks,
)
from ml.features import FEATURE_NAMES, extract_features, build_tick_index

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "training_data"


def extract_window_features(
    window: TickWindow,
    price_history: deque,
    tick_buffer: deque,
    persistent_raw_buffer: deque | None = None,
) -> list[dict]:
    """Replay one tick window, extracting features at every 10s checkpoint.

    Returns list of feature dicts, each with 'label' (1=BULLISH, 0=BEARISH).
    """
    resample_ms = 250

    # Feed warmup ticks
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

    # Use persistent buffer if provided, otherwise local
    if persistent_raw_buffer is not None:
        raw_tick_buffer = persistent_raw_buffer
    else:
        raw_tick_buffer = deque(maxlen=500000)
    for tick in window.ticks_before:
        raw_tick_buffer.append({
            "ts": tick.ts,
            "price": tick.price,
            "qty": tick.qty,
            "is_buyer": tick.is_buyer,
        })

    # Pre-resample decision zone
    if window.ticks_during:
        decision_bars = resample_ticks(
            window.ticks_during,
            window.ticks_during[0].ts,
            window.ticks_during[-1].ts + timedelta(seconds=1),
            interval_ms=resample_ms,
        )
    else:
        decision_bars = []

    # Add raw ticks from decision zone to raw_tick_buffer as we go
    raw_tick_idx = 0

    label = 1 if window.actual_direction == "BULLISH" else 0
    rows = []

    decision_start = window.window_start + timedelta(minutes=5)
    check_interval = timedelta(seconds=10)
    current_check = decision_start + check_interval
    bar_idx = 0

    # Build sorted index ONCE from persistent buffer (O(n log n))
    # Then incrementally insert new ticks (O(log n) per tick)
    import bisect
    from ml.features import _ensure_utc
    ts_idx = []
    sorted_raw = []
    for t in raw_tick_buffer:
        ts = _ensure_utc(t["ts"])
        idx = bisect.bisect_left(ts_idx, ts)
        ts_idx.insert(idx, ts)
        sorted_raw.insert(idx, t)

    while current_check < window.window_end:
        next_check = current_check + check_interval

        # Feed resampled bars up to current_check
        while bar_idx < len(decision_bars) and decision_bars[bar_idx]["ts"] < current_check:
            bar = decision_bars[bar_idx]
            price_history.append(Decimal(str(bar["price"])))
            tick_buffer.append(bar)
            bar_idx += 1

        # Feed raw ticks up to current_check — add to both buffer AND sorted index
        while raw_tick_idx < len(window.ticks_during) and window.ticks_during[raw_tick_idx].ts < current_check:
            tick = window.ticks_during[raw_tick_idx]
            tick_dict = {
                "ts": tick.ts,
                "price": tick.price,
                "qty": tick.qty,
                "is_buyer": tick.is_buyer,
            }
            raw_tick_buffer.append(tick_dict)
            # Incremental insert into sorted index (O(log n))
            ts = _ensure_utc(tick.ts)
            idx = bisect.bisect_left(ts_idx, ts)
            ts_idx.insert(idx, ts)
            sorted_raw.insert(idx, tick_dict)
            raw_tick_idx += 1

        if len(price_history) < 20:
            current_check = next_check
            continue

        current_price = float(price_history[-1])

        # Compute decision minute
        elapsed_s = (current_check - window.window_start).total_seconds()
        dm = int((elapsed_s - 300) / 60)

        # Use pre-built sorted index for O(log n) lookups
        feats = extract_features(
            tick_buffer=None,
            price_history=list(price_history),
            current_price=current_price,
            timestamp=current_check,
            decision_minute=dm,
            window_open_price=window.price_open,
            ts_index=ts_idx,
            sorted_ticks=sorted_raw,
        )
        feats["label"] = label
        feats["window_start"] = window.window_start.isoformat()
        feats["hour_utc"] = float(current_check.hour)
        rows.append(feats)

        current_check = next_check

    # Feed remaining during ticks so next window has them in persistent buffer
    while raw_tick_idx < len(window.ticks_during):
        tick = window.ticks_during[raw_tick_idx]
        raw_tick_buffer.append({
            "ts": tick.ts,
            "price": tick.price,
            "qty": tick.qty,
            "is_buyer": tick.is_buyer,
        })
        raw_tick_idx += 1

    return rows


def generate_for_asset(asset: str, days: int | None, min_move: float = 0.0, day_filter: str = "all") -> None:
    """Generate training data CSV for one asset."""
    ticks = load_aggtrades_multi(DATA_DIR, asset, days=days)
    if not ticks:
        logger.error("No aggTrades data found for {}. Run download_binance_aggtrades.py first.", asset)
        return

    windows = generate_tick_windows(ticks)
    if not windows:
        logger.error("No valid tick windows for {}", asset)
        return

    # Filter windows with tiny price moves (noisy labels)
    if min_move > 0:
        before = len(windows)
        windows = [
            w for w in windows
            if w.price_open != 0 and abs(w.price_close - w.price_open) / w.price_open >= min_move
        ]
        filtered = before - len(windows)
        logger.info("Filtered {} windows with move < {:.4%} ({} -> {})", filtered, min_move, before, len(windows))

    # Filter by day of week
    if day_filter == "weekday":
        before = len(windows)
        windows = [w for w in windows if w.window_start.weekday() < 5]
        logger.info("Day filter weekday: {} -> {} windows", before, len(windows))
    elif day_filter == "weekend":
        before = len(windows)
        windows = [w for w in windows if w.window_start.weekday() >= 5]
        logger.info("Day filter weekend: {} -> {} windows", before, len(windows))

    logger.info("Generating training data for {} ({} windows)", asset, len(windows))

    price_history: deque = deque(maxlen=200)
    tick_buffer: deque = deque(maxlen=300)
    persistent_raw_buffer: deque = deque(maxlen=500000)

    all_rows: list[dict] = []
    log_interval = max(1, len(windows) // 20)

    for i, window in enumerate(windows):
        if i % log_interval == 0:
            logger.info("Processing window {}/{}", i + 1, len(windows))

        rows = extract_window_features(
            window, price_history, tick_buffer,
            persistent_raw_buffer=persistent_raw_buffer,
        )
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No feature rows generated for {}", asset)
        return

    # Write CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    day_suffix = f"_{day_filter}" if day_filter != "all" else ""
    out_path = OUTPUT_DIR / f"{asset.upper()}_features{day_suffix}.csv"

    columns = FEATURE_NAMES + ["label", "window_start", "hour_utc"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info("Wrote {} rows to {}", len(all_rows), out_path)

    # Quick stats
    bullish = sum(1 for r in all_rows if r["label"] == 1)
    bearish = len(all_rows) - bullish
    unique_windows = len(set(r["window_start"] for r in all_rows))
    print(f"\n=== {asset} Training Data ===")
    print(f"Total rows:      {len(all_rows)}")
    print(f"Unique windows:  {unique_windows}")
    print(f"Rows/window avg: {len(all_rows) / unique_windows:.1f}")
    print(f"Label balance:   {bullish} BULLISH / {bearish} BEARISH ({bullish/len(all_rows)*100:.1f}%)")
    print(f"Output:          {out_path}")
    print()


def main():
    logger.remove()
    log_path = PROJECT_ROOT / "logs" / "generate_training_data.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), mode="w", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="Generate ML training data from tick replay")
    parser.add_argument(
        "--asset", required=True,
        help="Asset(s) to process, comma-separated (e.g. BTC or BTC,ETH,SOL,XRP)",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Limit to last N days of data (default: all available)",
    )
    parser.add_argument(
        "--min-move", type=float, default=0.0001,
        help="Min price move pct to include window (default: 0.0001 = 0.01%%). "
             "Windows with smaller moves have ambiguous labels and add noise.",
    )
    parser.add_argument(
        "--day-filter", choices=["all", "weekday", "weekend"], default="all",
        help="Filter windows by day of week (default: all)",
    )
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    for asset in assets:
        generate_for_asset(asset, args.days, min_move=args.min_move, day_filter=args.day_filter)


if __name__ == "__main__":
    main()
