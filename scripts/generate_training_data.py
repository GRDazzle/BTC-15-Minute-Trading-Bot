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
import bisect
import csv
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader_ticks import (
    Tick,
    TickWindow,
    load_aggtrades_multi,
    generate_tick_windows,
    resample_ticks,
)
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from ml.features import FEATURE_NAMES, extract_features

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "training_data"


def build_processors():
    """Instantiate signal processors with sweep-validated parameters."""
    tickvel = TickVelocityProcessor(
        velocity_threshold_60s=0.001,
        velocity_threshold_30s=0.0007,
        min_ticks=5,
        min_confidence=0.55,
    )
    return tickvel


def _build_btc_timestamp_index(btc_ticks: list[Tick]) -> list[datetime]:
    """Pre-build sorted timestamp list for bisect lookups."""
    return [t.ts for t in btc_ticks]


def _bisect_closest_price(btc_ticks: list[Tick], ts_index: list[datetime], target: datetime, max_diff_s: float = 30) -> Optional[float]:
    """Find BTC tick price closest to target timestamp using bisect. O(log n)."""
    idx = bisect.bisect_left(ts_index, target)
    best_price = None
    best_diff = float("inf")
    # Check idx-1 and idx (the two candidates around the insertion point)
    for i in (idx - 1, idx):
        if 0 <= i < len(btc_ticks):
            diff = abs((ts_index[i] - target).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_price = btc_ticks[i].price
    if best_diff > max_diff_s:
        return None
    return best_price


def _compute_btc_velocity_at(btc_ticks: list[Tick], ts_index: list[datetime], timestamp: datetime) -> float:
    """Compute BTC 60s velocity at a given timestamp. O(log n) via bisect."""
    if not btc_ticks:
        return 0.0
    price_now = _bisect_closest_price(btc_ticks, ts_index, timestamp)
    price_60s = _bisect_closest_price(btc_ticks, ts_index, timestamp - timedelta(seconds=60))
    if price_now is None or price_60s is None or price_60s == 0:
        return 0.0
    return (price_now - price_60s) / price_60s


def _get_btc_ticks_for_window(
    btc_ticks: list[Tick], ts_index: list[datetime], window_start: datetime, window_end: datetime
) -> tuple[list[Tick], list[datetime]]:
    """Extract BTC ticks covering the window period (with 2min buffer). O(log n) slice."""
    buf_start = window_start - timedelta(minutes=2)
    buf_end = window_end + timedelta(minutes=1)
    lo = bisect.bisect_left(ts_index, buf_start)
    hi = bisect.bisect_right(ts_index, buf_end)
    return btc_ticks[lo:hi], ts_index[lo:hi]


def extract_window_features(
    window: TickWindow,
    tickvel_proc: TickVelocityProcessor,
    price_history: deque,
    tick_buffer: deque,
    btc_ticks: Optional[list[Tick]] = None,
    btc_ts_index: Optional[list[datetime]] = None,
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

    # Enrich bars with volume data from raw ticks
    raw_tick_buffer: deque = deque(maxlen=1200)
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

    # Get BTC ticks covering this window (for cross-asset feature)
    window_btc_ticks = None
    window_btc_ts = None
    if btc_ticks is not None and btc_ts_index is not None:
        window_btc_ticks, window_btc_ts = _get_btc_ticks_for_window(
            btc_ticks, btc_ts_index, window.window_start, window.window_end
        )

    label = 1 if window.actual_direction == "BULLISH" else 0
    rows = []

    decision_start = window.window_start + timedelta(minutes=5)
    check_interval = timedelta(seconds=10)
    current_check = decision_start + check_interval
    bar_idx = 0

    while current_check < window.window_end:
        next_check = current_check + check_interval

        # Feed resampled bars up to current_check
        while bar_idx < len(decision_bars) and decision_bars[bar_idx]["ts"] < current_check:
            bar = decision_bars[bar_idx]
            price_history.append(Decimal(str(bar["price"])))
            tick_buffer.append(bar)
            bar_idx += 1

        # Feed raw ticks up to current_check for volume features
        while raw_tick_idx < len(window.ticks_during) and window.ticks_during[raw_tick_idx].ts < current_check:
            tick = window.ticks_during[raw_tick_idx]
            raw_tick_buffer.append({
                "ts": tick.ts,
                "price": tick.price,
                "qty": tick.qty,
                "is_buyer": tick.is_buyer,
            })
            raw_tick_idx += 1

        if len(price_history) < 20:
            current_check = next_check
            continue

        current_price = float(price_history[-1])

        # Compute decision minute
        elapsed_s = (current_check - window.window_start).total_seconds()
        dm = int((elapsed_s - 300) / 60)

        # Compute momentum for metadata
        if len(price_history) >= 6:
            prev = float(price_history[-6])
            momentum = (current_price - prev) / prev if prev != 0 else 0.0
        else:
            momentum = 0.0

        metadata = {
            "tick_buffer": list(tick_buffer),
            "spot_price": current_price,
            "momentum": momentum,
            "sentiment_score": 50,
        }

        # Run tickvel processor
        tickvel_signal = None
        try:
            tickvel_signal = tickvel_proc.process(
                Decimal(str(current_price)), list(price_history), metadata
            )
        except Exception:
            pass

        # Compute cross-asset BTC velocity
        btc_vel = None
        if window_btc_ticks is not None and window_btc_ts is not None:
            btc_vel = _compute_btc_velocity_at(window_btc_ticks, window_btc_ts, current_check)

        # Extract features using raw_tick_buffer (has qty/is_buyer)
        feats = extract_features(
            tick_buffer=list(raw_tick_buffer),
            price_history=list(price_history),
            current_price=current_price,
            timestamp=current_check,
            tickvel_signal=tickvel_signal,
            decision_minute=dm,
            window_open_price=window.price_open,
            btc_velocity_60s=btc_vel,
        )
        feats["label"] = label
        feats["window_start"] = window.window_start.isoformat()
        feats["hour_utc"] = float(current_check.hour)
        rows.append(feats)

        current_check = next_check

    return rows


def generate_for_asset(asset: str, days: int | None, min_move: float = 0.0) -> None:
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

    # Load BTC ticks for cross-asset feature (non-BTC assets only)
    btc_ticks = None
    btc_ts_index = None
    if asset.upper() != "BTC":
        btc_ticks_raw = load_aggtrades_multi(DATA_DIR, "BTC", days=days)
        if btc_ticks_raw:
            btc_ticks = btc_ticks_raw
            btc_ts_index = _build_btc_timestamp_index(btc_ticks)
            logger.info("Loaded {} BTC ticks for cross-asset features", len(btc_ticks))
        else:
            logger.warning("No BTC aggTrades data found -- btc_velocity_60s will be 0")

    logger.info("Generating training data for {} ({} windows)", asset, len(windows))

    tickvel_proc = build_processors()

    price_history: deque = deque(maxlen=200)
    tick_buffer: deque = deque(maxlen=300)

    all_rows: list[dict] = []
    log_interval = max(1, len(windows) // 20)

    for i, window in enumerate(windows):
        if i % log_interval == 0:
            logger.info("Processing window {}/{}", i + 1, len(windows))

        rows = extract_window_features(
            window, tickvel_proc, price_history, tick_buffer,
            btc_ticks=btc_ticks, btc_ts_index=btc_ts_index,
        )
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No feature rows generated for {}", asset)
        return

    # Write CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{asset.upper()}_features.csv"

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
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    for asset in assets:
        generate_for_asset(asset, args.days, min_move=args.min_move)


if __name__ == "__main__":
    main()
