"""
Generate LSTM training data from tick replay.

Replays tick windows and extracts 1-second price sequences at every
10-second checkpoint, producing (seq_len, features) arrays for LSTM training.

Usage:
  python scripts/generate_lstm_training_data.py --asset BTC
  python scripts/generate_lstm_training_data.py --asset BTC,ETH,SOL,XRP --days 30
"""
import argparse
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader_ticks import (
    Tick,
    load_aggtrades_multi,
    generate_tick_windows,
)
from ml.lstm_features import LSTM_SEQ_LEN, LSTM_NUM_FEATURES, extract_lstm_sequence
from ml.multi_exchange import KrakenTickIndex

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
KRAKEN_DIR = PROJECT_ROOT / "data" / "aggtrades_kraken"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "training_data"


def extract_window_sequences(
    window,
    seq_len: int = LSTM_SEQ_LEN,
    kalshi_settlements: dict | None = None,
    asset: str = "",
    kraken_index=None,
) -> list[dict]:
    """Replay one tick window, extracting LSTM sequences at every 10s checkpoint.

    Returns list of dicts with 'sequence' (ndarray), 'label', 'window_start', 'dm'.
    """
    # Build raw tick buffer from warmup + during ticks
    raw_tick_buffer: deque = deque(maxlen=max(5000, seq_len * 10))

    for tick in window.ticks_before:
        raw_tick_buffer.append({
            "ts": tick.ts,
            "price": tick.price,
            "qty": tick.qty,
            "is_buyer": tick.is_buyer,
        })

    # Merge Kraken warmup ticks into tick buffer (aggregated stream)
    if kraken_index is not None:
        warmup_start = window.window_start - timedelta(minutes=10)
        warmup_end = window.window_start + timedelta(minutes=5)
        for tick in kraken_index.get_ticks_in_window(warmup_start, warmup_end):
            raw_tick_buffer.append(tick)

    # Pre-load Kraken decision-zone ticks (fed incrementally below)
    kr_all_ticks = []
    if kraken_index is not None:
        kr_all_ticks = kraken_index.get_ticks_in_window(
            window.window_start + timedelta(minutes=5), window.window_end,
        )
    kr_tick_idx = 0

    # Use Kalshi settlement as ground truth if available
    label = None
    if kalshi_settlements and asset:
        close_time_str = window.window_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        kalshi_result = kalshi_settlements.get(close_time_str)
        if kalshi_result:
            label = 1 if kalshi_result == "yes" else 0
    if label is None:
        label = 1 if window.actual_direction == "BULLISH" else 0
    price_return = 0.0
    if window.price_open != 0:
        price_return = (window.price_close - window.price_open) / window.price_open
    rows = []

    decision_start = window.window_start + timedelta(minutes=5)
    check_interval = timedelta(seconds=10)
    current_check = decision_start + check_interval

    # Build sorted tick list once from warmup, append incrementally (ticks arrive in order)
    sorted_ticks = list(raw_tick_buffer)

    tick_idx = 0

    while current_check < window.window_end:
        next_check = current_check + check_interval

        # Feed Coinbase ticks up to current_check
        while tick_idx < len(window.ticks_during) and window.ticks_during[tick_idx].ts < current_check:
            tick = window.ticks_during[tick_idx]
            tick_dict = {
                "ts": tick.ts,
                "price": tick.price,
                "qty": tick.qty,
                "is_buyer": tick.is_buyer,
            }
            sorted_ticks.append(tick_dict)
            tick_idx += 1

        # Feed Kraken ticks up to current_check (aggregated stream)
        while kr_tick_idx < len(kr_all_ticks) and kr_all_ticks[kr_tick_idx]["ts"] < current_check:
            sorted_ticks.append(kr_all_ticks[kr_tick_idx])
            kr_tick_idx += 1

        # Compute decision minute
        elapsed_s = (current_check - window.window_start).total_seconds()
        dm = int((elapsed_s - 300) / 60)

        # Re-sort so Coinbase and Kraken ticks interleave by timestamp, matching
        # live behavior (live merges then sorts before passing to LSTM).
        # Without this, bisect inside extract_lstm_sequence gives wrong indices
        # when kraken ticks are appended after coinbase ticks with earlier ts.
        sorted_ticks.sort(key=lambda t: t["ts"])

        # Extract LSTM sequence — pass list directly (no copy needed)
        seq = extract_lstm_sequence(
            sorted_ticks,
            current_check,
            decision_minute=dm,
            seq_len=seq_len,
            window_open_price=window.price_open,
        )

        if seq is not None:
            rows.append({
                "sequence": seq,
                "label": label,
                "price_return": price_return,
                "window_start": window.window_start.isoformat(),
                "dm": dm,
            })

        current_check = next_check

    return rows


def generate_for_asset(asset: str, days: int | None, min_move: float = 0.0, day_filter: str = "all") -> None:
    """Generate LSTM training data for one asset."""
    ticks = load_aggtrades_multi(DATA_DIR, asset, days=days)
    if not ticks:
        logger.error("No aggTrades data found for {}.", asset)
        return

    windows = generate_tick_windows(ticks)
    if not windows:
        logger.error("No valid tick windows for {}", asset)
        return

    # Filter tiny moves
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

    # Load Kraken tick index for aggregated tick stream
    kraken_index = None
    if (KRAKEN_DIR / asset.upper()).exists():
        kraken_index = KrakenTickIndex(KRAKEN_DIR, asset.upper())
        logger.info("Kraken ticks loaded for {}: {} (will merge into tick stream)", asset, kraken_index.n_ticks())
    else:
        logger.info("No Kraken data for {} -- Coinbase ticks only", asset)

    logger.info("Generating LSTM training data for {} ({} windows)", asset, len(windows))

    # Load Kalshi settlement results for accurate labels
    import json as _json
    kalshi_settlements = None
    settlements_path = PROJECT_ROOT / "data" / "kalshi_settlements" / f"{asset.upper()}_settlements.json"
    if settlements_path.exists():
        with open(settlements_path) as f:
            sdata = _json.load(f)
        kalshi_settlements = sdata.get("by_close_time", {})
        logger.info("Kalshi settlements loaded for {}: {} outcomes", asset, len(kalshi_settlements))
    else:
        logger.info("No Kalshi settlement data for {} -- using Coinbase price labels", asset)

    all_sequences = []
    all_labels = []
    all_returns = []
    all_window_starts = []
    all_dms = []

    log_interval = max(1, len(windows) // 20)

    for i, window in enumerate(windows):
        if i % log_interval == 0:
            logger.info("Processing window {}/{}", i + 1, len(windows))

        rows = extract_window_sequences(
            window, kalshi_settlements=kalshi_settlements, asset=asset,
            kraken_index=kraken_index,
        )
        for row in rows:
            all_sequences.append(row["sequence"])
            all_labels.append(row["label"])
            all_returns.append(row["price_return"])
            all_window_starts.append(row["window_start"])
            all_dms.append(row["dm"])

    if not all_sequences:
        logger.error("No LSTM sequences generated for {}", asset)
        return

    X = np.stack(all_sequences)  # (N, seq_len, features)
    y = np.array(all_labels, dtype=np.int32)
    returns = np.array(all_returns, dtype=np.float32)
    dms = np.array(all_dms, dtype=np.int32)

    # Save as .npz
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    day_suffix = f"_{day_filter}" if day_filter != "all" else ""
    out_path = OUTPUT_DIR / f"{asset.upper()}_lstm_sequences{day_suffix}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        returns=returns,
        window_starts=np.array(all_window_starts),
        dms=dms,
    )

    bullish = int(y.sum())
    bearish = len(y) - bullish
    unique_windows = len(set(all_window_starts))

    print(f"\n=== {asset} LSTM Training Data ===")
    print(f"Total sequences:  {len(y)}")
    print(f"Shape:            {X.shape}")
    print(f"Unique windows:   {unique_windows}")
    print(f"Seqs/window avg:  {len(y) / unique_windows:.1f}")
    print(f"Label balance:    {bullish} BULLISH / {bearish} BEARISH ({bullish/len(y)*100:.1f}%)")
    print(f"Output:           {out_path}")
    print()


def main():
    logger.remove()
    log_path = PROJECT_ROOT / "logs" / "generate_lstm_data.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), mode="w", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="Generate LSTM training data from tick replay")
    parser.add_argument("--asset", required=True, help="Asset(s), comma-separated")
    parser.add_argument("--days", type=int, default=None, help="Limit to last N days")
    parser.add_argument("--min-move", type=float, default=0.0001, help="Min price move pct")
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
