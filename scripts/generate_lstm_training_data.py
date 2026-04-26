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
from core.tick_window_slicer import TickWindowSlicer
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
    check_interval_seconds: int = 10,
) -> list[dict]:
    """Replay one tick window, extracting LSTM sequences at every checkpoint.

    check_interval_seconds controls sampling density. Default 10s; use 2s to
    match live bot's ~2s cadence — yields 5x more sequences per window.

    Returns list of dicts with 'sequence' (ndarray), 'label', 'window_start', 'dm'.
    """
    # Authoritative tick primitive — same semantics as live/replay/sweep.
    # One slicer per window, populated with all CB + KR ticks in range. Per-
    # checkpoint queries pull the exact 180s window live's LSTM would see.
    slicer = TickWindowSlicer()
    cb_warmup = [
        {"ts": t.ts, "price": t.price, "qty": t.qty, "is_buyer": t.is_buyer}
        for t in window.ticks_before
    ]
    cb_during = [
        {"ts": t.ts, "price": t.price, "qty": t.qty, "is_buyer": t.is_buyer}
        for t in window.ticks_during
    ]
    if cb_warmup:
        slicer.extend("coinbase", cb_warmup)
    if cb_during:
        slicer.extend("coinbase", cb_during)

    if kraken_index is not None:
        kr_ticks = kraken_index.get_ticks_in_window(
            window.window_start - timedelta(minutes=10),
            window.window_end,
        )
        if kr_ticks:
            slicer.extend("kraken", list(kr_ticks))

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
    check_interval = timedelta(seconds=check_interval_seconds)
    current_check = decision_start + check_interval

    while current_check < window.window_end:
        # Compute decision minute
        elapsed_s = (current_check - window.window_start).total_seconds()
        dm = int((elapsed_s - 300) / 60)

        # Merged CB+KR window sized for LSTM lookback. Same `sources` order as
        # live (kalshi_strategy.py:2005-2009) — CB first, then KR.
        tick_window = slicer.get_merged_window(
            current_check, seq_len, sources=("coinbase", "kraken"),
        )

        seq = extract_lstm_sequence(
            tick_window,
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

        current_check += check_interval

    return rows


def generate_for_asset(asset: str, days: int | None, min_move: float = 0.0,
                       day_filter: str = "all",
                       check_interval_seconds: int = 10) -> None:
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
            check_interval_seconds=check_interval_seconds,
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
    import os as _os
    logger.remove()
    # Per-process log path so parallel regens don't truncate each other's logs.
    log_path = PROJECT_ROOT / "logs" / f"generate_lstm_data_{_os.getpid()}.log"
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
    parser.add_argument(
        "--check-interval-seconds", type=int, default=10,
        help="Seconds between decision-zone samples (default: 10). "
             "Use 2 to match live bot's ~2s WS cadence — 5x more sequences.",
    )
    args = parser.parse_args()

    if args.check_interval_seconds < 1:
        parser.error("--check-interval-seconds must be >= 1")

    assets = [a.strip().upper() for a in args.asset.split(",")]
    for asset in assets:
        generate_for_asset(
            asset, args.days, min_move=args.min_move,
            day_filter=args.day_filter,
            check_interval_seconds=args.check_interval_seconds,
        )


if __name__ == "__main__":
    main()
