"""Unified XGB + LSTM training-data generator.

Loads tick data once, builds one TickWindowSlicer, iterates windows once, and
emits both:
  - {ASSET}_features.csv         (XGB)
  - {ASSET}_lstm_sequences.npz   (LSTM)

per asset. Structurally guarantees both heads see the exact same merged
CB+KR buffer at every checkpoint, eliminating a class of training-inference
alignment bugs and cutting wall time ~50% vs running the two legacy scripts
back-to-back.

Semantics are preserved from the two original scripts
(`generate_training_data.py`, `generate_lstm_training_data.py`):
  - 2s checkpoint cadence (configurable)
  - Warmup + decision-zone iteration, 5-14min window coverage
  - Kalshi settlement as label ground-truth (fallback to Coinbase direction)
  - Kraken merged into the primary buffer via the slicer
  - Per-checkpoint Kalshi poll snapshot (XGB only), Kraken-only 60s slice
    (XGB cross-exchange features), SMA lookup per window date
  - day_filter, min_move options

Usage:
    python scripts/generate_aligned_training_data.py --asset BTC --days 30
    python scripts/generate_aligned_training_data.py --asset BTC,ETH,SOL,XRP \\
        --days 30 --check-interval-seconds 2 --day-filter all
    # emit only one branch if you only need to regen one:
    python scripts/generate_aligned_training_data.py --asset BTC --outputs xgb
    python scripts/generate_aligned_training_data.py --asset BTC --outputs lstm
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader_ticks import load_aggtrades_multi, generate_tick_windows
from core.tick_window_slicer import TickWindowSlicer
from ml.features import (
    FEATURE_NAMES,
    extract_features,
    load_daily_closes,
    compute_daily_smas,
)
from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker
from ml.lstm_features import LSTM_SEQ_LEN, LSTM_NUM_FEATURES, extract_lstm_sequence
from ml.multi_exchange import KrakenTickIndex, BitstampOHLCVIndex

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
KRAKEN_DIR = PROJECT_ROOT / "data" / "aggtrades_kraken"
BITSTAMP_DIR = PROJECT_ROOT / "data" / "ohlcv_bitstamp"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "training_data"

# Feature-window lookback. XGB's velocity_900s needs 900s; LSTM bisects to 180s
# internally. Any lookback >= 900 preserves both; we use 900 for efficiency.
SLICER_LOOKBACK = 900


def process_window(
    window,
    slicer: TickWindowSlicer,
    *,
    asset: str,
    kalshi_index: KalshiPollIndex | None,
    kalshi_settlements: dict | None,
    bitstamp_index: BitstampOHLCVIndex | None,
    sma5: float | None,
    sma15: float | None,
    sma30: float | None,
    check_interval_seconds: int,
    want_xgb: bool,
    want_lstm: bool,
) -> tuple[list[dict], list[dict]]:
    """Replay one window, extract both XGB rows and LSTM sequences per checkpoint.

    Returns (xgb_rows, lstm_rows) — each entry is a dict. LSTM entries carry a
    numpy 'sequence' field; XGB entries are flat feature dicts.

    Both branches query the SAME slicer via the SAME anchor timestamp at each
    checkpoint, so the tick buffer they see is byte-identical.
    """
    # Compute event_ticker once (used for Kalshi poll lookups + label source)
    event_ticker = window_start_to_event_ticker(asset, window.window_end) if kalshi_index else None

    # Label priority — same source sweep/replay uses, for train/eval alignment:
    #   1. Poll archive outcome record (matches what pnl_sweep and replay use).
    #   2. kalshi_settlements.json (backup — historical coverage, may be stale).
    #   3. Coinbase price direction (fallback — different ground truth; shifts label).
    # Logging the source once per window would be too verbose; callers can audit
    # via the written rows (which inherit label from here) vs the same event
    # lookups in a diagnostic pass.
    label = None
    if kalshi_index and event_ticker:
        outcome = kalshi_index.get_outcome(event_ticker)
        if outcome in ("yes", "no"):
            label = 1 if outcome == "yes" else 0
    if label is None and kalshi_settlements and asset:
        key = window.window_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        val = kalshi_settlements.get(key)
        if val:
            label = 1 if val == "yes" else 0
    if label is None:
        label = 1 if window.actual_direction == "BULLISH" else 0

    price_return = 0.0
    if window.price_open != 0:
        price_return = (window.price_close - window.price_open) / window.price_open

    xgb_rows: list[dict] = []
    lstm_rows: list[dict] = []

    decision_start = window.window_start + timedelta(minutes=5)
    check_interval = timedelta(seconds=check_interval_seconds)
    current_check = decision_start + check_interval

    # event_ticker already computed above (line ~93) and reused here for
    # per-checkpoint Kalshi poll lookups.

    while current_check < window.window_end:
        elapsed_s = (current_check - window.window_start).total_seconds()
        dm = int((elapsed_s - 300) / 60)

        # AUTHORITATIVE SHARED BUFFER for this checkpoint. Both XGB and LSTM
        # branches pull their feature inputs from this single merged window.
        merged = slicer.get_merged_window(
            current_check, SLICER_LOOKBACK, sources=("coinbase", "kraken"),
        )
        if not merged:
            current_check += check_interval
            continue

        # --- LSTM branch -----------------------------------------------------
        if want_lstm:
            seq = extract_lstm_sequence(
                merged,
                current_check,
                decision_minute=dm,
                seq_len=LSTM_SEQ_LEN,
                window_open_price=window.price_open,
            )
            if seq is not None:
                lstm_rows.append({
                    "sequence": seq,
                    "label": label,
                    "price_return": price_return,
                    "window_start": window.window_start.isoformat(),
                    "dm": dm,
                })

        # --- XGB branch ------------------------------------------------------
        if want_xgb:
            current_price = float(merged[-1]["price"])

            # Kalshi poll snapshot (per-checkpoint)
            k_yes_ask = k_yes_bid = k_no_ask = k_mtc = None
            k_poll_history = None
            if kalshi_index and event_ticker:
                poll = kalshi_index.find_poll(event_ticker, current_check)
                if poll:
                    k_yes_ask = poll["yes_ask"]
                    k_yes_bid = poll["yes_bid"]
                    k_no_ask = poll["no_ask"]
                    k_mtc = poll["mins_to_close"]
                k_poll_history = kalshi_index.get_poll_history(
                    event_ticker, current_check, lookback_seconds=60,
                )

            # Cross-exchange: Kraken-only last 60s slice + current Kraken price
            kr_ticks = slicer.get_merged_window(
                current_check, 60, sources=("kraken",),
            )
            kr_price = kr_ticks[-1]["price"] if kr_ticks else None

            # Bitstamp (separate index — not via slicer)
            bs_price = None
            bs_ticks = None
            if bitstamp_index:
                bs_price = bitstamp_index.get_price_at(current_check)
                candle = bitstamp_index.get_candle_at(current_check)
                if candle:
                    bs_ticks = [{
                        "qty": candle["volume"],
                        "ts": candle["ts"],
                        "price": candle["close"],
                    }]

            # Build ts_index once per checkpoint for O(log n) price lookups in
            # extract_features (for price_30s/price_60s etc).
            ts_index = [t["ts"] for t in merged]

            feats = extract_features(
                tick_buffer=None,
                price_history=[float(t["price"]) for t in merged[-200:]],
                current_price=current_price,
                timestamp=current_check,
                decision_minute=dm,
                window_open_price=window.price_open,
                ts_index=ts_index,
                sorted_ticks=merged,
                sma5=sma5,
                sma15=sma15,
                sma30=sma30,
                kalshi_yes_ask=k_yes_ask,
                kalshi_yes_bid=k_yes_bid,
                kalshi_no_ask=k_no_ask,
                kalshi_mins_to_close=k_mtc,
                kraken_tick_buffer=kr_ticks or None,
                kraken_current_price=kr_price,
                bitstamp_tick_buffer=bs_ticks,
                bitstamp_current_price=bs_price,
                kalshi_poll_history=k_poll_history,
            )
            feats["label"] = label
            feats["window_start"] = window.window_start.isoformat()
            feats["hour_utc"] = float(current_check.hour)
            xgb_rows.append(feats)

        current_check += check_interval

    return xgb_rows, lstm_rows


def generate_for_asset(
    asset: str,
    days: int | None,
    min_move: float,
    day_filter: str,
    check_interval_seconds: int,
    want_xgb: bool,
    want_lstm: bool,
) -> None:
    """Generate training data for one asset — writes XGB CSV and/or LSTM NPZ."""
    asset = asset.upper()

    # Load Coinbase + Kraken ticks up-front
    cb_ticks = load_aggtrades_multi(DATA_DIR, asset, days=days)
    if not cb_ticks:
        logger.error("No Coinbase aggTrades for {}. Skipping.", asset)
        return

    # Windows from Coinbase ticks (the primary source — defines the window grid)
    windows = generate_tick_windows(cb_ticks)
    if not windows:
        logger.error("No valid tick windows for {}", asset)
        return

    if min_move > 0:
        before = len(windows)
        windows = [
            w for w in windows
            if w.price_open != 0 and abs(w.price_close - w.price_open) / w.price_open >= min_move
        ]
        logger.info(
            "Filtered {} windows with move < {:.4%} ({} -> {})",
            before - len(windows), min_move, before, len(windows),
        )

    if day_filter == "weekday":
        before = len(windows)
        windows = [w for w in windows if w.window_start.weekday() < 5]
        logger.info("Day filter weekday: {} -> {} windows", before, len(windows))
    elif day_filter == "weekend":
        before = len(windows)
        windows = [w for w in windows if w.window_start.weekday() >= 5]
        logger.info("Day filter weekend: {} -> {} windows", before, len(windows))

    # Build ONE slicer across all windows (ticks are pre-sorted by load path)
    slicer = TickWindowSlicer()
    slicer.extend("coinbase", [{
        "ts": t.ts, "price": t.price, "qty": t.qty, "is_buyer": t.is_buyer,
    } for t in cb_ticks])
    logger.info("Slicer: loaded {} Coinbase ticks", len(cb_ticks))

    # Kraken via KrakenTickIndex — convert to slicer format
    kraken_index = None
    if KRAKEN_DIR.exists():
        kraken_index = KrakenTickIndex(KRAKEN_DIR, asset)
        # Pull all KR ticks in the window range and push through the slicer
        if windows:
            kr_ticks = kraken_index.get_ticks_in_window(
                windows[0].window_start - timedelta(minutes=15),
                windows[-1].window_end + timedelta(minutes=5),
            )
            if kr_ticks:
                slicer.extend("kraken", list(kr_ticks))
                logger.info("Slicer: loaded {} Kraken ticks", len(kr_ticks))

    # Load Kalshi poll data. Used for:
    #   (1) XGB v10 Kalshi features (yes_ask / yes_bid / etc.) — XGB only.
    #   (2) label outcomes via get_outcome() — BOTH branches (primary source so
    #       training labels align with what sweep/replay evaluate on).
    kalshi_index = None
    if KALSHI_DIR.exists():
        logger.info("Loading Kalshi poll data for {} ...", asset)
        kalshi_index = KalshiPollIndex(KALSHI_DIR, asset)
        n_outcomes = len(kalshi_index._outcomes)
        logger.info(
            "Kalshi polls loaded: {} events, {} polls, {} outcomes",
            len(kalshi_index.event_tickers), kalshi_index.n_polls(), n_outcomes,
        )

    # Secondary label source: kalshi_settlements.json (historical backfill).
    # Used only when the poll archive doesn't have an outcome for a window
    # (typically older windows before the bot started polling).
    kalshi_settlements = None
    settlements_path = PROJECT_ROOT / "data" / "kalshi_settlements" / f"{asset}_settlements.json"
    if settlements_path.exists():
        with open(settlements_path) as f:
            kalshi_settlements = json.load(f).get("by_close_time", {})
        logger.info("Kalshi settlements.json (backup labels): {} outcomes", len(kalshi_settlements))

    # Load Bitstamp OHLCV (XGB only)
    bitstamp_index = None
    if want_xgb and BITSTAMP_DIR.exists():
        bitstamp_index = BitstampOHLCVIndex(BITSTAMP_DIR, asset)
        logger.info("Bitstamp candles loaded: {}", bitstamp_index.n_candles())

    # Precompute daily closing prices for SMA features (XGB only)
    daily_closes = {}
    if want_xgb:
        daily_closes = load_daily_closes(DATA_DIR, asset)
        logger.info("Loaded {} daily closing prices for SMA computation", len(daily_closes))

    # Iterate
    logger.info(
        "Generating aligned training data for {} ({} windows, check={}s, outputs={})",
        asset, len(windows), check_interval_seconds,
        ",".join(x for x, b in [("xgb", want_xgb), ("lstm", want_lstm)] if b),
    )

    xgb_all: list[dict] = []
    log_interval = max(1, len(windows) // 20)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    day_suffix = f"_{day_filter}" if day_filter != "all" else ""

    # --- LSTM: memmap-backed streaming writer ---
    # Accumulating 1M+ sequences in a Python list causes OOM at np.stack on
    # large assets (BTC/ETH: 1.25M seqs × 15 KB = 19 GB peak list + 19 GB
    # stack output = 40+ GB per process). Instead: pre-allocate X on disk at
    # an upper bound, write sequences row-by-row as the loop emits them, keep
    # only small per-row metadata in RAM (labels, ints, ISO strings).
    lstm_mmap = None
    tmp_X_path = None
    y_list: list[int] = []
    returns_list: list[float] = []
    dms_list: list[int] = []
    window_starts_list: list[str] = []
    lstm_row_idx = 0

    if want_lstm:
        # Upper bound: decision-zone iteration is 10min / check_interval - 1,
        # typically ≈300 checkpoints per window. Pad to 310 for safety.
        upper_n = max(1, len(windows) * 310)
        tmp_X_path = OUTPUT_DIR / f"{asset}_lstm_X{day_suffix}.tmp.npy"
        lstm_mmap = np.lib.format.open_memmap(
            tmp_X_path, mode="w+", dtype=np.float32,
            shape=(upper_n, LSTM_SEQ_LEN, LSTM_NUM_FEATURES),
        )
        logger.info(
            "Pre-allocated LSTM memmap at {} (upper={}, actual truncated at end)",
            tmp_X_path, upper_n,
        )

    for i, window in enumerate(windows):
        if i % log_interval == 0:
            logger.info("Processing window {}/{}", i + 1, len(windows))

        # Window-date SMAs (XGB only, but cheap — OK to compute always)
        if want_xgb and daily_closes:
            window_date = window.window_start.strftime("%Y-%m-%d")
            sma5, sma15, sma30 = compute_daily_smas(daily_closes, window_date)
        else:
            sma5 = sma15 = sma30 = None

        xgb_rows, lstm_rows = process_window(
            window,
            slicer,
            asset=asset,
            kalshi_index=kalshi_index,
            kalshi_settlements=kalshi_settlements,
            bitstamp_index=bitstamp_index,
            sma5=sma5, sma15=sma15, sma30=sma30,
            check_interval_seconds=check_interval_seconds,
            want_xgb=want_xgb,
            want_lstm=want_lstm,
        )
        xgb_all.extend(xgb_rows)

        # Stream LSTM rows directly to disk-backed memmap — lstm_rows is a
        # small per-window list (~300 entries), freed on next iteration.
        if want_lstm and lstm_rows:
            for r in lstm_rows:
                lstm_mmap[lstm_row_idx] = r["sequence"]
                y_list.append(r["label"])
                returns_list.append(r["price_return"])
                dms_list.append(r["dm"])
                window_starts_list.append(r["window_start"])
                lstm_row_idx += 1

    # --- Write XGB CSV ------------------------------------------------------
    if want_xgb:
        if not xgb_all:
            logger.warning("No XGB rows generated for {}", asset)
        else:
            out_path = OUTPUT_DIR / f"{asset}_features{day_suffix}.csv"
            columns = FEATURE_NAMES + ["label", "window_start", "hour_utc"]
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(xgb_all)
            logger.info("Wrote {} XGB rows to {}", len(xgb_all), out_path)
            bullish = sum(1 for r in xgb_all if r["label"] == 1)
            unique = len(set(r["window_start"] for r in xgb_all))
            print(f"\n=== {asset} XGB Training Data ===")
            print(f"  rows            : {len(xgb_all)}")
            print(f"  unique windows  : {unique}")
            print(f"  rows/window avg : {len(xgb_all)/unique:.1f}")
            print(f"  label balance   : {bullish} BULLISH / {len(xgb_all)-bullish} BEARISH "
                  f"({bullish/len(xgb_all)*100:.1f}%)")

    # --- Write LSTM: split into X.npy + meta.npz ---------------------------
    # Format change: legacy single-npz required ~2× the X size in RAM at
    # write time (list + np.stack). Split format streams X to its own .npy
    # (memmap → np.save chunked write), keeps metadata in a small .npz.
    # train_lstm.py loads both and reconstructs the same (X, y, returns,
    # dms, window_starts) bundle.
    if want_lstm:
        assert lstm_mmap is not None and tmp_X_path is not None
        # Flush + close the pre-allocated memmap BEFORE reopening for copy
        lstm_mmap.flush()
        del lstm_mmap  # release handle so np.load can memmap it cleanly

        if lstm_row_idx == 0:
            logger.warning("No LSTM sequences generated for {}", asset)
            if tmp_X_path.exists():
                tmp_X_path.unlink()
        else:
            final_X_path = OUTPUT_DIR / f"{asset}_lstm_X{day_suffix}.npy"
            meta_path = OUTPUT_DIR / f"{asset}_lstm_meta{day_suffix}.npz"

            # Reopen temp as readonly memmap, save truncated slice.
            # np.save writes in 256MB chunks via np.nditer — no full-array RAM load.
            X_view = np.load(tmp_X_path, mmap_mode="r")
            np.save(final_X_path, X_view[:lstm_row_idx])
            del X_view
            tmp_X_path.unlink()

            y_arr = np.array(y_list, dtype=np.int32)
            np.savez_compressed(
                meta_path,
                y=y_arr,
                returns=np.array(returns_list, dtype=np.float32),
                dms=np.array(dms_list, dtype=np.int32),
                window_starts=np.array(window_starts_list, dtype=object),
            )
            shape_tuple = (lstm_row_idx, LSTM_SEQ_LEN, LSTM_NUM_FEATURES)
            logger.info(
                "Wrote {} LSTM sequences to {} + {} (shape {})",
                lstm_row_idx, final_X_path, meta_path, shape_tuple,
            )
            bullish = int(y_arr.sum())
            print(f"\n=== {asset} LSTM Training Data ===")
            print(f"  sequences       : {lstm_row_idx}")
            print(f"  shape           : {shape_tuple}")
            print(f"  label balance   : {bullish} BULLISH / {lstm_row_idx-bullish} BEARISH "
                  f"({bullish/lstm_row_idx*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True,
                    help="Asset symbol or comma-separated list (e.g. BTC or BTC,ETH,SOL,XRP)")
    ap.add_argument("--days", type=int, default=None,
                    help="Only load the most recent N days of aggTrades")
    ap.add_argument("--min-move", type=float, default=0.0,
                    help="Filter windows whose |close-open|/open < this (default 0: no filter)")
    ap.add_argument("--day-filter", choices=("all", "weekday", "weekend"), default="all",
                    help="Filter windows by day of week")
    ap.add_argument("--check-interval-seconds", type=int, default=2,
                    help="Checkpoint cadence within decision zone (default 2s for live parity)")
    ap.add_argument("--outputs", default="xgb,lstm",
                    help="Comma-separated: xgb,lstm or either alone. Default both.")
    args = ap.parse_args()

    # Per-PID log so parallel runs don't clobber each other
    log_path = PROJECT_ROOT / "logs" / f"generate_aligned_training_data_{os.getpid()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, level="INFO")

    outputs = {o.strip().lower() for o in args.outputs.split(",") if o.strip()}
    want_xgb = "xgb" in outputs
    want_lstm = "lstm" in outputs
    if not (want_xgb or want_lstm):
        logger.error("--outputs must include at least one of xgb, lstm")
        sys.exit(2)

    assets = [a.strip().upper() for a in args.asset.split(",") if a.strip()]
    for asset in assets:
        generate_for_asset(
            asset=asset,
            days=args.days,
            min_move=args.min_move,
            day_filter=args.day_filter,
            check_interval_seconds=args.check_interval_seconds,
            want_xgb=want_xgb,
            want_lstm=want_lstm,
        )


if __name__ == "__main__":
    main()
