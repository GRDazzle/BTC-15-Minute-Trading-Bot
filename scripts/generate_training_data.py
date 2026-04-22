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
from ml.features import FEATURE_NAMES, extract_features, build_tick_index, load_daily_closes, compute_daily_smas
from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker
from ml.multi_exchange import KrakenTickIndex, BitstampOHLCVIndex

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
KRAKEN_DIR = PROJECT_ROOT / "data" / "aggtrades_kraken"
BITSTAMP_DIR = PROJECT_ROOT / "data" / "ohlcv_bitstamp"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "training_data"


def extract_window_features(
    window: TickWindow,
    price_history: deque,
    tick_buffer: deque,
    persistent_raw_buffer: deque | None = None,
    sma5: float | None = None,
    sma15: float | None = None,
    sma30: float | None = None,
    kalshi_index: KalshiPollIndex | None = None,
    asset: str = "",
    kalshi_settlements: dict | None = None,
    kraken_index: KrakenTickIndex | None = None,
    bitstamp_index: BitstampOHLCVIndex | None = None,
    check_interval_seconds: int = 10,
) -> list[dict]:
    """Replay one tick window, extracting features at every checkpoint.

    check_interval_seconds controls sampling density within each dm minute.
    Default 10s -> 6 rows per dm. Use 2s to match live bot's ~2s WS cadence
    (30 rows per dm) — yields ~5x more training samples per window and
    covers the sub-minute tick-buffer states live actually sees.

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

    # Merge Kraken warmup ticks into raw_tick_buffer (aggregated tick stream)
    if kraken_index:
        warmup_start = window.window_start - timedelta(minutes=10)
        warmup_end = window.window_start + timedelta(minutes=5)
        for tick in kraken_index.get_ticks_in_window(warmup_start, warmup_end):
            raw_tick_buffer.append(tick)

    # Pre-load all Kraken ticks for the decision zone (fed incrementally below)
    kr_all_ticks = []
    if kraken_index:
        kr_all_ticks = kraken_index.get_ticks_in_window(
            window.window_start + timedelta(minutes=5), window.window_end,
        )
    kr_tick_idx = 0

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

    # Use Kalshi settlement as ground truth if available, else fall back to Coinbase direction
    label = None
    if kalshi_settlements and asset:
        # Look up by close_time (window_end in UTC ISO format)
        close_time_str = window.window_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        kalshi_result = kalshi_settlements.get(close_time_str)
        if kalshi_result:
            label = 1 if kalshi_result == "yes" else 0
    if label is None:
        # Fallback: use Coinbase price direction
        label = 1 if window.actual_direction == "BULLISH" else 0
    rows = []

    decision_start = window.window_start + timedelta(minutes=5)
    check_interval = timedelta(seconds=check_interval_seconds)
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

        # Feed Kraken ticks up to current_check — merged into same sorted index
        # This aggregates volume, tick intensity, and order flow from both exchanges
        while kr_tick_idx < len(kr_all_ticks) and kr_all_ticks[kr_tick_idx]["ts"] < current_check:
            tick = kr_all_ticks[kr_tick_idx]
            raw_tick_buffer.append(tick)
            ts = _ensure_utc(tick["ts"])
            idx = bisect.bisect_left(ts_idx, ts)
            ts_idx.insert(idx, ts)
            sorted_raw.insert(idx, tick)
            kr_tick_idx += 1

        if len(price_history) < 20:
            current_check = next_check
            continue

        current_price = float(price_history[-1])

        # Compute decision minute
        elapsed_s = (current_check - window.window_start).total_seconds()
        dm = int((elapsed_s - 300) / 60)

        # Kalshi market data lookup (if available)
        k_yes_ask = None
        k_yes_bid = None
        k_no_ask = None
        k_mtc = None
        k_poll_history = None
        if kalshi_index and asset:
            event_ticker = window_start_to_event_ticker(asset, window.window_end)
            poll = kalshi_index.find_poll(event_ticker, current_check)
            if poll:
                k_yes_ask = poll["yes_ask"]
                k_yes_bid = poll["yes_bid"]
                k_no_ask = poll["no_ask"]
                k_mtc = poll["mins_to_close"]
            # Get poll history for Kalshi RT features (last 60s of polls)
            k_poll_history = kalshi_index.get_poll_history(event_ticker, current_check, lookback_seconds=60)

        # Cross-exchange data lookup
        kr_price = None
        kr_ticks = None
        bs_price = None
        bs_ticks = None
        if kraken_index:
            kr_price = kraken_index.get_price_at(current_check)
            # Get Kraken ticks in last 60s for velocity/volume features
            from datetime import timedelta as _td
            kr_ticks = kraken_index.get_ticks_in_window(
                current_check - _td(seconds=60), current_check,
            )
        if bitstamp_index:
            bs_price = bitstamp_index.get_price_at(current_check)
            # Get Bitstamp candles in last 60s for volume
            candle = bitstamp_index.get_candle_at(current_check)
            if candle:
                bs_ticks = [{"qty": candle["volume"], "ts": candle["ts"], "price": candle["close"]}]

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
            sma5=sma5,
            sma15=sma15,
            sma30=sma30,
            kalshi_yes_ask=k_yes_ask,
            kalshi_yes_bid=k_yes_bid,
            kalshi_no_ask=k_no_ask,
            kalshi_mins_to_close=k_mtc,
            kraken_tick_buffer=kr_ticks,
            kraken_current_price=kr_price,
            bitstamp_tick_buffer=bs_ticks,
            bitstamp_current_price=bs_price,
            kalshi_poll_history=k_poll_history,
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


def generate_for_asset(asset: str, days: int | None, min_move: float = 0.0,
                       day_filter: str = "all",
                       check_interval_seconds: int = 10) -> None:
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

    # Precompute daily closing prices for SMA features
    daily_closes = load_daily_closes(DATA_DIR, asset)
    logger.info("Loaded {} daily closing prices for SMA computation", len(daily_closes))

    # Load Kalshi poll data for market-aware features (v10)
    kalshi_index = None
    if KALSHI_DIR.exists():
        logger.info("Loading Kalshi poll data for {} ...", asset)
        kalshi_index = KalshiPollIndex(KALSHI_DIR, asset)
        logger.info("Kalshi polls loaded: {} events, {} polls", len(kalshi_index.event_tickers), kalshi_index.n_polls())
    else:
        logger.info("No Kalshi poll data found -- Kalshi features will use defaults")

    # Load Kalshi settlement results for accurate labels (replaces Coinbase price direction)
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

    # Load Kraken tick data for cross-exchange features
    kraken_index = None
    if KRAKEN_DIR.exists():
        logger.info("Loading Kraken tick data for {} ...", asset)
        kraken_index = KrakenTickIndex(KRAKEN_DIR, asset)
        logger.info("Kraken ticks loaded: {}", kraken_index.n_ticks())

    # Load Bitstamp OHLCV data for cross-exchange features
    bitstamp_index = None
    if BITSTAMP_DIR.exists():
        logger.info("Loading Bitstamp OHLCV data for {} ...", asset)
        bitstamp_index = BitstampOHLCVIndex(BITSTAMP_DIR, asset)
        logger.info("Bitstamp candles loaded: {}", bitstamp_index.n_candles())

    all_rows: list[dict] = []
    log_interval = max(1, len(windows) // 20)

    for i, window in enumerate(windows):
        if i % log_interval == 0:
            logger.info("Processing window {}/{}", i + 1, len(windows))

        # Compute SMAs for this window's date
        window_date = window.window_start.strftime("%Y-%m-%d")
        sma5, sma15, sma30 = compute_daily_smas(daily_closes, window_date)

        rows = extract_window_features(
            window, price_history, tick_buffer,
            persistent_raw_buffer=persistent_raw_buffer,
            sma5=sma5, sma15=sma15, sma30=sma30,
            kalshi_index=kalshi_index, asset=asset,
            kalshi_settlements=kalshi_settlements,
            kraken_index=kraken_index,
            bitstamp_index=bitstamp_index,
            check_interval_seconds=check_interval_seconds,
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
    parser.add_argument(
        "--check-interval-seconds", type=int, default=10,
        help="Seconds between decision-zone samples (default: 10). "
             "Use 2 to match live bot's WS-driven ~2s cadence — yields 5x more "
             "rows and covers sub-minute tick-buffer states live actually sees.",
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
