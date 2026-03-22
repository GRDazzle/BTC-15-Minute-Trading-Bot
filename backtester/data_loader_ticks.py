"""
Tick-level data loader for backtesting.
Loads Binance aggTrades CSVs and generates 15-minute TickWindows with real
sub-second price data -- matching what the live system sees.

aggTrades CSV format (no header):
  agg_trade_id, price, qty, first_id, last_id, timestamp, is_buyer_maker, best_price_match

Note: Since Jan 2025, Binance timestamps are in microseconds.
"""
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class Tick:
    ts: datetime
    price: float
    qty: float
    is_buyer: bool


@dataclass
class TickWindow:
    """15-minute window with raw tick data."""
    window_start: datetime       # e.g., 2026-03-01T10:00:00Z
    window_end: datetime         # e.g., 2026-03-01T10:15:00Z
    ticks_before: list[Tick]     # 5 min of ticks before decision zone (warmup)
    ticks_during: list[Tick]     # 10 min of ticks during decision zone (min 5-14)
    price_open: float            # Price at window_start + 5min
    price_close: float           # Price at window_end - 1min
    actual_direction: str        # "BULLISH" or "BEARISH"


def _parse_timestamp(ts_raw: str) -> datetime:
    """Parse Binance aggTrade timestamp.

    Since Jan 2025, timestamps are in microseconds.
    Pre-2025 data used milliseconds. We detect by magnitude.
    """
    ts_int = int(ts_raw)
    if ts_int > 1e15:  # microseconds (16+ digits)
        return datetime.fromtimestamp(ts_int / 1_000_000, tz=timezone.utc)
    else:  # milliseconds (13 digits)
        return datetime.fromtimestamp(ts_int / 1_000, tz=timezone.utc)


def load_aggtrades(csv_path: Path) -> list[Tick]:
    """Load aggTrades CSV (no header).

    Returns list of Tick sorted by timestamp.
    """
    ticks = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            try:
                ts = _parse_timestamp(row[5])
                ticks.append(Tick(
                    ts=ts,
                    price=float(row[1]),
                    qty=float(row[2]),
                    is_buyer=row[6].strip().lower() != "true",  # is_buyer_maker=True means seller-initiated
                ))
            except (ValueError, IndexError):
                continue

    ticks.sort(key=lambda t: t.ts)
    return ticks


def load_aggtrades_multi(data_dir: Path, asset: str, days: Optional[int] = None) -> list[Tick]:
    """Load multiple days of aggTrades CSVs for an asset.

    Args:
        data_dir: Root data/aggtrades directory
        asset: e.g. "BTC"
        days: If set, only load the most recent N days of files
    """
    asset_dir = data_dir / asset.upper()
    if not asset_dir.exists():
        logger.error("aggTrades directory not found: {}", asset_dir)
        return []

    csv_files = sorted(asset_dir.glob("*.csv"))
    if days is not None and len(csv_files) > days:
        csv_files = csv_files[-days:]

    logger.info("Loading {} aggTrades files for {}", len(csv_files), asset)

    all_ticks: list[Tick] = []
    for csv_path in csv_files:
        day_ticks = load_aggtrades(csv_path)
        if day_ticks:
            logger.info("  {}: {} ticks", csv_path.name, len(day_ticks))
            all_ticks.extend(day_ticks)
        else:
            logger.warning("  {}: no ticks loaded", csv_path.name)

    all_ticks.sort(key=lambda t: t.ts)
    logger.info("Total: {} ticks loaded for {}", len(all_ticks), asset)
    return all_ticks


def resample_ticks(
    ticks: list[Tick],
    start: datetime,
    end: datetime,
    interval_ms: int = 250,
) -> list[dict]:
    """Resample raw ticks to fixed-interval bars.

    Args:
        ticks: Sorted list of Tick objects
        start: Start time for resampling
        end: End time for resampling
        interval_ms: Interval in milliseconds (default 250ms, matching live WS cadence)

    Returns list of {"ts": datetime, "price": float} at the given interval,
    using the last trade price in each bucket.
    """
    if not ticks:
        return []

    interval = timedelta(milliseconds=interval_ms)
    bars = []
    # Align start to interval boundary
    current = start.replace(microsecond=0)
    tick_idx = 0
    last_price = ticks[0].price

    while current < end:
        next_slot = current + interval
        # Find the last tick in this interval
        while tick_idx < len(ticks) and ticks[tick_idx].ts < next_slot:
            if ticks[tick_idx].ts >= current:
                last_price = ticks[tick_idx].price
            tick_idx += 1

        bars.append({"ts": current, "price": last_price})
        current = next_slot

    return bars


def generate_tick_windows(
    ticks: list[Tick],
    warmup_minutes: int = 5,
    min_warmup_ticks: int = 100,
    min_during_ticks: int = 10,
) -> list[TickWindow]:
    """Group ticks into 15-minute windows aligned to :00/:15/:30/:45.

    Each window:
    - warmup: ticks from minute 0-4 (before decision zone)
    - during: ticks from minute 5-14 (decision zone, dm 0-9)
    - price_open: first tick price at minute 5
    - price_close: last tick price at minute 14

    Args:
        ticks: Sorted list of ticks
        warmup_minutes: Minutes of warmup data at start of window (default 5)
        min_warmup_ticks: Minimum ticks needed in warmup period
        min_during_ticks: Minimum ticks needed in decision zone (default 10)
    """
    if not ticks:
        return []

    # Find the range of 15-min boundaries
    first_ts = ticks[0].ts
    last_ts = ticks[-1].ts

    # Align to next 15-min boundary after first tick
    minute = first_ts.minute
    aligned_minute = ((minute // 15) + 1) * 15
    if aligned_minute >= 60:
        boundary = first_ts.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        boundary = first_ts.replace(minute=aligned_minute, second=0, microsecond=0)

    # Skip the first boundary to ensure warmup data from previous window
    boundary += timedelta(minutes=15)

    # Build tick index by 15-min window for efficient lookup
    # Pre-sort ticks into buckets
    tick_idx = 0

    windows: list[TickWindow] = []
    total_boundaries = 0

    while boundary + timedelta(minutes=15) <= last_ts:
        total_boundaries += 1
        window_start = boundary
        window_end = boundary + timedelta(minutes=15)
        decision_start = window_start + timedelta(minutes=warmup_minutes)
        # price_close reference: 1 min before window end
        close_ref = window_end - timedelta(minutes=1)

        # Collect warmup ticks (minute 0 to warmup_minutes)
        warmup_ticks = []
        during_ticks = []

        # Advance tick_idx to window_start (but don't overshoot)
        while tick_idx < len(ticks) and ticks[tick_idx].ts < window_start:
            tick_idx += 1

        # Collect ticks for this window
        scan_idx = tick_idx
        while scan_idx < len(ticks) and ticks[scan_idx].ts < window_end:
            tick = ticks[scan_idx]
            if tick.ts < decision_start:
                warmup_ticks.append(tick)
            else:
                during_ticks.append(tick)
            scan_idx += 1

        # Need sufficient data
        if len(warmup_ticks) < min_warmup_ticks or len(during_ticks) < min_during_ticks:
            boundary += timedelta(minutes=15)
            continue

        # Determine open price (first tick at decision start) and close price
        price_open = during_ticks[0].price

        # Close price: last tick at or before close_ref
        close_ticks = [t for t in during_ticks if t.ts <= close_ref]
        if not close_ticks:
            boundary += timedelta(minutes=15)
            continue
        price_close = close_ticks[-1].price

        actual_direction = "BULLISH" if price_close > price_open else "BEARISH"

        windows.append(TickWindow(
            window_start=window_start,
            window_end=window_end,
            ticks_before=warmup_ticks,
            ticks_during=during_ticks,
            price_open=price_open,
            price_close=price_close,
            actual_direction=actual_direction,
        ))

        boundary += timedelta(minutes=15)

    logger.info("Generated {} tick windows from {} boundaries", len(windows), total_boundaries)
    if windows:
        logger.info("  Range: {} -> {}", windows[0].window_start, windows[-1].window_start)
        bullish = sum(1 for w in windows if w.actual_direction == "BULLISH")
        logger.info("  Distribution: {} BULLISH / {} BEARISH", bullish, len(windows) - bullish)

    return windows
