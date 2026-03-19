"""
Data loader for backtesting.
Loads Binance 1m klines and Fear & Greed CSV, generates 15-minute windows.
"""
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class Kline:
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass
class Window:
    start_time: datetime
    klines_before: list[Kline]   # up to 200 klines before window for buffer init
    klines_during: list[Kline]   # klines at minutes :05 through :55 within window (up to 14)
    price_open: Decimal           # close at minute :05
    price_close: Decimal          # close at minute :55 (i.e. 14:55 into the window)
    actual_direction: str         # "BULLISH" or "BEARISH"


def load_binance_klines(csv_path: Path) -> list[Kline]:
    """Load Binance 1m klines from CSV, sorted by timestamp."""
    klines = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row["timestamp"]
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            klines.append(Kline(
                timestamp=ts,
                open=Decimal(row["open"]),
                high=Decimal(row["high"]),
                low=Decimal(row["low"]),
                close=Decimal(row["close"]),
                volume=Decimal(row["volume"]),
            ))
    klines.sort(key=lambda k: k.timestamp)
    logger.info(f"Loaded {len(klines)} klines from {csv_path.name}")
    if klines:
        logger.info(f"  Range: {klines[0].timestamp} → {klines[-1].timestamp}")
    return klines


def load_fear_greed(csv_path: Path) -> dict[str, int]:
    """Load Fear & Greed index CSV. Returns {"2026-03-18": 45, ...}."""
    scores: dict[str, int] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row["date"]
            scores[date_str] = int(row["value"])
    logger.info(f"Loaded {len(scores)} F&G scores")
    return scores


def generate_windows(
    klines: list[Kline],
    min_history: int = 20,
) -> list[Window]:
    """
    Generate 15-minute windows aligned to :00, :15, :30, :45.

    Each window uses klines at minutes :05 through :55 relative to window start
    (i.e. 5 minutes in through 14 minutes 55 seconds).

    Since klines are 1-minute candles, the kline at minute :05 has timestamp
    window_start + 5min, and :55 is window_start + 14min (the 14th minute candle).
    """
    # Build index: timestamp -> kline for O(1) lookup
    kline_index: dict[datetime, Kline] = {}
    for k in klines:
        # Normalize to UTC
        ts = k.timestamp.astimezone(timezone.utc)
        # Round to minute (strip seconds/microseconds)
        ts = ts.replace(second=0, microsecond=0)
        kline_index[ts] = k

    # Find all 15-min boundary timestamps
    if not klines:
        return []

    first_ts = klines[0].timestamp.astimezone(timezone.utc).replace(second=0, microsecond=0)
    last_ts = klines[-1].timestamp.astimezone(timezone.utc).replace(second=0, microsecond=0)

    # Align first_ts to next 15-min boundary
    minute = first_ts.minute
    aligned_minute = ((minute // 15) + 1) * 15
    if aligned_minute >= 60:
        start = first_ts.replace(minute=0, second=0, microsecond=0)
        from datetime import timedelta
        start += timedelta(hours=1)
    else:
        start = first_ts.replace(minute=aligned_minute, second=0, microsecond=0)

    from datetime import timedelta

    windows: list[Window] = []
    current = start

    while current + timedelta(minutes=15) <= last_ts + timedelta(minutes=1):
        # Window klines during: minutes 5 through 14 (inclusive)
        # These correspond to timestamps current+5min through current+14min
        klines_during: list[Kline] = []
        for offset_min in range(5, 15):  # 5, 6, 7, ..., 14
            ts = current + timedelta(minutes=offset_min)
            ts = ts.replace(second=0, microsecond=0)
            if ts in kline_index:
                klines_during.append(kline_index[ts])

        if len(klines_during) < 2:
            # Need at least the open and close klines
            current += timedelta(minutes=15)
            continue

        # price_open = close of the first kline (minute :05)
        # price_close = close of the last kline (minute :14, i.e. 14:55 into window)
        price_open = klines_during[0].close
        price_close = klines_during[-1].close

        actual_direction = "BULLISH" if price_close > price_open else "BEARISH"

        # Collect klines_before: up to 200 klines ending at current+4min
        klines_before: list[Kline] = []
        for offset_min in range(200, 0, -1):
            ts = current + timedelta(minutes=5) - timedelta(minutes=offset_min)
            ts = ts.replace(second=0, microsecond=0)
            if ts in kline_index:
                klines_before.append(kline_index[ts])

        if len(klines_before) < min_history:
            current += timedelta(minutes=15)
            continue

        windows.append(Window(
            start_time=current,
            klines_before=klines_before,
            klines_during=klines_during,
            price_open=price_open,
            price_close=price_close,
            actual_direction=actual_direction,
        ))

        current += timedelta(minutes=15)

    logger.info(f"Generated {len(windows)} windows")
    if windows:
        logger.info(f"  Range: {windows[0].start_time} → {windows[-1].start_time}")
        bullish = sum(1 for w in windows if w.actual_direction == "BULLISH")
        logger.info(f"  Distribution: {bullish} BULLISH / {len(windows) - bullish} BEARISH")

    return windows
