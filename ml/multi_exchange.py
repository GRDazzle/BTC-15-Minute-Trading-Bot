"""Multi-exchange data loader for training.

Loads Kraken tick data and Bitstamp OHLCV data alongside Coinbase ticks,
aligned by timestamp. Provides functions to look up cross-exchange
prices and volumes at any point in time.
"""
from __future__ import annotations

import bisect
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class KrakenTickIndex:
    """Pre-loads Kraken tick data indexed by timestamp for O(log n) lookups."""

    def __init__(self, data_dir: Path, asset: str, min_date: str = None, max_date: str = None):
        self._timestamps: list[datetime] = []
        self._ticks: list[dict] = []

        asset_dir = data_dir / asset
        if not asset_dir.exists():
            return

        for csv_file in sorted(asset_dir.glob("*-aggTrades-*.csv")):
            file_date = csv_file.stem.split("aggTrades-")[-1] if "aggTrades-" in csv_file.stem else ""
            if min_date and file_date < min_date:
                continue
            if max_date and file_date > max_date:
                continue

            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 7:
                        continue
                    try:
                        price = float(row[1])
                        qty = float(row[2])
                        ts_raw = int(row[5])
                        is_buyer = row[6].strip().lower() in ("true", "1")
                        # Auto-detect timestamp unit: ms (13 digits) vs us (16 digits)
                        # Binance Data Vision and recent Kraken writer use us.
                        # Older Kraken REST fetches use ms.
                        if ts_raw > 10**14:  # microseconds
                            ts = datetime.fromtimestamp(ts_raw / 1_000_000, tz=timezone.utc)
                        else:  # milliseconds
                            ts = datetime.fromtimestamp(ts_raw / 1000, tz=timezone.utc)
                    except (ValueError, IndexError, OSError, OverflowError):
                        continue

                    self._timestamps.append(ts)
                    self._ticks.append({
                        "ts": ts,
                        "price": price,
                        "qty": qty,
                        "is_buyer": is_buyer,
                    })

    def n_ticks(self) -> int:
        return len(self._ticks)

    def get_price_at(self, timestamp: datetime) -> Optional[float]:
        """Get the most recent Kraken price at or before timestamp."""
        if not self._timestamps:
            return None
        idx = bisect.bisect_right(self._timestamps, timestamp) - 1
        if idx < 0:
            return None
        return self._ticks[idx]["price"]

    def get_ticks_in_window(self, start: datetime, end: datetime) -> list[dict]:
        """Get all Kraken ticks between start and end."""
        if not self._timestamps:
            return []
        i_start = bisect.bisect_left(self._timestamps, start)
        i_end = bisect.bisect_right(self._timestamps, end)
        return self._ticks[i_start:i_end]


class BitstampOHLCVIndex:
    """Pre-loads Bitstamp 1-minute OHLCV data indexed by timestamp."""

    def __init__(self, data_dir: Path, asset: str, min_date: str = None, max_date: str = None):
        self._timestamps: list[datetime] = []
        self._candles: list[dict] = []

        asset_dir = data_dir / asset
        if not asset_dir.exists():
            return

        for csv_file in sorted(asset_dir.glob("*-ohlcv-*.csv")):
            file_date = csv_file.stem.split("ohlcv-")[-1] if "ohlcv-" in csv_file.stem else ""
            if min_date and file_date < min_date:
                continue
            if max_date and file_date > max_date:
                continue

            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc)
                        candle = {
                            "ts": ts,
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"]),
                        }
                    except (ValueError, KeyError):
                        continue

                    self._timestamps.append(ts)
                    self._candles.append(candle)

    def n_candles(self) -> int:
        return len(self._candles)

    def get_price_at(self, timestamp: datetime) -> Optional[float]:
        """Get the most recent Bitstamp close price at or before timestamp."""
        if not self._timestamps:
            return None
        idx = bisect.bisect_right(self._timestamps, timestamp) - 1
        if idx < 0:
            return None
        return self._candles[idx]["close"]

    def get_candle_at(self, timestamp: datetime) -> Optional[dict]:
        """Get the most recent 1-min candle at or before timestamp."""
        if not self._timestamps:
            return None
        idx = bisect.bisect_right(self._timestamps, timestamp) - 1
        if idx < 0:
            return None
        return self._candles[idx]

    def get_volume_in_window(self, start: datetime, end: datetime) -> float:
        """Sum Bitstamp volume between start and end."""
        if not self._timestamps:
            return 0.0
        i_start = bisect.bisect_left(self._timestamps, start)
        i_end = bisect.bisect_right(self._timestamps, end)
        return sum(c["volume"] for c in self._candles[i_start:i_end])
