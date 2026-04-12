"""Kalshi poll data loader for training feature extraction.

Loads JSONL poll files, indexes them by event_ticker and timestamp,
and provides O(log n) lookups for the most recent poll at any point in time.
Used by generate_training_data.py to join Kalshi market state with tick features.
"""
from __future__ import annotations

import bisect
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

SERIES_MAP = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
    "HYPE": "KXHYPE15M",
    "BNB": "KXBNB15M",
    "DOGE": "KXDOGE15M",
}


class KalshiPollIndex:
    """Pre-loads Kalshi poll data and provides fast lookups.

    For each event_ticker, stores a sorted list of (timestamp, poll_dict)
    so that find_poll(event_ticker, timestamp) returns the most recent
    poll at or before that timestamp in O(log n).
    """

    def __init__(self, polls_dir: Path, asset: str, min_date: str = None, max_date: str = None):
        """Load all JSONL files for the given asset's series.

        Args:
            polls_dir: Path to data/kalshi_polls/
            asset: e.g. "SOL"
            min_date: Optional YYYY-MM-DD string, skip files before this date
            max_date: Optional YYYY-MM-DD string, skip files after this date
        """
        self._polls: dict[str, list[tuple[datetime, dict]]] = {}
        self._outcomes: dict[str, str] = {}
        series = SERIES_MAP.get(asset.upper())
        if not series:
            return

        series_dir = polls_dir / series
        if not series_dir.exists():
            return

        for jsonl_file in sorted(series_dir.glob("*.jsonl")):
            file_date = jsonl_file.name[:10]
            if min_date and file_date < min_date:
                continue
            if max_date and file_date > max_date:
                continue

            with open(jsonl_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue

                    if entry.get("type") == "outcome":
                        et = entry.get("event_ticker", "")
                        self._outcomes[et] = entry.get("outcome", "")

                    elif entry.get("type") == "poll":
                        et = entry.get("event_ticker", "")
                        ts_str = entry.get("ts", "")
                        try:
                            ts = datetime.fromisoformat(ts_str)
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                        except Exception:
                            continue

                        poll = {
                            "yes_ask": int(entry.get("yes_ask", 50)),
                            "yes_bid": int(entry.get("yes_bid", 50)),
                            "no_ask": int(entry.get("no_ask", 50)),
                            "no_bid": int(entry.get("no_bid", 50)),
                            "mins_to_close": float(entry.get("mins_to_close", 7.5)),
                        }

                        if et not in self._polls:
                            self._polls[et] = []
                        self._polls[et].append((ts, poll))

        # Sort each event's polls by timestamp
        for et in self._polls:
            self._polls[et].sort(key=lambda x: x[0])

    @property
    def event_tickers(self) -> set[str]:
        return set(self._polls.keys())

    @property
    def outcomes(self) -> dict[str, str]:
        return self._outcomes

    def n_polls(self) -> int:
        return sum(len(v) for v in self._polls.values())

    def find_poll(self, event_ticker: str, timestamp: datetime) -> Optional[dict]:
        """Find the most recent poll for event_ticker at or before timestamp.

        Returns a dict with keys: yes_ask, yes_bid, no_ask, no_bid, mins_to_close.
        Returns None if no poll data exists for this event_ticker.
        """
        polls = self._polls.get(event_ticker)
        if not polls:
            return None

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Binary search for the rightmost poll with ts <= timestamp
        timestamps = [p[0] for p in polls]
        idx = bisect.bisect_right(timestamps, timestamp) - 1

        if idx < 0:
            return None

        return polls[idx][1]

    def get_outcome(self, event_ticker: str) -> Optional[str]:
        return self._outcomes.get(event_ticker)


def window_start_to_event_ticker(asset: str, window_start: datetime) -> str:
    """Convert a window start (UTC close_time) to a Kalshi event_ticker.

    Kalshi event tickers encode the time in US Eastern Time (ET = UTC-4).
    The window_start we pass is the UTC close_time of the window.

    e.g. SOL, close_time=2026-04-07 00:15:00 UTC
         -> ET: 2026-04-06 20:15
         -> KXSOL15M-26APR062015
    """
    series = SERIES_MAP.get(asset.upper(), f"KX{asset.upper()}15M")
    # Convert UTC -> US Eastern (EDT=UTC-4, EST=UTC-5)
    try:
        from zoneinfo import ZoneInfo
        et = window_start.astimezone(ZoneInfo("America/New_York"))
    except ImportError:
        # Fallback for Python < 3.9: assume EDT (UTC-4), correct Mar-Nov
        et = window_start - timedelta(hours=4)
    month_abbr = et.strftime("%b").upper()
    return f"{series}-{et.strftime('%y')}{month_abbr}{et.strftime('%d%H%M')}"


def window_close_to_event_ticker(asset: str, window_close_utc: datetime) -> str:
    """Convert a window close time (UTC) to a Kalshi event_ticker.

    This is the same as window_start_to_event_ticker but named more clearly.
    The "start" in generate_training_data is actually 15 minutes before the
    Kalshi close_time, so callers should pass window_end (= close_time).
    """
    return window_start_to_event_ticker(asset, window_close_utc)
