"""
Kalshi JSONL data loader for PnL backtesting.

Parses Kalshi polling data (sub-second bid/ask snapshots + settlement outcomes)
from JSONL files at TradingBot/data/KX{ASSET}15M/.

JSONL entry types:
  - poll: {"type": "poll", "ts": "...", "event_ticker": "...", "close_time": "...",
           "yes_bid": 42, "yes_ask": 43, "no_bid": 57, "no_ask": 58,
           "mins_to_close": 14.46, ...}
  - outcome: {"type": "outcome", "event_ticker": "...", "outcome": "yes"|"no", ...}
"""
import json
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class KalshiWindow:
    """One 15-minute Kalshi market window with polls + outcome."""
    event_ticker: str
    close_time: datetime              # UTC, when this market expires
    polls: list[dict] = field(default_factory=list)
    # Each poll: {"ts": datetime, "yes_bid": int, "yes_ask": int,
    #             "no_bid": int, "no_ask": int, "mins_to_close": float}
    outcome: Optional[str] = None     # "yes" or "no"

    # Cached sorted timestamps for bisect lookup
    _poll_timestamps: list[float] = field(default_factory=list, repr=False)

    def build_index(self):
        """Sort polls by ts and build timestamp index for bisect lookup."""
        self.polls.sort(key=lambda p: p["ts"])
        self._poll_timestamps = [p["ts"].timestamp() for p in self.polls]


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO 8601 timestamp string to UTC datetime."""
    # Handle both "2026-03-13T04:00:32.168170+00:00" and "2026-03-13T04:15:00Z"
    ts_str = ts_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_kalshi_windows(data_dir: Path, asset: str) -> dict[datetime, KalshiWindow]:
    """Load all JSONL files for an asset.

    Args:
        data_dir: Root data directory containing KX{ASSET}15M/ subdirs
                  (e.g., TradingBot/data/)
        asset: e.g. "BTC"

    Returns:
        Dict keyed by close_time (UTC datetime) -> KalshiWindow.
    """
    series = f"KX{asset.upper()}15M"
    asset_dir = data_dir / series
    if not asset_dir.exists():
        logger.error("Kalshi data directory not found: {}", asset_dir)
        return {}

    jsonl_files = sorted(asset_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error("No JSONL files found in {}", asset_dir)
        return {}

    logger.info("Loading {} Kalshi JSONL files for {}", len(jsonl_files), asset)

    # Collect polls and outcomes by event_ticker
    polls_by_event: dict[str, list[dict]] = {}
    close_times: dict[str, datetime] = {}
    outcomes: dict[str, str] = {}

    for jsonl_path in jsonl_files:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type", "")

                if entry_type == "poll":
                    event_ticker = entry.get("event_ticker", "")
                    if not event_ticker:
                        continue

                    ts = _parse_ts(entry["ts"])
                    close_time = _parse_ts(entry["close_time"])

                    poll = {
                        "ts": ts,
                        "yes_bid": int(entry.get("yes_bid", 0)),
                        "yes_ask": int(entry.get("yes_ask", 0)),
                        "no_bid": int(entry.get("no_bid", 0)),
                        "no_ask": int(entry.get("no_ask", 0)),
                        "mins_to_close": float(entry.get("mins_to_close", 0)),
                    }

                    if event_ticker not in polls_by_event:
                        polls_by_event[event_ticker] = []
                        close_times[event_ticker] = close_time
                    polls_by_event[event_ticker].append(poll)

                elif entry_type == "outcome":
                    event_ticker = entry.get("event_ticker", "")
                    outcome = entry.get("outcome", "")
                    if event_ticker and outcome:
                        outcomes[event_ticker] = outcome

    # Build KalshiWindow objects indexed by close_time
    windows: dict[datetime, KalshiWindow] = {}
    matched_outcomes = 0
    missing_outcomes = 0

    for event_ticker, polls in polls_by_event.items():
        close_time = close_times[event_ticker]
        outcome = outcomes.get(event_ticker)

        if outcome is None:
            missing_outcomes += 1
            continue

        matched_outcomes += 1
        kw = KalshiWindow(
            event_ticker=event_ticker,
            close_time=close_time,
            polls=polls,
            outcome=outcome,
        )
        kw.build_index()
        windows[close_time] = kw

    logger.info(
        "Loaded {} Kalshi windows for {} ({} with outcomes, {} missing outcomes)",
        len(windows), asset, matched_outcomes, missing_outcomes,
    )

    if windows:
        dates = sorted(set(ct.date() for ct in windows.keys()))
        logger.info("  Date range: {} -> {} ({} days)", dates[0], dates[-1], len(dates))

    return windows


def get_kalshi_prices(window: KalshiWindow, target_ts: datetime) -> Optional[dict]:
    """Find the nearest poll to target_ts using bisect.

    Args:
        window: KalshiWindow with sorted polls and built index
        target_ts: The timestamp to find prices for

    Returns:
        {"yes_bid": int, "yes_ask": int, "no_bid": int, "no_ask": int,
         "mins_to_close": float}
        or None if no polls available.
    """
    if not window.polls or not window._poll_timestamps:
        return None

    target_epoch = target_ts.timestamp()
    idx = bisect_left(window._poll_timestamps, target_epoch)

    # Find closest poll (check idx and idx-1)
    best_idx = None
    best_diff = float("inf")

    for candidate in (idx - 1, idx):
        if 0 <= candidate < len(window._poll_timestamps):
            diff = abs(window._poll_timestamps[candidate] - target_epoch)
            if diff < best_diff:
                best_diff = diff
                best_idx = candidate

    if best_idx is None:
        return None

    poll = window.polls[best_idx]
    return {
        "yes_bid": poll["yes_bid"],
        "yes_ask": poll["yes_ask"],
        "no_bid": poll["no_bid"],
        "no_ask": poll["no_ask"],
        "mins_to_close": poll["mins_to_close"],
    }
