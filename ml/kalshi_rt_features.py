"""Real-time Kalshi market microstructure features.

Computes features from Kalshi poll history within a single 15-min window.
Used by both live inference (from strategy poll state) and training data
generation (from JSONL poll files).

Features:
    kalshi_ask_velocity_30s   - rate of change of yes_ask over 30s
    kalshi_spread_change_30s  - is spread widening or narrowing
    kalshi_volume_rate        - contracts traded per minute (recent)
    kalshi_oi_change          - open interest delta from window start
"""
from datetime import datetime, timedelta, timezone
from typing import Optional


def compute_kalshi_rt_features(
    poll_history: list[dict],
    current_ts: datetime,
) -> dict[str, float]:
    """Compute real-time Kalshi features from a list of recent polls.

    Args:
        poll_history: List of poll dicts, each with:
            {ts: datetime, yes_ask: int, yes_bid: int, no_ask: int, no_bid: int,
             volume: float, oi: float}
            Sorted by ts ascending. Should cover at least 30s of history.
        current_ts: Current evaluation timestamp.

    Returns:
        Dict of feature name -> float value.
    """
    features = {
        "kalshi_ask_velocity_30s": 0.0,
        "kalshi_spread_change_30s": 0.0,
        "kalshi_volume_rate": 0.0,
        "kalshi_oi_change": 0.0,
    }

    if not poll_history or len(poll_history) < 2:
        return features

    # Current poll (most recent)
    curr = poll_history[-1]
    curr_ask = curr.get("yes_ask", 50)
    curr_spread = curr_ask - curr.get("yes_bid", curr_ask)
    curr_vol = float(curr.get("volume", 0))
    curr_oi = float(curr.get("oi", 0))

    # Find poll ~30s ago
    target_30s = current_ts - timedelta(seconds=30)
    poll_30s = None
    for p in reversed(poll_history):
        if p["ts"] <= target_30s:
            poll_30s = p
            break

    if poll_30s is not None:
        old_ask = poll_30s.get("yes_ask", 50)
        old_spread = old_ask - poll_30s.get("yes_bid", old_ask)

        # Ask velocity: cents per 30s (positive = market moving toward YES)
        features["kalshi_ask_velocity_30s"] = float(curr_ask - old_ask)

        # Spread change: positive = widening (more uncertainty)
        features["kalshi_spread_change_30s"] = float(curr_spread - old_spread)

    # Volume rate: contracts per minute over last 60s
    poll_60s = None
    target_60s = current_ts - timedelta(seconds=60)
    for p in reversed(poll_history):
        if p["ts"] <= target_60s:
            poll_60s = p
            break

    if poll_60s is not None:
        vol_delta = curr_vol - float(poll_60s.get("volume", 0))
        elapsed_mins = max(0.5, (current_ts - poll_60s["ts"]).total_seconds() / 60.0)
        features["kalshi_volume_rate"] = vol_delta / elapsed_mins
    elif len(poll_history) >= 2:
        # Use whatever history we have
        first = poll_history[0]
        vol_delta = curr_vol - float(first.get("volume", 0))
        elapsed_mins = max(0.5, (current_ts - first["ts"]).total_seconds() / 60.0)
        features["kalshi_volume_rate"] = vol_delta / elapsed_mins

    # OI change from window start (first poll in history)
    first = poll_history[0]
    features["kalshi_oi_change"] = curr_oi - float(first.get("oi", 0))

    return features


# Feature names for registration in ml/features.py
KALSHI_RT_FEATURE_NAMES = [
    "kalshi_ask_velocity_30s",
    "kalshi_spread_change_30s",
    "kalshi_volume_rate",
    "kalshi_oi_change",
]
