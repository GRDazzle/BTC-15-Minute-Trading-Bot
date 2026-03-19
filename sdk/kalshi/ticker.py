"""Ticker and series mapping utilities for Kalshi 15-minute markets."""
from __future__ import annotations

from typing import Optional

SERIES_MAP: dict[str, str] = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
}


def series_for_asset(asset: str) -> str:
    """Map an asset symbol (e.g. 'BTC') to its Kalshi series ticker.

    Raises KeyError if the asset is not supported.
    """
    key = asset.upper()
    if key not in SERIES_MAP:
        raise KeyError(f"Unknown asset '{asset}'. Supported: {list(SERIES_MAP.keys())}")
    return SERIES_MAP[key]


def extract_window_id(event_ticker: str) -> Optional[str]:
    """Extract window ID from event ticker.

    Example: 'KXBTC15M-26MAR092230' -> '26MAR092230'
    """
    if event_ticker and "-" in event_ticker:
        return event_ticker.rsplit("-", 1)[-1]
    return None
