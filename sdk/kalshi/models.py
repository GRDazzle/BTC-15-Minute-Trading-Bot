"""Simple dataclasses for Kalshi SDK records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PollRecord:
    """A single market poll snapshot."""
    type: str = "poll"
    ts: str = ""
    series: str = ""
    event_ticker: str = ""
    market_ticker: str = ""
    close_time: str = ""
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    no_bid: Optional[int] = None
    no_ask: Optional[int] = None
    volume: Optional[int] = None
    oi: Optional[int] = None
    mins_to_close: Optional[float] = None


@dataclass
class OutcomeRecord:
    """Settlement outcome for an event."""
    type: str = "outcome"
    ts: str = ""
    series: str = ""
    event_ticker: str = ""
    outcome: Optional[str] = None  # "yes" | "no" | None
    outcome_source: str = ""


@dataclass
class OrderResult:
    """Parsed result from an order placement or status check."""
    order_id: Optional[str] = None
    status: str = ""
    filled: int = 0
    remaining: int = 0
    cost_dollars: float = 0.0
    fees_dollars: float = 0.0
