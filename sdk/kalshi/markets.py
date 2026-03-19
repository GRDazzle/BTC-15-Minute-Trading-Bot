"""Market data helpers for Kalshi 15-minute series.

One-shot fetches only — no continuous polling.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from .client import KalshiClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price helpers — handle both legacy int-cents and new dollar-string fields
# ---------------------------------------------------------------------------
def _to_cents(val: Any) -> Optional[int]:
    """Convert a price value to integer cents.

    Handles both legacy integer-cents (e.g. 68) and new dollar-floats/strings
    (e.g. 0.68 or "0.68").  Values < 1.0 are treated as dollars; values >= 1
    that are whole numbers are treated as cents.
    """
    if val is None:
        return None
    try:
        f = float(val)
    except (ValueError, TypeError):
        return None
    # Dollar-format: values like 0.68, 0.05, 0.00
    # Cent-format: values like 68, 5, 100
    # Heuristic: if 0 <= f <= 1.0, it's dollars. If > 1.0, it's cents.
    # Edge case: 1 cent (0.01 dollars) vs 1 cent (int 1) — both give 1.
    if f <= 1.0:
        return int(round(f * 100))
    return int(round(f))


def _bid_cents(market: dict[str, Any], side: str) -> Optional[int]:
    """Get bid in cents, supporting both old int and new dollars-string formats."""
    # Try explicit dollars field first (unambiguous)
    val_dollars = market.get(f"{side}_bid_dollars")
    if val_dollars is not None:
        return _to_cents_dollars(val_dollars)
    return _to_cents(market.get(f"{side}_bid"))


def _ask_cents(market: dict[str, Any], side: str) -> Optional[int]:
    """Get ask in cents, supporting both old int and new dollars-string formats."""
    val_dollars = market.get(f"{side}_ask_dollars")
    if val_dollars is not None:
        return _to_cents_dollars(val_dollars)
    return _to_cents(market.get(f"{side}_ask"))


def _to_cents_dollars(val: Any) -> Optional[int]:
    """Convert an explicit dollar-denominated value to cents."""
    if val is None:
        return None
    try:
        return int(round(float(val) * 100))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Fetch current market (one-shot)
# ---------------------------------------------------------------------------
def fetch_current_market(
    client: KalshiClient,
    series: str,
) -> Optional[dict[str, Any]]:
    """Fetch the nearest open market for a series.

    Returns a dict with:
      market_ticker, event_ticker, close_time, yes_bid, yes_ask,
      no_bid, no_ask, volume, oi, mins_to_close

    Returns None if no open market is found.
    """
    try:
        st, data = client.get_markets(series_ticker=series, status="open")
        if st != 200:
            logger.warning("markets fetch failed for %s status=%s", series, st)
            return None

        markets = data.get("markets", []) or []
        now = datetime.now(timezone.utc)
        best = None
        best_dt = None

        for mk in markets:
            ct = mk.get("close_time")
            if not ct:
                continue
            try:
                dtobj = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            except Exception:
                continue
            if dtobj <= now:
                continue
            if best_dt is None or dtobj < best_dt:
                best_dt = dtobj
                best = mk

        if not best:
            return None

        mins_to_close = None
        if best_dt is not None:
            mins_to_close = (best_dt - now).total_seconds() / 60.0

        return {
            "market_ticker": best.get("ticker"),
            "event_ticker": best.get("event_ticker"),
            "close_time": best.get("close_time"),
            "yes_bid": _bid_cents(best, "yes"),
            "yes_ask": _ask_cents(best, "yes"),
            "no_bid": _bid_cents(best, "no"),
            "no_ask": _ask_cents(best, "no"),
            "volume": best.get("volume") or best.get("volume_fp"),
            "oi": best.get("open_interest") or best.get("open_interest_fp"),
            "mins_to_close": mins_to_close,
        }
    except Exception as e:
        logger.warning("fetch_current_market error for %s: %s", series, e)
        return None


# ---------------------------------------------------------------------------
# Fetch event outcome (with 429 retry)
# ---------------------------------------------------------------------------
def fetch_event_outcome(
    client: KalshiClient,
    event_ticker: str,
) -> Optional[str]:
    """Look up a settled event's result.

    Returns ``"yes"``, ``"no"``, or ``None`` (not yet settled / error).
    Retries on HTTP 429 with exponential backoff.
    """
    if not event_ticker:
        return None

    def _try_fetch(auth: bool) -> tuple[int, Any]:
        return client.get_markets(event_ticker=event_ticker)

    max_attempts = 4
    backoff = 0.5

    try:
        st, data = _try_fetch(True)
        attempts = 1

        while st == 429 and attempts < max_attempts:
            time.sleep(backoff)
            backoff *= 2
            st, data = _try_fetch(True)
            attempts += 1

        if st != 200:
            # Retry with a fresh call
            st, data = _try_fetch(False)
            attempts += 1
            while st == 429 and attempts < max_attempts:
                time.sleep(backoff)
                backoff *= 2
                st, data = _try_fetch(False)
                attempts += 1

        if st != 200:
            logger.warning(
                "event outcome lookup failed for %s status=%s", event_ticker, st,
            )
            return None

        markets = data.get("markets", []) or []
        for m in markets:
            result = m.get("result")
            if result in ("yes", "no"):
                return result

        logger.debug(
            "event outcome: no result for %s (%d markets checked)",
            event_ticker, len(markets),
        )
        return None
    except Exception as e:
        logger.warning("event outcome lookup error for %s: %s", event_ticker, e)
        return None
