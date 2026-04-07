"""Order helpers for Kalshi 15-minute markets.

Ported from TradingBot/live_bot.py with a cleaner interface.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from .client import KalshiClient
from .models import OrderResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: parse order response body
# ---------------------------------------------------------------------------
def _parse_order_response(body: dict[str, Any]) -> OrderResult:
    """Extract an OrderResult from a Kalshi order API response."""
    order = body.get("order", body)

    def _int_field(key: str) -> int:
        val = order.get(key, "0")
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    filled = _int_field("fill_count_fp")
    remaining = _int_field("remaining_count_fp")

    taker_cost = order.get("taker_fill_cost_dollars", "0")
    maker_cost = order.get("maker_fill_cost_dollars", "0")
    try:
        cost = float(taker_cost) + float(maker_cost)
    except (ValueError, TypeError):
        cost = 0.0

    try:
        fees = float(order.get("taker_fees_dollars", "0"))
    except (ValueError, TypeError):
        fees = 0.0

    return OrderResult(
        order_id=order.get("order_id"),
        status=order.get("status", ""),
        filled=filled,
        remaining=remaining,
        cost_dollars=cost,
        fees_dollars=fees,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def place_limit_order(
    client: KalshiClient,
    market_ticker: str,
    side: str,
    action: str,
    price_cents: int,
    count: int,
) -> OrderResult:
    """Place a limit order on Kalshi.

    Args:
        client: authenticated KalshiClient
        market_ticker: e.g. ``"KXBTC15M-26MAR18-T1200-B87500"``
        side: ``"yes"`` or ``"no"``
        action: ``"buy"`` or ``"sell"``
        price_cents: price in cents (1-99)
        count: number of contracts

    Returns:
        OrderResult with order_id, status, filled, remaining, cost, fees
    """
    price_dollars = f"{price_cents / 100:.2f}"
    payload: dict[str, Any] = {
        "ticker": market_ticker,
        "side": side,
        "action": action,
        "type": "limit",
        "count": count,
        "client_order_id": str(uuid.uuid4()),
    }
    if side == "yes":
        payload["yes_price_dollars"] = price_dollars
    else:
        payload["no_price_dollars"] = price_dollars

    logger.info(
        "[order] placing: %s %s %s @ %dc x%d",
        market_ticker, action, side, price_cents, count,
    )

    try:
        st, body = client.create_order(payload)
    except Exception as e:
        logger.warning("[order] exception: %s", e)
        return OrderResult(status="error")

    # Retry once on 429
    if st == 429:
        logger.warning("[order] rate limited, backing off 1s...")
        time.sleep(1)
        try:
            st, body = client.create_order(payload)
        except Exception as e:
            logger.warning("[order] retry exception: %s", e)
            return OrderResult(status="error")

    if st == 401:
        return OrderResult(status="auth_failure")

    if st not in (200, 201):
        logger.warning("[order] failed status=%s body=%s", st, body)
        return OrderResult(status="error")

    result = _parse_order_response(body)
    logger.info(
        "[order] result: status=%s filled=%d remaining=%d cost=$%.2f id=%s",
        result.status, result.filled, result.remaining,
        result.cost_dollars, result.order_id,
    )
    return result


def check_order(client: KalshiClient, order_id: str) -> OrderResult:
    """Check an order's current status."""
    try:
        st, body = client.get_order(order_id)
        if st == 200:
            return _parse_order_response(body)
    except Exception as e:
        logger.warning("[order] check failed for %s: %s", order_id, e)
    return OrderResult(order_id=order_id, status="unknown")


def cancel_order(client: KalshiClient, order_id: str) -> OrderResult:
    """Cancel an order. Returns the final order state (fills preserved)."""
    try:
        st, body = client.cancel_order(order_id)
        if st == 200:
            result = _parse_order_response(body)
            logger.info(
                "[order] cancelled %s: filled=%d", order_id, result.filled,
            )
            return result
    except Exception as e:
        logger.warning("[order] cancel failed for %s: %s", order_id, e)
    return OrderResult(order_id=order_id, status="unknown")


def fetch_balance(client: KalshiClient) -> Optional[float]:
    """Fetch Kalshi account balance in dollars. Returns None on error."""
    try:
        st, data = client.get_balance()
        if st != 200:
            logger.warning("balance fetch failed status=%s", st)
            return None
        balance_cents = data.get("balance", 0)
        return balance_cents / 100.0
    except Exception as e:
        logger.warning("balance fetch error: %s", e)
        return None


def fetch_settlements(client: KalshiClient, limit: int = 100) -> list[dict[str, Any]]:
    """Fetch personal settlement history. Returns list of settlement dicts."""
    try:
        st, data = client.get_settlements(limit=limit)
        if st != 200:
            logger.warning("settlements fetch failed status=%s", st)
            return []
        return data.get("settlements", []) or []
    except Exception as e:
        logger.warning("settlements fetch error: %s", e)
        return []


def find_settlement_for_ticker(
    client: KalshiClient,
    market_ticker: str,
    limit: int = 200,
) -> Optional[dict[str, Any]]:
    """Look up a settlement record for a specific market ticker.

    Returns the matching settlement dict if present in the most recent
    ``limit`` settlements, otherwise None. Used by the post-settlement
    verification step to fetch Kalshi's actual revenue for a trade.
    """
    settlements = fetch_settlements(client, limit=limit)
    for s in settlements:
        if s.get("ticker") == market_ticker or s.get("market_ticker") == market_ticker:
            return s
    return None


def fetch_positions(
    client: KalshiClient,
    *,
    ticker: str | None = None,
    event_ticker: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch currently-held positions. Returns list of position dicts."""
    try:
        st, data = client.get_positions(
            ticker=ticker, event_ticker=event_ticker, limit=limit,
        )
        if st != 200:
            logger.warning("positions fetch failed status=%s", st)
            return []
        # Kalshi returns positions under either "positions" or "market_positions"
        return (
            data.get("market_positions")
            or data.get("positions")
            or []
        )
    except Exception as e:
        logger.warning("positions fetch error: %s", e)
        return []


def fetch_resting_orders(
    client: KalshiClient,
    *,
    ticker: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch unfilled (resting) orders.

    For 15-min markets these should typically be empty -- our orders are
    fill-or-kill within the same window. Any resting orders at startup
    likely indicate the bot was killed mid-window.
    """
    try:
        st, data = client.get_orders(status="resting", ticker=ticker, limit=limit)
        if st != 200:
            logger.warning("resting orders fetch failed status=%s", st)
            return []
        return data.get("orders", []) or []
    except Exception as e:
        logger.warning("resting orders fetch error: %s", e)
        return []


def cancel_all_resting(
    client: KalshiClient,
    *,
    ticker: str | None = None,
) -> int:
    """Cancel every resting order. Returns the number cancelled."""
    orders = fetch_resting_orders(client, ticker=ticker)
    cancelled = 0
    for order in orders:
        order_id = order.get("order_id")
        if not order_id:
            continue
        try:
            result = cancel_order(client, order_id)
            if result.status not in ("error", "unknown"):
                cancelled += 1
        except Exception as e:
            logger.warning("cancel_all_resting: failed to cancel %s: %s", order_id, e)
    return cancelled


def fetch_fills(
    client: KalshiClient,
    *,
    order_id: str | None = None,
    ticker: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch fill history. Useful for reconciling actual fill prices."""
    try:
        st, data = client.get_fills(order_id=order_id, ticker=ticker, limit=limit)
        if st != 200:
            logger.warning("fills fetch failed status=%s", st)
            return []
        return data.get("fills", []) or []
    except Exception as e:
        logger.warning("fills fetch error: %s", e)
        return []
