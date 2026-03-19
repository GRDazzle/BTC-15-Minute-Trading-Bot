"""Tests for sdk.kalshi.orders response parsing (no network calls)."""
from sdk.kalshi.orders import _parse_order_response
from sdk.kalshi.models import OrderResult


def test_parse_order_response_full():
    body = {
        "order": {
            "order_id": "abc-123",
            "status": "executed",
            "fill_count_fp": "10",
            "remaining_count_fp": "0",
            "initial_count_fp": "10",
            "taker_fill_cost_dollars": "5.50",
            "maker_fill_cost_dollars": "0",
            "taker_fees_dollars": "0.11",
        }
    }
    result = _parse_order_response(body)
    assert isinstance(result, OrderResult)
    assert result.order_id == "abc-123"
    assert result.status == "executed"
    assert result.filled == 10
    assert result.remaining == 0
    assert result.cost_dollars == 5.50
    assert result.fees_dollars == 0.11


def test_parse_order_response_missing_fields():
    body = {"order_id": "x", "status": "resting"}
    result = _parse_order_response(body)
    assert result.order_id == "x"
    assert result.filled == 0
    assert result.cost_dollars == 0.0


def test_parse_order_response_partial_fill():
    body = {
        "order": {
            "order_id": "partial-1",
            "status": "resting",
            "fill_count_fp": "3",
            "remaining_count_fp": "7",
            "initial_count_fp": "10",
            "taker_fill_cost_dollars": "1.65",
            "maker_fill_cost_dollars": "0",
            "taker_fees_dollars": "0.03",
        }
    }
    result = _parse_order_response(body)
    assert result.filled == 3
    assert result.remaining == 7
    assert result.status == "resting"
