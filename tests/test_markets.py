"""Tests for sdk.kalshi.markets price helpers (no network calls)."""
from sdk.kalshi.markets import _bid_cents, _ask_cents


def test_bid_cents_legacy_int():
    m = {"yes_bid": 45}
    assert _bid_cents(m, "yes") == 45


def test_bid_cents_dollars_string():
    m = {"yes_bid_dollars": "0.5600"}
    assert _bid_cents(m, "yes") == 56


def test_bid_cents_prefers_legacy():
    m = {"yes_bid": 50, "yes_bid_dollars": "0.5600"}
    assert _bid_cents(m, "yes") == 50


def test_bid_cents_missing():
    assert _bid_cents({}, "yes") is None


def test_ask_cents_legacy_int():
    m = {"no_ask": 55}
    assert _ask_cents(m, "no") == 55


def test_ask_cents_dollars_string():
    m = {"no_ask_dollars": "0.4200"}
    assert _ask_cents(m, "no") == 42


def test_ask_cents_missing():
    assert _ask_cents({}, "no") is None
