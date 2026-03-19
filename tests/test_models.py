"""Tests for sdk.kalshi.models dataclasses."""
from sdk.kalshi.models import PollRecord, OutcomeRecord, OrderResult


def test_poll_record_defaults():
    rec = PollRecord()
    assert rec.type == "poll"
    assert rec.yes_ask is None
    assert rec.mins_to_close is None


def test_poll_record_with_values():
    rec = PollRecord(
        ts="2026-03-18T12:00:00Z",
        series="KXBTC15M",
        event_ticker="KXBTC15M-26MAR18T1200",
        yes_ask=55,
        no_ask=46,
        mins_to_close=14.5,
    )
    assert rec.yes_ask == 55
    assert rec.series == "KXBTC15M"


def test_outcome_record():
    rec = OutcomeRecord(
        ts="2026-03-18T12:15:00Z",
        event_ticker="KXBTC15M-26MAR18T1200",
        outcome="yes",
        outcome_source="settlement_api",
    )
    assert rec.outcome == "yes"
    assert rec.type == "outcome"


def test_order_result_defaults():
    res = OrderResult()
    assert res.order_id is None
    assert res.filled == 0
    assert res.cost_dollars == 0.0
    assert res.fees_dollars == 0.0
