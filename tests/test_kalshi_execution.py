"""Tests for KalshiExecutionAdapter — mock-based, no network."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from execution.kalshi_execution import (
    KalshiExecutionAdapter,
    TradeRecord,
    KALSHI_FEE_CENTS,
)
from sdk.kalshi.account import AccountManager, SubAccount


@pytest.fixture
def account_manager(tmp_path: Path) -> AccountManager:
    """Create an AccountManager backed by a temp file with a BTC sub-account."""
    mgr = AccountManager(state_path=tmp_path / "state.json")
    mgr.create_account("KXBTC15M", initial_balance=100.0)
    return mgr


@pytest.fixture
def client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def adapter(client, account_manager) -> KalshiExecutionAdapter:
    return KalshiExecutionAdapter(
        client=client,
        account_manager=account_manager,
        dry_run=True,
        max_contracts_per_trade={"BTC": 10, "_default": 10},
        max_price_cents={"BTC": 85, "_default": 85},
        min_price_cents={"BTC": 15, "_default": 15},
    )


@pytest.fixture
def market_info() -> dict:
    return {
        "market_ticker": "KXBTC15M-26MAR18-T1200-B87500",
        "event_ticker": "KXBTC15M-26MAR18-T1200",
        "close_time": "2026-03-18T12:15:00Z",
        "yes_bid": 48,
        "yes_ask": 52,
        "no_bid": 46,
        "no_ask": 50,
        "volume": 100,
        "oi": 50,
        "mins_to_close": 13.5,
    }


# === Direction mapping ===

class TestDirectionMapping:
    def test_bullish_maps_to_yes(self, adapter, market_info):
        side, price = adapter._determine_side_and_price("BULLISH", market_info)
        assert side == "yes"
        assert price == 52

    def test_bearish_maps_to_no(self, adapter, market_info):
        side, price = adapter._determine_side_and_price("BEARISH", market_info)
        assert side == "no"
        assert price == 50

    def test_bullish_case_insensitive(self, adapter, market_info):
        side, _ = adapter._determine_side_and_price("bullish", market_info)
        assert side == "yes"

    def test_enum_direction_bullish(self, adapter, market_info):
        side, _ = adapter._determine_side_and_price(
            "SignalDirection.BULLISH", market_info,
        )
        assert side == "yes"

    def test_unknown_direction_returns_none_price(self, adapter, market_info):
        side, price = adapter._determine_side_and_price("NEUTRAL", market_info)
        assert side == "yes"
        assert price is None


# === Contract calculation ===

class TestContractCalculation:
    def test_within_budget(self, adapter, account_manager):
        # balance=100, price=50c → cost_per=0.52 → max_by_balance=192
        # scale = 0.8 * (75/100) = 0.6 → desired = max(1, int(192*0.6)) = 115
        # capped at max_contracts_per_trade=10
        count = adapter._calculate_contracts("KXBTC15M", 50, 0.8, 75)
        assert count == 10  # capped

    def test_minimum_one(self, adapter, account_manager):
        # Low confidence/score but has balance
        count = adapter._calculate_contracts("KXBTC15M", 50, 0.1, 10)
        assert count >= 1

    def test_zero_when_no_account(self, adapter):
        count = adapter._calculate_contracts("NONEXISTENT", 50, 0.8, 75)
        assert count == 0

    def test_zero_when_no_balance(self, adapter, account_manager):
        acct = account_manager.get_account("KXBTC15M")
        acct.balance_dollars = 0.0
        count = adapter._calculate_contracts("KXBTC15M", 50, 0.8, 75)
        assert count == 0

    def test_scales_with_confidence(self, adapter, account_manager):
        # Same budget, lower confidence → fewer contracts (but both capped)
        adapter._max_contracts = {"BTC": 100, "_default": 100}  # raise cap
        high = adapter._calculate_contracts("KXBTC15M", 50, 0.9, 90, asset="BTC")
        # reset balance for second call (fill deducted nothing here)
        low = adapter._calculate_contracts("KXBTC15M", 50, 0.3, 30, asset="BTC")
        assert high > low


# === Price bounds ===

class TestPriceBounds:
    def test_rejects_price_above_max(self, adapter):
        assert adapter._validate_price(90, "BTC") is False

    def test_rejects_price_below_min(self, adapter):
        assert adapter._validate_price(10, "BTC") is False

    def test_accepts_price_at_boundary(self, adapter):
        assert adapter._validate_price(15, "BTC") is True
        assert adapter._validate_price(85, "BTC") is True

    def test_accepts_price_in_range(self, adapter):
        assert adapter._validate_price(50, "BTC") is True


# === Dry run execution ===

class TestDryRunExecution:
    def test_dry_run_creates_trade_record(self, adapter, market_info):
        trade = adapter.execute_trade("BTC", "BULLISH", 0.8, 75.0, market_info)
        assert trade is not None
        assert trade.dry_run is True
        assert trade.side == "yes"
        assert trade.filled > 0
        assert trade.cost_dollars > 0

    def test_dry_run_deducts_from_account(self, adapter, account_manager, market_info):
        before = account_manager.get_account("KXBTC15M").balance_dollars
        trade = adapter.execute_trade("BTC", "BULLISH", 0.8, 75.0, market_info)
        after = account_manager.get_account("KXBTC15M").balance_dollars
        assert after < before

    def test_dry_run_does_not_call_sdk(self, adapter, client, market_info):
        adapter.execute_trade("BTC", "BULLISH", 0.8, 75.0, market_info)
        client.create_order.assert_not_called()

    def test_bearish_buys_no_side(self, adapter, market_info):
        trade = adapter.execute_trade("BTC", "BEARISH", 0.8, 75.0, market_info)
        assert trade is not None
        assert trade.side == "no"

    def test_returns_none_on_invalid_price(self, adapter, market_info):
        market_info["yes_ask"] = 95  # above max
        trade = adapter.execute_trade("BTC", "BULLISH", 0.8, 75.0, market_info)
        assert trade is None

    def test_returns_none_on_no_ask_price(self, adapter, market_info):
        market_info["yes_ask"] = None
        trade = adapter.execute_trade("BTC", "BULLISH", 0.8, 75.0, market_info)
        assert trade is None


# === Settlement ===

class TestSettlement:
    def test_winning_yes_credits_revenue(self, adapter, account_manager, market_info):
        trade = adapter.execute_trade("BTC", "BULLISH", 0.8, 75.0, market_info)
        assert trade is not None
        bal_after_trade = account_manager.get_account("KXBTC15M").balance_dollars

        # Simulate outcome (bypass dry-run skip by setting dry_run=False on the trade)
        trade.dry_run = False
        with patch(
            "execution.kalshi_execution.fetch_event_outcome", return_value="yes",
        ):
            adapter.settle_window(trade)

        assert trade.settlement_outcome == "yes"
        assert trade.revenue_dollars == trade.filled * 1.0
        bal_after_settle = account_manager.get_account("KXBTC15M").balance_dollars
        assert bal_after_settle > bal_after_trade

    def test_losing_credits_zero(self, adapter, account_manager, market_info):
        trade = adapter.execute_trade("BTC", "BULLISH", 0.8, 75.0, market_info)
        assert trade is not None
        bal_after_trade = account_manager.get_account("KXBTC15M").balance_dollars

        trade.dry_run = False
        with patch(
            "execution.kalshi_execution.fetch_event_outcome", return_value="no",
        ):
            adapter.settle_window(trade)

        assert trade.settlement_outcome == "no"
        assert trade.revenue_dollars == 0.0
        bal_after_settle = account_manager.get_account("KXBTC15M").balance_dollars
        assert bal_after_settle == bal_after_trade

    def test_no_outcome_yet_leaves_unsettled(self, adapter, market_info):
        trade = adapter.execute_trade("BTC", "BULLISH", 0.8, 75.0, market_info)
        assert trade is not None
        trade.dry_run = False
        with patch(
            "execution.kalshi_execution.fetch_event_outcome", return_value=None,
        ):
            adapter.settle_window(trade)

        assert trade.settlement_outcome is None

    def test_already_settled_is_idempotent(self, adapter, market_info):
        trade = adapter.execute_trade("BTC", "BULLISH", 0.8, 75.0, market_info)
        assert trade is not None
        trade.settlement_outcome = "yes"
        trade.revenue_dollars = 5.0

        # Should return immediately without calling fetch_event_outcome
        result = adapter.settle_window(trade)
        assert result.settlement_outcome == "yes"
        assert result.revenue_dollars == 5.0


# === Cost estimation ===

class TestCostEstimation:
    def test_estimate_cost(self, adapter):
        cost, fees = adapter._estimate_cost(50, 5)
        assert cost == pytest.approx(2.50, abs=0.01)
        assert fees == pytest.approx(0.10, abs=0.01)

    def test_estimate_cost_single_contract(self, adapter):
        cost, fees = adapter._estimate_cost(75, 1)
        assert cost == pytest.approx(0.75, abs=0.01)
        assert fees == pytest.approx(0.02, abs=0.01)
