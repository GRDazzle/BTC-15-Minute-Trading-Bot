"""Tests for sdk.kalshi.account module (AccountManager)."""
import json
import pytest
from pathlib import Path
from sdk.kalshi.account import AccountManager, SubAccount


@pytest.fixture
def state_file(tmp_path):
    return tmp_path / "test_account_state.json"


@pytest.fixture
def mgr(state_file):
    return AccountManager(state_path=state_file)


# -- creation and listing --------------------------------------------------

def test_create_account(mgr):
    acct = mgr.create_account("btc_bot", initial_balance=50.0)
    assert acct.name == "btc_bot"
    assert acct.balance_dollars == 50.0
    assert acct.pnl_dollars == 0.0


def test_create_duplicate_raises(mgr):
    mgr.create_account("alpha")
    with pytest.raises(ValueError, match="already exists"):
        mgr.create_account("alpha")


def test_list_accounts(mgr):
    mgr.create_account("a", 10.0)
    mgr.create_account("b", 20.0)
    accounts = mgr.list_accounts()
    assert len(accounts) == 2
    names = {a.name for a in accounts}
    assert names == {"a", "b"}


def test_get_account_not_found(mgr):
    with pytest.raises(KeyError, match="not found"):
        mgr.get_account("nonexistent")


# -- persistence -----------------------------------------------------------

def test_persistence(state_file):
    mgr1 = AccountManager(state_path=state_file)
    mgr1.create_account("persist_test", 100.0)
    mgr1.save()

    mgr2 = AccountManager(state_path=state_file)
    acct = mgr2.get_account("persist_test")
    assert acct.balance_dollars == 100.0


# -- fund management -------------------------------------------------------

def test_deposit_to_target(mgr):
    mgr.create_account("target", 10.0)
    mgr.deposit(25.0, target="target")
    assert mgr.get_account("target").balance_dollars == 35.0


def test_deposit_distributed(mgr):
    mgr.create_account("a", 10.0)
    mgr.create_account("b", 20.0)
    mgr.deposit(10.0)  # no target => split evenly
    assert mgr.get_account("a").balance_dollars == 15.0
    assert mgr.get_account("b").balance_dollars == 25.0


def test_deposit_negative_raises(mgr):
    mgr.create_account("x")
    with pytest.raises(ValueError, match="positive"):
        mgr.deposit(-5.0, target="x")


def test_withdraw(mgr):
    mgr.create_account("w", 50.0)
    mgr.withdraw(20.0, source="w")
    assert mgr.get_account("w").balance_dollars == 30.0


def test_withdraw_insufficient_raises(mgr):
    mgr.create_account("w", 10.0)
    with pytest.raises(ValueError, match="Insufficient"):
        mgr.withdraw(20.0, source="w")


def test_transfer(mgr):
    mgr.create_account("src", 100.0)
    mgr.create_account("dst", 0.0)
    mgr.transfer(40.0, from_account="src", to_account="dst")
    assert mgr.get_account("src").balance_dollars == 60.0
    assert mgr.get_account("dst").balance_dollars == 40.0


def test_transfer_insufficient_raises(mgr):
    mgr.create_account("src", 10.0)
    mgr.create_account("dst", 0.0)
    with pytest.raises(ValueError, match="Insufficient"):
        mgr.transfer(20.0, from_account="src", to_account="dst")


# -- order lifecycle -------------------------------------------------------

def test_reserve_and_release(mgr):
    mgr.create_account("r", 100.0)
    mgr.reserve("r", 30.0)
    acct = mgr.get_account("r")
    assert acct.reserved_dollars == 30.0
    assert acct.balance_dollars == 100.0  # balance unchanged until fill

    mgr.release_reserve("r", 30.0)
    assert mgr.get_account("r").reserved_dollars == 0.0


def test_reserve_insufficient_raises(mgr):
    mgr.create_account("r", 50.0)
    mgr.reserve("r", 40.0)
    with pytest.raises(ValueError, match="Insufficient available"):
        mgr.reserve("r", 20.0)  # only 10 available


def test_record_fill(mgr):
    mgr.create_account("f", 100.0)
    mgr.reserve("f", 30.0)
    mgr.record_fill("f", cost_dollars=25.0, fees_dollars=2.0)
    acct = mgr.get_account("f")
    assert acct.balance_dollars == 73.0  # 100 - 25 - 2
    assert acct.pnl_dollars == -27.0
    assert acct.reserved_dollars == 3.0  # 30 - 27


def test_record_settlement(mgr):
    mgr.create_account("s", 50.0)
    mgr.record_settlement("s", revenue_dollars=100.0)
    acct = mgr.get_account("s")
    assert acct.balance_dollars == 150.0
    assert acct.pnl_dollars == 100.0


# -- reconciliation -------------------------------------------------------

def test_reconcile_no_discrepancy(mgr):
    mgr.create_account("a", 50.0)
    mgr.create_account("b", 50.0)
    result = mgr.reconcile(actual_kalshi_balance=100.0)
    assert result["discrepancy"] == 0.0
    assert result["local_total"] == 100.0


def test_reconcile_with_discrepancy(mgr):
    mgr.create_account("a", 50.0)
    mgr.create_account("b", 50.0)
    result = mgr.reconcile(actual_kalshi_balance=110.0)
    assert result["discrepancy"] == 10.0
    assert "a" in result["accounts"]
    assert "b" in result["accounts"]
