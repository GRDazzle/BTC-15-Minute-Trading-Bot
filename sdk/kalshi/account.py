"""Local virtual sub-account management.

Kalshi doesn't offer sub-accounts for small clients, so this module provides
a local bookkeeping layer that tracks virtual sub-accounts backed by a single
real Kalshi balance.

State is persisted to a JSON file so it survives bot restarts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("data/account_state.json")


@dataclass
class SubAccount:
    """A virtual sub-account."""
    name: str
    balance_dollars: float = 0.0
    reserved_dollars: float = 0.0
    pnl_dollars: float = 0.0


class AccountManager:
    """Manages virtual sub-accounts backed by a single Kalshi balance."""

    def __init__(self, state_path: Path | str = DEFAULT_STATE_PATH):
        self.state_path = Path(state_path)
        self._accounts: dict[str, SubAccount] = {}
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self):
        if not self.state_path.exists():
            self._accounts = {}
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._accounts = {
                name: SubAccount(**data) for name, data in raw.items()
            }
            logger.info("Loaded %d sub-accounts from %s", len(self._accounts), self.state_path)
        except Exception as e:
            logger.warning("Failed to load account state from %s: %s", self.state_path, e)
            self._accounts = {}

    def save(self):
        """Persist current state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(acct) for name, acct in self._accounts.items()}
        self.state_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

    # -- account CRUD ------------------------------------------------------

    def create_account(self, name: str, initial_balance: float = 0.0) -> SubAccount:
        """Create a new virtual sub-account."""
        if name in self._accounts:
            raise ValueError(f"Sub-account '{name}' already exists")
        acct = SubAccount(name=name, balance_dollars=initial_balance)
        self._accounts[name] = acct
        self.save()
        return acct

    def get_account(self, name: str) -> SubAccount:
        """Get a sub-account by name. Raises KeyError if not found."""
        if name not in self._accounts:
            raise KeyError(f"Sub-account '{name}' not found")
        return self._accounts[name]

    def list_accounts(self) -> list[SubAccount]:
        """List all sub-accounts."""
        return list(self._accounts.values())

    # -- fund management ---------------------------------------------------

    def deposit(self, amount: float, target: Optional[str] = None):
        """Deposit funds. If target is specified, credit that account.
        Otherwise distribute proportionally across all accounts.
        """
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")

        if target is not None:
            acct = self.get_account(target)
            acct.balance_dollars += amount
        else:
            accounts = self.list_accounts()
            if not accounts:
                raise ValueError("No sub-accounts exist to distribute deposit to")
            share = amount / len(accounts)
            for acct in accounts:
                acct.balance_dollars += share
        self.save()

    def withdraw(self, amount: float, source: str):
        """Withdraw funds from a sub-account."""
        if amount <= 0:
            raise ValueError("Withdraw amount must be positive")
        acct = self.get_account(source)
        if acct.balance_dollars < amount:
            raise ValueError(
                f"Insufficient balance in '{source}': "
                f"${acct.balance_dollars:.2f} < ${amount:.2f}"
            )
        acct.balance_dollars -= amount
        self.save()

    def transfer(self, amount: float, from_account: str, to_account: str):
        """Transfer funds between sub-accounts."""
        if amount <= 0:
            raise ValueError("Transfer amount must be positive")
        src = self.get_account(from_account)
        dst = self.get_account(to_account)
        if src.balance_dollars < amount:
            raise ValueError(
                f"Insufficient balance in '{from_account}': "
                f"${src.balance_dollars:.2f} < ${amount:.2f}"
            )
        src.balance_dollars -= amount
        dst.balance_dollars += amount
        self.save()

    # -- order lifecycle ---------------------------------------------------

    def reserve(self, account_name: str, amount: float):
        """Earmark funds for a pending order."""
        acct = self.get_account(account_name)
        available = acct.balance_dollars - acct.reserved_dollars
        if available < amount:
            raise ValueError(
                f"Insufficient available balance in '{account_name}': "
                f"${available:.2f} < ${amount:.2f}"
            )
        acct.reserved_dollars += amount
        self.save()

    def release_reserve(self, account_name: str, amount: float):
        """Release a reservation (order cancelled/expired)."""
        acct = self.get_account(account_name)
        acct.reserved_dollars = max(0.0, acct.reserved_dollars - amount)
        self.save()

    def record_fill(self, account_name: str, cost_dollars: float, fees_dollars: float = 0.0):
        """Deduct cost and fees from balance on fill. Releases the reserved amount."""
        acct = self.get_account(account_name)
        total = cost_dollars + fees_dollars
        acct.balance_dollars -= total
        acct.pnl_dollars -= total
        # Release reservation (the reserved amount should match cost)
        acct.reserved_dollars = max(0.0, acct.reserved_dollars - total)
        self.save()

    def record_settlement(self, account_name: str, revenue_dollars: float):
        """Credit revenue on settlement."""
        acct = self.get_account(account_name)
        acct.balance_dollars += revenue_dollars
        acct.pnl_dollars += revenue_dollars
        self.save()

    # -- reconciliation ----------------------------------------------------

    def reconcile(self, actual_kalshi_balance: float) -> dict:
        """Compare sum of sub-accounts against the real Kalshi balance.

        Returns a dict with:
          local_total, actual_balance, discrepancy, accounts
        """
        local_total = sum(a.balance_dollars for a in self._accounts.values())
        discrepancy = actual_kalshi_balance - local_total
        result = {
            "local_total": round(local_total, 2),
            "actual_balance": round(actual_kalshi_balance, 2),
            "discrepancy": round(discrepancy, 2),
            "accounts": {
                name: {
                    "balance": round(a.balance_dollars, 2),
                    "reserved": round(a.reserved_dollars, 2),
                    "pnl": round(a.pnl_dollars, 2),
                }
                for name, a in self._accounts.items()
            },
        }
        if abs(discrepancy) > 0.01:
            logger.warning(
                "Balance discrepancy: local=$%.2f actual=$%.2f diff=$%.2f",
                local_total, actual_kalshi_balance, discrepancy,
            )
        return result
