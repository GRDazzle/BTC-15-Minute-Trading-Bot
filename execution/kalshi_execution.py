"""Kalshi execution adapter — bridges FusedSignal to SDK orders.

Maps signal direction to Kalshi side, calculates position sizing from
sub-account balance and confidence, and handles dry-run simulation.

Supports automatic max_contracts scaling: for every doubling of the
sub-account balance relative to its initial balance, max_contracts
increases by 50%.
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sdk.kalshi.client import KalshiClient
from sdk.kalshi.account import AccountManager
from sdk.kalshi.markets import fetch_current_market, fetch_event_outcome
from sdk.kalshi.orders import place_limit_order
from sdk.kalshi.ticker import series_for_asset

logger = logging.getLogger(__name__)

# Kalshi charges ~2c per contract (taker fee)
KALSHI_FEE_CENTS = 2


@dataclass
class TradeRecord:
    """Full audit trail for a single trade."""
    trade_id: str = ""
    asset: str = ""
    series: str = ""
    market_ticker: str = ""
    event_ticker: str = ""
    window_id: str = ""

    # Order details
    side: str = ""          # "yes" or "no"
    action: str = "buy"
    price_cents: int = 0
    count: int = 0
    cost_dollars: float = 0.0
    fees_dollars: float = 0.0

    # Signal context
    direction: str = ""     # "BULLISH" or "BEARISH"
    confidence: float = 0.0
    score: float = 0.0

    # Execution
    order_id: Optional[str] = None
    filled: int = 0
    dry_run: bool = True
    placed_at: Optional[datetime] = None

    # Settlement
    settlement_outcome: Optional[str] = None  # "yes" / "no" / None
    revenue_dollars: float = 0.0
    settled_at: Optional[datetime] = None


class KalshiExecutionAdapter:
    """Bridges FusedSignal decisions to Kalshi SDK order calls."""

    # Class-level defaults used when an asset has no per-asset override
    DEFAULT_MAX_CONTRACTS = 10
    DEFAULT_MAX_PRICE = 85
    DEFAULT_MIN_PRICE = 15

    def __init__(
        self,
        client: KalshiClient,
        account_manager: AccountManager,
        dry_run: bool = True,
        max_contracts_per_trade: dict[str, int] | int = 10,
        max_price_cents: dict[str, int] | int = 85,
        min_price_cents: dict[str, int] | int = 15,
        initial_balances: dict[str, float] | None = None,
    ):
        self.client = client
        self.account_manager = account_manager
        # dry_run can be a bool (global) or dict (per-asset)
        self._dry_run_global = dry_run
        self._dry_run_per_asset: dict[str, bool] = {}
        # Accept either a per-asset dict or a single int (backwards compat)
        self._max_contracts = (
            max_contracts_per_trade
            if isinstance(max_contracts_per_trade, dict)
            else {"_default": max_contracts_per_trade}
        )
        self._max_price = (
            max_price_cents
            if isinstance(max_price_cents, dict)
            else {"_default": max_price_cents}
        )
        self._min_price = (
            min_price_cents
            if isinstance(min_price_cents, dict)
            else {"_default": min_price_cents}
        )
        # Per-asset initial balances for contract scaling
        self._initial_balances: dict[str, float] = initial_balances or {}

    def is_dry_run(self, asset: str) -> bool:
        """Check if an asset is in dry-run mode. Per-asset overrides global."""
        return self._dry_run_per_asset.get(asset, self._dry_run_global)

    def set_dry_run(self, asset: str, dry_run: bool) -> None:
        """Set per-asset dry-run mode."""
        self._dry_run_per_asset[asset] = dry_run

    @property
    def dry_run(self) -> bool:
        """Global dry-run flag (backwards compat)."""
        return self._dry_run_global

    # Hard cap to avoid disrupting Kalshi market liquidity
    MAX_CONTRACTS_CAP = 500

    def _get_max_contracts(self, asset: str) -> int:
        """Get max contracts for an asset, power-scaled by balance ratio.

        Power 0.67: 2x balance = 1.59x contracts, 6x balance = 3.4x contracts.
        Capped at MAX_CONTRACTS_CAP to avoid disrupting market liquidity.
        """
        base = self._max_contracts.get(asset, self._max_contracts.get("_default", self.DEFAULT_MAX_CONTRACTS))
        initial = self._initial_balances.get(asset, 0.0)
        if initial <= 0:
            return base

        series = series_for_asset(asset)
        try:
            acct = self.account_manager.get_account(series)
        except KeyError:
            return base

        current = acct.balance_dollars
        ratio = current / initial
        scaled = int(base * (ratio ** 0.67))
        scaled = max(1, min(scaled, self.MAX_CONTRACTS_CAP))
        return scaled

    def _get_max_price(self, asset: str) -> int:
        return self._max_price.get(asset, self._max_price.get("_default", self.DEFAULT_MAX_PRICE))

    def _get_min_price(self, asset: str) -> int:
        return self._min_price.get(asset, self._min_price.get("_default", self.DEFAULT_MIN_PRICE))

    def execute_trade(
        self,
        asset: str,
        direction: str,
        confidence: float,
        score: float,
        market_info: dict[str, Any],
    ) -> Optional[TradeRecord]:
        """Execute a trade based on fused signal output.

        Args:
            asset: e.g. "BTC"
            direction: "BULLISH" or "BEARISH"
            confidence: 0.0-1.0
            score: 0-100 consensus score
            market_info: dict from fetch_current_market()

        Returns:
            TradeRecord on success, None if trade was skipped.
        """
        side, price_cents = self._determine_side_and_price(direction, market_info)
        if price_cents is None:
            logger.warning("[kalshi-exec] No ask price for %s side=%s", asset, side)
            return None

        if not self._validate_price(price_cents, asset):
            return None

        series = series_for_asset(asset)
        account_name = series  # sub-account per series

        count = self._calculate_contracts(
            account_name, price_cents, confidence, score, asset=asset,
        )
        if count < 1:
            logger.info("[kalshi-exec] Insufficient funds or score for %s", asset)
            return None

        cost, fees = self._estimate_cost(price_cents, count)

        record = TradeRecord(
            trade_id=str(uuid.uuid4()),
            asset=asset,
            series=series,
            market_ticker=market_info.get("market_ticker", ""),
            event_ticker=market_info.get("event_ticker", ""),
            window_id=market_info.get("close_time", ""),
            side=side,
            action="buy",
            price_cents=price_cents,
            count=count,
            cost_dollars=cost,
            fees_dollars=fees,
            direction=direction,
            confidence=confidence,
            score=score,
            dry_run=self.is_dry_run(asset),
            placed_at=datetime.now(timezone.utc),
        )

        # Reserve funds
        try:
            self.account_manager.reserve(account_name, cost + fees)
        except ValueError as e:
            logger.warning("[kalshi-exec] Reserve failed for %s: %s", asset, e)
            return None

        if self.is_dry_run(asset):
            record.filled = count
            self.account_manager.record_fill(account_name, cost, fees)
            logger.info(
                "[kalshi-exec] DRY RUN %s: %s %s @ %dc x%d  cost=$%.2f",
                asset, "buy", side, price_cents, count, cost + fees,
            )
            return record

        # Live order
        try:
            result = place_limit_order(
                self.client,
                record.market_ticker,
                side,
                "buy",
                price_cents,
                count,
            )
            record.order_id = result.order_id
            record.filled = result.filled

            if result.filled > 0:
                actual_cost = result.cost_dollars or cost
                actual_fees = result.fees_dollars or fees
                self.account_manager.record_fill(
                    account_name, actual_cost, actual_fees,
                )
                logger.info(
                    "[kalshi-exec] FILLED %s: %s %s @ %dc x%d (filled=%d)",
                    asset, "buy", side, price_cents, count, result.filled,
                )
            else:
                # No fill — release reservation
                self.account_manager.release_reserve(account_name, cost + fees)
                logger.info(
                    "[kalshi-exec] NOT FILLED %s: %s — released reserve", asset, side,
                )
        except Exception:
            self.account_manager.release_reserve(account_name, cost + fees)
            logger.exception("[kalshi-exec] Order error for %s", asset)
            return None

        return record

    def settle_window(self, trade: TradeRecord) -> TradeRecord:
        """Check settlement outcome and credit/debit the sub-account.

        Works for both live and dry-run trades. Dry-run fetches the real
        Kalshi outcome (public data) and simulates the PnL on the sub-account.

        Mutates and returns the same TradeRecord.
        """
        if trade.settlement_outcome is not None:
            return trade  # already settled

        outcome = fetch_event_outcome(self.client, trade.event_ticker)
        if outcome is None:
            logger.debug(
                "[kalshi-exec] No outcome yet for %s", trade.event_ticker,
            )
            return trade

        trade.settlement_outcome = outcome
        trade.settled_at = datetime.now(timezone.utc)

        won = (trade.side == outcome)
        prefix = "DRY RUN " if trade.dry_run else ""
        if won:
            revenue = trade.filled * 1.00  # $1 per winning contract
            trade.revenue_dollars = revenue
            self.account_manager.record_settlement(trade.series, revenue)
            logger.info(
                "[kalshi-exec] %sWON %s: side=%s outcome=%s -> +$%.2f",
                prefix, trade.asset, trade.side, outcome, revenue,
            )
        else:
            trade.revenue_dollars = 0.0
            logger.info(
                "[kalshi-exec] %sLOST %s: side=%s outcome=%s -> -$%.2f",
                prefix, trade.asset, trade.side, outcome,
                trade.cost_dollars + trade.fees_dollars,
            )

        return trade

    # -- helpers ---------------------------------------------------------------

    def _determine_side_and_price(
        self, direction: str, market_info: dict[str, Any],
    ) -> tuple[str, Optional[int]]:
        """Map signal direction to Kalshi side and extract ask price."""
        direction_upper = direction.upper()
        if "BULLISH" in direction_upper:
            side = "yes"
            price = market_info.get("yes_ask")
        elif "BEARISH" in direction_upper:
            side = "no"
            price = market_info.get("no_ask")
        else:
            return "yes", None
        return side, price

    def _validate_price(self, price_cents: int, asset: str) -> bool:
        """Check that price is within safety bounds."""
        max_price = self._get_max_price(asset)
        min_price = self._get_min_price(asset)
        if price_cents > max_price:
            logger.warning(
                "[kalshi-exec] Price %dc > max %dc for %s — skipping",
                price_cents, max_price, asset,
            )
            return False
        if price_cents < min_price:
            logger.warning(
                "[kalshi-exec] Price %dc < min %dc for %s — skipping",
                price_cents, min_price, asset,
            )
            return False
        return True

    def _calculate_contracts(
        self,
        account_name: str,
        price_cents: int,
        confidence: float,
        score: float,
        asset: str = "",
    ) -> int:
        """Determine how many contracts to buy.

        Scales by confidence * (score/100), capped by per-asset
        max_contracts_per_trade and available sub-account balance.
        """
        try:
            acct = self.account_manager.get_account(account_name)
        except KeyError:
            return 0

        available = acct.balance_dollars - acct.reserved_dollars
        cost_per_contract = (price_cents / 100.0) + (KALSHI_FEE_CENTS / 100.0)

        if cost_per_contract <= 0 or available <= 0:
            return 0

        max_contracts = self._get_max_contracts(asset)
        max_by_balance = int(available / cost_per_contract)
        scale = min(1.0, confidence * (score / 100.0))
        desired = max(1, int(max_by_balance * scale))
        return min(desired, max_contracts)

    def _estimate_cost(
        self, price_cents: int, count: int,
    ) -> tuple[float, float]:
        """Estimate total cost and fees for a trade."""
        cost = (price_cents / 100.0) * count
        fees = (KALSHI_FEE_CENTS / 100.0) * count
        return round(cost, 4), round(fees, 4)
