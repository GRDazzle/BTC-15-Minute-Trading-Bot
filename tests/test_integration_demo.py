"""Integration tests against the Kalshi DEMO API.

These make real HTTP calls to the demo environment — no real money involved.

Run:  python -m pytest tests/test_integration_demo.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path

from sdk.kalshi import KalshiClient, KalshiConfig, load_config
from sdk.kalshi.markets import fetch_current_market, fetch_event_outcome
from sdk.kalshi.orders import (
    fetch_balance,
    fetch_settlements,
    place_limit_order,
    check_order,
    cancel_order,
)
from sdk.kalshi.ticker import SERIES_MAP

DEMO_ENV_PATH = Path(r"C:\Users\graso\clawd-buzz\secrets\kalshi_demo.env")
DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"


@pytest.fixture(scope="module")
def demo_client() -> KalshiClient:
    """Create a KalshiClient pointed at the demo API."""
    cfg = load_config(DEMO_ENV_PATH)
    # Override base_url to demo
    cfg = KalshiConfig(
        base_url=DEMO_BASE_URL,
        api_key_id=cfg.api_key_id,
        private_key_pem=cfg.private_key_pem,
    )
    return KalshiClient(cfg)


# ---------------------------------------------------------------------------
# Auth / Balance
# ---------------------------------------------------------------------------
class TestAuth:
    def test_get_balance_succeeds(self, demo_client: KalshiClient):
        st, data = demo_client.get_balance()
        assert st == 200, f"Expected 200, got {st}: {data}"
        assert "balance" in data

    def test_fetch_balance_returns_float(self, demo_client: KalshiClient):
        balance = fetch_balance(demo_client)
        assert balance is not None
        assert isinstance(balance, float)
        assert balance >= 0.0


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------
class TestMarkets:
    def test_get_markets_raw(self, demo_client: KalshiClient):
        st, data = demo_client.get_markets(limit=5)
        assert st == 200, f"Expected 200, got {st}: {data}"
        assert "markets" in data
        assert isinstance(data["markets"], list)

    def test_get_markets_with_series_filter(self, demo_client: KalshiClient):
        st, data = demo_client.get_markets(series_ticker="KXBTC15M", status="open", limit=5)
        assert st == 200, f"Expected 200, got {st}: {data}"
        markets = data.get("markets", [])
        # All returned markets should belong to the series
        for m in markets:
            assert "KXBTC15M" in (m.get("event_ticker") or ""), (
                f"Market {m.get('ticker')} doesn't match series filter"
            )

    def test_fetch_current_market_btc(self, demo_client: KalshiClient):
        market = fetch_current_market(demo_client, "KXBTC15M")
        # May be None outside trading hours on demo, but if returned it should be well-formed
        if market is not None:
            assert "market_ticker" in market
            assert "event_ticker" in market
            assert "close_time" in market
            assert "yes_ask" in market
            assert "no_ask" in market
            assert "mins_to_close" in market
            assert market["mins_to_close"] > 0

    def test_fetch_current_market_all_assets(self, demo_client: KalshiClient):
        """Try fetching current market for every supported asset."""
        for asset, series in SERIES_MAP.items():
            market = fetch_current_market(demo_client, series)
            # Just verify no exception — market may be None outside hours
            if market is not None:
                assert market["market_ticker"] is not None
                assert market["event_ticker"] is not None

    def test_fetch_event_outcome_nonexistent(self, demo_client: KalshiClient):
        """Outcome for a bogus ticker should return None, not crash."""
        result = fetch_event_outcome(demo_client, "KXFAKE-99DEC010101")
        assert result is None

    def test_fetch_event_outcome_recent(self, demo_client: KalshiClient):
        """Find a settled market and verify outcome fetch returns yes/no."""
        st, data = demo_client.get_markets(
            series_ticker="KXBTC15M", status="settled", limit=5,
        )
        if st != 200:
            pytest.skip("Could not fetch settled markets")
        markets = data.get("markets", []) or []
        if not markets:
            pytest.skip("No settled BTC markets found on demo")

        event_ticker = markets[0].get("event_ticker")
        outcome = fetch_event_outcome(demo_client, event_ticker)
        # Settled market should have an outcome
        assert outcome in ("yes", "no"), (
            f"Expected 'yes' or 'no' for settled event {event_ticker}, got {outcome}"
        )


# ---------------------------------------------------------------------------
# Settlements
# ---------------------------------------------------------------------------
class TestSettlements:
    def test_get_settlements_raw(self, demo_client: KalshiClient):
        st, data = demo_client.get_settlements(limit=5)
        assert st == 200, f"Expected 200, got {st}: {data}"
        # settlements key should exist even if empty
        assert "settlements" in data

    def test_fetch_settlements_returns_list(self, demo_client: KalshiClient):
        result = fetch_settlements(demo_client, limit=5)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Orders (place + cancel — uses demo money)
# ---------------------------------------------------------------------------
class TestOrders:
    def _find_any_open_market(self, demo_client: KalshiClient):
        """Find any open 15-min market across all assets."""
        for series in SERIES_MAP.values():
            market = fetch_current_market(demo_client, series)
            if market is not None and market.get("market_ticker"):
                return market
        return None

    def test_place_and_cancel_order(self, demo_client: KalshiClient):
        """Place a 1c limit order (way out of range), verify it rests, then cancel."""
        market = self._find_any_open_market(demo_client)
        if market is None:
            pytest.skip("No open 15-min market on demo right now")

        market_ticker = market["market_ticker"]
        balance_before = fetch_balance(demo_client)

        # Place a yes buy at 1c — way below any real ask, guaranteed to rest
        result = place_limit_order(
            demo_client,
            market_ticker=market_ticker,
            side="yes",
            action="buy",
            price_cents=1,
            count=1,
        )
        assert result.order_id is not None, f"Expected order_id, got status={result.status}"
        assert result.status == "resting", f"Expected resting at 1c, got {result.status}"
        assert result.filled == 0, "Should not have filled at 1c"

        # Cancel the resting order
        cancel_result = cancel_order(demo_client, result.order_id)
        assert cancel_result.order_id == result.order_id
        assert cancel_result.status == "canceled"
        assert cancel_result.filled == 0

        # Balance should be unchanged (no fill, no cost)
        balance_after = fetch_balance(demo_client)
        assert balance_after == balance_before, (
            f"Balance changed: ${balance_before} -> ${balance_after}"
        )

    def test_place_and_cancel_no_side(self, demo_client: KalshiClient):
        """Same test but on the 'no' side at 1c."""
        market = self._find_any_open_market(demo_client)
        if market is None:
            pytest.skip("No open 15-min market on demo right now")

        result = place_limit_order(
            demo_client,
            market_ticker=market["market_ticker"],
            side="no",
            action="buy",
            price_cents=1,
            count=1,
        )
        assert result.order_id is not None
        assert result.status == "resting"

        cancel_result = cancel_order(demo_client, result.order_id)
        assert cancel_result.status == "canceled"

    def test_check_order_nonexistent(self, demo_client: KalshiClient):
        """Checking a bogus order_id should return gracefully."""
        result = check_order(demo_client, "nonexistent-order-id-12345")
        assert result.status == "unknown"
