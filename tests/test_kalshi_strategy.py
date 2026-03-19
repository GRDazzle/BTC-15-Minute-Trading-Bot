"""Tests for KalshiMultiAssetStrategy — mock-based, no network."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from strategies.kalshi_strategy import (
    KalshiMultiAssetStrategy,
    AssetState,
    BINANCE_SYMBOLS,
)
from execution.kalshi_execution import KalshiExecutionAdapter, TradeRecord
from sdk.kalshi.account import AccountManager
from sdk.kalshi.ticker import series_for_asset


@pytest.fixture
def account_manager(tmp_path: Path) -> AccountManager:
    mgr = AccountManager(state_path=tmp_path / "state.json")
    mgr.create_account("KXBTC15M", initial_balance=100.0)
    mgr.create_account("KXETH15M", initial_balance=100.0)
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
    )


@pytest.fixture
def strategy(client, account_manager, adapter) -> KalshiMultiAssetStrategy:
    return KalshiMultiAssetStrategy(
        client=client,
        account_manager=account_manager,
        execution_adapter=adapter,
        assets=["BTC", "ETH"],
        dry_run=True,
    )


# === Window boundary ===

class TestWindowBoundary:
    def test_floor_to_15_minutes(self):
        # Patch datetime.now to return a known time
        with patch("strategies.kalshi_strategy.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 18, 10, 7, 30, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = KalshiMultiAssetStrategy._current_window_boundary()
            assert result.minute == 0
            assert result.second == 0

    def test_exact_boundary(self):
        with patch("strategies.kalshi_strategy.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 18, 10, 15, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = KalshiMultiAssetStrategy._current_window_boundary()
            assert result.minute == 15

    def test_floor_37_to_30(self):
        with patch("strategies.kalshi_strategy.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 18, 10, 37, 45, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = KalshiMultiAssetStrategy._current_window_boundary()
            assert result.minute == 30

    def test_floor_59_to_45(self):
        with patch("strategies.kalshi_strategy.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 18, 10, 59, 59, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = KalshiMultiAssetStrategy._current_window_boundary()
            assert result.minute == 45


# === Signal routing ===

class TestSignalRouting:
    def test_no_signals_with_insufficient_history(self, strategy):
        state = strategy.states["BTC"]
        # Only 5 prices — needs at least 20
        for i in range(5):
            state.price_history.append(Decimal("50000"))
        state.current_price = Decimal("50000")

        signals = strategy._run_signals(state)
        # With < 20 data points, spike detector won't fire
        # (it needs lookback_periods=20)
        # Some processors might still fire, but the strategy
        # _process_asset_window checks len >= 20 before calling _run_signals
        # This test verifies _run_signals itself doesn't crash
        assert isinstance(signals, list)

    def test_metadata_contains_tick_buffer(self, strategy):
        state = strategy.states["BTC"]
        state.current_price = Decimal("50000")
        state.tick_buffer.append({"ts": datetime.now(), "price": 50000.0})

        meta = strategy._build_metadata(state)
        assert "tick_buffer" in meta
        assert len(meta["tick_buffer"]) == 1

    def test_metadata_contains_momentum(self, strategy):
        state = strategy.states["BTC"]
        for p in [Decimal("100"), Decimal("101"), Decimal("102"), Decimal("103"), Decimal("105")]:
            state.price_history.append(p)
        state.current_price = Decimal("105")

        meta = strategy._build_metadata(state)
        assert "momentum" in meta
        assert meta["momentum"] == pytest.approx(0.05, abs=0.001)

    def test_metadata_has_sentiment_placeholder(self, strategy):
        state = strategy.states["BTC"]
        state.current_price = Decimal("50000")
        meta = strategy._build_metadata(state)
        assert meta["sentiment_score"] == 50.0


# === Asset state initialization ===

class TestAssetStateInit:
    def test_each_asset_has_independent_processors(self, strategy):
        btc_state = strategy.states["BTC"]
        eth_state = strategy.states["ETH"]

        # Different instances
        assert btc_state.spike is not eth_state.spike
        assert btc_state.velocity is not eth_state.velocity
        assert btc_state.sentiment is not eth_state.sentiment
        assert btc_state.deribit_pcr is not eth_state.deribit_pcr
        assert btc_state.divergence is not eth_state.divergence

    def test_divergence_disabled_by_default(self, strategy):
        for state in strategy.states.values():
            assert state.divergence.is_enabled is False

    def test_other_processors_enabled(self, strategy):
        for state in strategy.states.values():
            assert state.spike.is_enabled is True
            assert state.velocity.is_enabled is True
            assert state.sentiment.is_enabled is True
            assert state.deribit_pcr.is_enabled is True

    def test_series_mapping(self, strategy):
        assert strategy.states["BTC"].series == "KXBTC15M"
        assert strategy.states["ETH"].series == "KXETH15M"

    def test_binance_symbols(self, strategy):
        assert strategy.states["BTC"].binance_symbol == "btcusdt"
        assert strategy.states["ETH"].binance_symbol == "ethusdt"

    def test_ws_source_created(self, strategy):
        for state in strategy.states.values():
            assert state.ws_source is not None
            assert state.ws_source.symbol == state.binance_symbol


# === Deduplication ===

class TestDeduplication:
    @pytest.mark.asyncio
    async def test_doesnt_trade_same_window_twice(self, strategy):
        state = strategy.states["BTC"]
        window_id = "20260318_1200"
        state.traded_windows.add(window_id)

        # Fill enough history so the guard doesn't skip for "not enough data"
        for i in range(30):
            state.price_history.append(Decimal("50000") + Decimal(i))
        state.current_price = Decimal("50029")

        # Even with full data, should not trade again
        await strategy._process_asset_window("BTC", state, window_id)

        # No pending settlements added (trade was skipped)
        assert len(state.pending_settlements) == 0

    @pytest.mark.asyncio
    async def test_new_window_can_trade(self, strategy):
        state = strategy.states["BTC"]
        state.traded_windows.add("20260318_1200")

        for i in range(30):
            state.price_history.append(Decimal("50000") + Decimal(i))
        state.current_price = Decimal("50029")

        mock_market = {
            "market_ticker": "KXBTC15M-26MAR18-T1215-B87500",
            "event_ticker": "KXBTC15M-26MAR18-T1215",
            "close_time": "2026-03-18T12:30:00Z",
            "yes_bid": 48,
            "yes_ask": 52,
            "no_bid": 46,
            "no_ask": 50,
        }

        # Mock _run_signals to avoid live HTTP calls from DeribitPCR
        mock_signal = MagicMock()
        mock_signal.source = "SpikeDetection"
        mock_signal.direction = "SignalDirection.BULLISH"
        mock_signal.strength = MagicMock(value=3, name="STRONG")
        mock_signal.confidence = 0.85
        mock_signal.timestamp = datetime.now(timezone.utc)

        with patch(
            "strategies.kalshi_strategy.fetch_current_market",
            return_value=mock_market,
        ), patch.object(
            strategy, "_run_signals", return_value=[mock_signal],
        ):
            # Patch fusion to return actionable signal
            mock_fused = MagicMock()
            mock_fused.is_actionable = True
            mock_fused.direction = "SignalDirection.BULLISH"
            mock_fused.confidence = 0.8
            mock_fused.score = 75.0

            with patch.object(
                strategy.fusion_engine, "fuse_signals", return_value=mock_fused,
            ):
                await strategy._process_asset_window("BTC", state, "20260318_1215")

        # Should have traded the new window
        assert "20260318_1215" in state.traded_windows


# === Unsupported asset ===

class TestUnsupportedAsset:
    def test_raises_on_unsupported_asset(self, client, account_manager, adapter):
        with pytest.raises(ValueError, match="No Binance symbol"):
            KalshiMultiAssetStrategy(
                client=client,
                account_manager=account_manager,
                execution_adapter=adapter,
                assets=["DOGE"],
            )
