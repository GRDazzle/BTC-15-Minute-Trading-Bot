"""Verify SDK package imports work correctly."""


def test_top_level_imports():
    from sdk.kalshi import KalshiClient, KalshiConfig, load_config
    assert KalshiClient is not None
    assert KalshiConfig is not None
    assert load_config is not None


def test_model_imports():
    from sdk.kalshi import OrderResult, OutcomeRecord, PollRecord
    assert OrderResult is not None
    assert OutcomeRecord is not None
    assert PollRecord is not None


def test_ticker_imports():
    from sdk.kalshi import SERIES_MAP, extract_window_id, series_for_asset
    assert "BTC" in SERIES_MAP
    assert series_for_asset("BTC") == "KXBTC15M"
    assert extract_window_id("KXBTC15M-26MAR092230") == "26MAR092230"


def test_submodule_imports():
    from sdk.kalshi.markets import fetch_current_market, fetch_event_outcome
    from sdk.kalshi.orders import place_limit_order, check_order, cancel_order
    from sdk.kalshi.orders import fetch_balance, fetch_settlements
    from sdk.kalshi.account import AccountManager, SubAccount
    assert fetch_current_market is not None
    assert AccountManager is not None
