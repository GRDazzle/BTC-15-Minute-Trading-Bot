"""Tests for sdk.kalshi.ticker module."""
import pytest
from sdk.kalshi.ticker import SERIES_MAP, series_for_asset, extract_window_id


def test_series_map_contains_all_assets():
    assert SERIES_MAP == {
        "BTC": "KXBTC15M",
        "ETH": "KXETH15M",
        "SOL": "KXSOL15M",
        "XRP": "KXXRP15M",
    }


def test_series_for_asset_valid():
    assert series_for_asset("BTC") == "KXBTC15M"
    assert series_for_asset("eth") == "KXETH15M"  # case-insensitive
    assert series_for_asset("Sol") == "KXSOL15M"


def test_series_for_asset_unknown():
    with pytest.raises(KeyError, match="Unknown asset"):
        series_for_asset("DOGE")


def test_extract_window_id():
    assert extract_window_id("KXBTC15M-26MAR092230") == "26MAR092230"
    assert extract_window_id("KXETH15M-26MAR18T1200") == "26MAR18T1200"


def test_extract_window_id_no_dash():
    assert extract_window_id("KXBTC15M") is None
    assert extract_window_id("") is None
    assert extract_window_id(None) is None
