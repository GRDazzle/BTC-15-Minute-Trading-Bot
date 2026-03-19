"""Tests for trading config loading (main.load_trading_config)."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from main import load_trading_config, get_asset_config


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    return tmp_path


def _write_config(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestLoadTradingConfig:
    def test_loads_defaults_and_assets(self, config_dir):
        cfg_path = _write_config(config_dir / "trading.json", {
            "defaults": {
                "initial_balance": 25.0,
                "max_contracts_per_trade": 10,
                "max_price_cents": 85,
                "min_price_cents": 15,
            },
            "assets": {
                "BTC": {"initial_balance": 50.0, "max_contracts_per_trade": 15},
                "SOL": {"max_contracts_per_trade": 8},
            },
        })
        config = load_trading_config(cfg_path)

        # BTC has overridden values
        btc = get_asset_config(config, "BTC")
        assert btc["initial_balance"] == 50.0
        assert btc["max_contracts_per_trade"] == 15
        # BTC inherits defaults for keys not overridden
        assert btc["max_price_cents"] == 85
        assert btc["min_price_cents"] == 15

    def test_asset_inherits_defaults(self, config_dir):
        cfg_path = _write_config(config_dir / "trading.json", {
            "defaults": {
                "initial_balance": 25.0,
                "max_contracts_per_trade": 10,
                "max_price_cents": 85,
                "min_price_cents": 15,
            },
            "assets": {
                "SOL": {"max_contracts_per_trade": 8},
            },
        })
        config = load_trading_config(cfg_path)
        sol = get_asset_config(config, "SOL")
        assert sol["initial_balance"] == 25.0  # from defaults
        assert sol["max_contracts_per_trade"] == 8  # overridden

    def test_unlisted_asset_gets_defaults(self, config_dir):
        cfg_path = _write_config(config_dir / "trading.json", {
            "defaults": {
                "initial_balance": 25.0,
                "max_contracts_per_trade": 10,
                "max_price_cents": 85,
                "min_price_cents": 15,
            },
            "assets": {"BTC": {"initial_balance": 50.0}},
        })
        config = load_trading_config(cfg_path)
        # DOGE is not listed — should get full defaults
        doge = get_asset_config(config, "DOGE")
        assert doge["initial_balance"] == 25.0
        assert doge["max_contracts_per_trade"] == 10

    def test_missing_config_raises_error(self):
        with pytest.raises(FileNotFoundError, match="Trading config not found"):
            load_trading_config("/nonexistent/path/trading.json")

    def test_empty_assets_section(self, config_dir):
        cfg_path = _write_config(config_dir / "trading.json", {
            "defaults": {
                "initial_balance": 30.0,
                "max_contracts_per_trade": 12,
                "max_price_cents": 90,
                "min_price_cents": 10,
            },
        })
        config = load_trading_config(cfg_path)
        btc = get_asset_config(config, "BTC")
        assert btc["initial_balance"] == 30.0
        assert btc["max_contracts_per_trade"] == 12

    def test_case_insensitive_asset_lookup(self, config_dir):
        cfg_path = _write_config(config_dir / "trading.json", {
            "defaults": {"initial_balance": 25.0, "max_contracts_per_trade": 10,
                         "max_price_cents": 85, "min_price_cents": 15},
            "assets": {"btc": {"initial_balance": 99.0}},
        })
        config = load_trading_config(cfg_path)
        btc = get_asset_config(config, "BTC")
        assert btc["initial_balance"] == 99.0
