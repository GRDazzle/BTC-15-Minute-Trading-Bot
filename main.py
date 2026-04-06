"""Kalshi 15-minute multi-asset trading bot — main entry point.

Usage:
    python main.py --assets BTC,ETH,SOL,XRP
    python main.py --assets BTC --env demo --log-level DEBUG
    REAL_TRADE=TRUE python main.py --assets BTC   # live trading

DRY_RUN is True by default. Set env var REAL_TRADE=TRUE to go live.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

from loguru import logger

from sdk.kalshi.client import KalshiClient, KalshiConfig, load_config
from sdk.kalshi.account import AccountManager
from sdk.kalshi.orders import fetch_balance
from sdk.kalshi.ticker import SERIES_MAP, series_for_asset
from execution.kalshi_execution import KalshiExecutionAdapter
from strategies.kalshi_strategy import KalshiMultiAssetStrategy

DEFAULT_ENV_PATH = Path(r"C:\Users\graso\clawd-buzz\secrets\kalshi.env")
DEMO_ENV_PATH = Path(r"C:\Users\graso\clawd-buzz\secrets\kalshi_demo.env")
DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"
STATE_PATH = Path("data/account_state.json")
DEFAULT_CONFIG_PATH = Path("config/trading.json")

# Fallback defaults when no config file exists
_BUILTIN_DEFAULTS = {
    "initial_balance": 25.0,
    "max_contracts_per_trade": 10,
    "max_price_cents": 85,
    "min_price_cents": 15,
}


def load_trading_config(path: str | Path) -> dict[str, dict]:
    """Load per-asset trading config from JSON.

    Returns a dict keyed by asset (e.g. "BTC") where each value is a dict
    with keys: initial_balance, max_contracts_per_trade, max_price_cents,
    min_price_cents.  Assets not listed in the file get full defaults.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Trading config not found: {config_path}")

    with open(config_path, "r") as f:
        raw = json.load(f)

    defaults = {**_BUILTIN_DEFAULTS, **raw.get("defaults", {})}
    asset_overrides = raw.get("assets", {})

    result: dict[str, dict] = {}
    # Merge per-asset overrides with defaults
    for asset, overrides in asset_overrides.items():
        result[asset.upper()] = {**defaults, **overrides}

    # Store defaults so callers can look up unlisted assets
    result["_defaults"] = defaults
    return result


def get_asset_config(trading_config: dict[str, dict], asset: str) -> dict:
    """Get config for a specific asset, falling back to defaults."""
    return trading_config.get(asset.upper(), trading_config["_defaults"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kalshi 15-min multi-asset trading bot",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default="BTC",
        help="Comma-separated assets: BTC,ETH,SOL,XRP",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["prod", "demo"],
        default="prod",
        help="API environment (prod or demo)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=str(STATE_PATH),
        help="Path to account state JSON file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to trading config JSON file",
    )
    parser.add_argument(
        "--tune-delay",
        type=int,
        default=7200,
        help="Seconds before first param tune run (default: 7200 = 2h). "
             "Set to 0 to tune immediately on startup.",
    )
    return parser.parse_args()


def determine_dry_run() -> bool:
    """Check REAL_TRADE env var. Default is dry run (True)."""
    val = os.environ.get("REAL_TRADE", "").strip().upper()
    if val == "TRUE":
        return False
    return True


def configure_logging(level: str):
    """Set up logging with loguru and stdlib bridge."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    # Log to file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        "logs/trading.log",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
    )
    # Bridge stdlib logging → loguru
    logging.basicConfig(handlers=[_LoguruHandler()], level=0, force=True)


class _LoguruHandler(logging.Handler):
    """Bridge stdlib log records to loguru."""
    def emit(self, record: logging.LogRecord):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def initialize_sub_accounts(
    account_manager: AccountManager,
    assets: list[str],
    trading_config: dict[str, dict],
):
    """Create or load sub-accounts. Existing accounts are preserved."""
    for asset in assets:
        series = series_for_asset(asset)
        asset_cfg = get_asset_config(trading_config, asset)
        initial_balance = asset_cfg["initial_balance"]
        try:
            acct = account_manager.get_account(series)
            logger.info(
                "Existing sub-account %s: balance=$%.2f pnl=$%.2f",
                series, acct.balance_dollars, acct.pnl_dollars,
            )
        except KeyError:
            acct = account_manager.create_account(series, initial_balance)
            logger.info(
                "Created sub-account %s with $%.2f",
                series, initial_balance,
            )


async def main():
    args = parse_args()
    configure_logging(args.log_level)

    dry_run = determine_dry_run()
    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]

    # Load per-asset trading config
    trading_config = load_trading_config(args.config)

    logger.info("=" * 60)
    logger.info("Kalshi 15-Min Multi-Asset Trading Bot")
    logger.info("=" * 60)
    logger.info("Assets:   %s", ", ".join(assets))
    logger.info("Env:      %s", args.env)
    logger.info("Dry Run:  %s", dry_run)
    logger.info("Config:   %s", args.config)
    logger.info("State:    %s", args.state_file)

    # Validate assets
    for asset in assets:
        if asset not in SERIES_MAP:
            logger.error("Unsupported asset: %s (valid: %s)", asset, list(SERIES_MAP.keys()))
            sys.exit(1)

    # Load config (demo uses separate credentials)
    if args.env == "demo":
        cfg = load_config(DEMO_ENV_PATH)
        cfg = KalshiConfig(
            base_url=DEMO_BASE_URL,
            api_key_id=cfg.api_key_id,
            private_key_pem=cfg.private_key_pem,
        )
    else:
        cfg = load_config()

    # Create client
    client = KalshiClient(cfg)

    # Verify connectivity
    logger.info("Verifying Kalshi connectivity...")
    balance = fetch_balance(client)
    if balance is not None:
        logger.info("Kalshi balance: $%.2f", balance)
    else:
        logger.warning("Could not fetch Kalshi balance — continuing anyway")

    # Initialize AccountManager and sub-accounts
    account_manager = AccountManager(state_path=Path(args.state_file))
    initialize_sub_accounts(account_manager, assets, trading_config)

    # Build per-asset config dicts for the execution adapter
    per_asset_max_contracts = {
        a: get_asset_config(trading_config, a)["max_contracts_per_trade"]
        for a in assets
    }
    per_asset_max_price = {
        a: get_asset_config(trading_config, a)["max_price_cents"]
        for a in assets
    }
    per_asset_min_price = {
        a: get_asset_config(trading_config, a)["min_price_cents"]
        for a in assets
    }
    per_asset_initial_balance = {
        a: get_asset_config(trading_config, a)["initial_balance"]
        for a in assets
    }

    # Create execution adapter
    execution_adapter = KalshiExecutionAdapter(
        client=client,
        account_manager=account_manager,
        dry_run=dry_run,
        max_contracts_per_trade=per_asset_max_contracts,
        max_price_cents=per_asset_max_price,
        min_price_cents=per_asset_min_price,
        initial_balances=per_asset_initial_balance,
    )

    # Set per-asset dry_run from config (overrides global --real flag)
    for a in assets:
        asset_cfg = get_asset_config(trading_config, a)
        if "dry_run" in asset_cfg:
            execution_adapter.set_dry_run(a, asset_cfg["dry_run"])
            if not asset_cfg["dry_run"]:
                logger.info("LIVE TRADING enabled for %s", a)

    # Create strategy
    strategy = KalshiMultiAssetStrategy(
        client=client,
        account_manager=account_manager,
        execution_adapter=execution_adapter,
        assets=assets,
        dry_run=dry_run,
        tune_delay_seconds=args.tune_delay,
    )

    # Graceful shutdown handler
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for SIGTERM
            pass

    # Start strategy
    await strategy.start()

    logger.info("Bot running. Press Ctrl+C to stop.")

    # Wait for shutdown
    try:
        await shutdown_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    # Stop gracefully
    await strategy.stop()
    account_manager.save()
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
