"""Promote one or more dry-run assets to LIVE trading.

This is the cutover script for Option B.2:

  1. Snapshot trades.csv, balance.csv, account_state.json to date-tagged
     archive files (full safety net before any modifications).
  2. Strip the specified assets' rows from trades.csv and balance.csv
     in place. Their dry-run history lives ONLY in the archive going forward.
  3. Create empty trades_live.csv and balance_live.csv with the live schemas
     (only created if they don't already exist).
  4. Reset the specified assets' sub-account entries in account_state.json
     to fresh balances ($50 default), pnl=$0, reserved=$0.
  5. Edit config/trading.json to set "dry_run": false and
     "initial_balance": <balance> for the specified assets.

After running, you MUST:
  - Stop the bot if it's running. This script does NOT stop the bot.
  - Verify your Kalshi account has at least N x <balance> in actual cash.
  - Restart the bot. The new live state takes effect on startup.

Usage:
    python scripts/go_live.py SOL XRP
    python scripts/go_live.py SOL XRP --balance 50
    python scripts/go_live.py SOL --balance 100
    python scripts/go_live.py SOL XRP --dry-run     # show what would happen
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRADES_CSV = PROJECT_ROOT / "output" / "trades.csv"
TRADES_LIVE_CSV = PROJECT_ROOT / "output" / "trades_live.csv"
BALANCE_CSV = PROJECT_ROOT / "output" / "balance.csv"
BALANCE_LIVE_CSV = PROJECT_ROOT / "output" / "balance_live.csv"
ACCOUNT_STATE = PROJECT_ROOT / "data" / "account_state.json"
CONFIG_JSON = PROJECT_ROOT / "config" / "trading.json"

ASSET_TO_SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
    "HYPE": "KXHYPE15M",
    "BNB": "KXBNB15M",
    "DOGE": "KXDOGE15M",
}

# Live schema -- must match strategies.kalshi_strategy.TRADE_LOG_LIVE_FIELDS
TRADES_LIVE_HEADER = [
    "timestamp", "asset", "window_id", "market_ticker", "event_ticker",
    "direction", "side",
    "intended_price_cents", "fill_price_cents",
    "intended_count", "filled_count",
    "cost", "fees",
    "kalshi_order_id",
    "dm", "mtc", "confidence", "score",
    "outcome",
    "expected_revenue", "verified_revenue",
    "settlement_verified", "verified_at",
    "pnl", "balance_after",
]

BALANCE_HEADER = ["timestamp", "event", "asset", "balance", "pnl", "kalshi_balance"]


def info(msg: str) -> None:
    print(f"  {msg}")


def warn(msg: str) -> None:
    print(f"  ! {msg}")


def err(msg: str) -> None:
    print(f"  X {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote dry-run assets to live trading")
    parser.add_argument("assets", nargs="+", help="Assets to go live (e.g. SOL XRP)")
    parser.add_argument(
        "--balance",
        type=float,
        default=50.0,
        help="Starting balance per asset in dollars (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without modifying any files",
    )
    return parser.parse_args()


def validate_assets(assets: list[str]) -> list[str]:
    out = []
    for a in assets:
        upper = a.upper()
        if upper not in ASSET_TO_SERIES:
            err(f"Unknown asset: {a}. Valid: {','.join(ASSET_TO_SERIES.keys())}")
            sys.exit(1)
        out.append(upper)
    return out


def snapshot_files(date_tag: str, dry_run: bool) -> None:
    """Copy state files to date-tagged archives."""
    pairs = []
    if TRADES_CSV.exists():
        pairs.append((TRADES_CSV, TRADES_CSV.with_name(f"trades_archive_{date_tag}.csv")))
    if BALANCE_CSV.exists():
        pairs.append((BALANCE_CSV, BALANCE_CSV.with_name(f"balance_archive_{date_tag}.csv")))
    if ACCOUNT_STATE.exists():
        pairs.append((ACCOUNT_STATE, ACCOUNT_STATE.with_name(f"account_state_archive_{date_tag}.json")))

    info(f"Snapshotting {len(pairs)} state file(s) to date-tagged archives:")
    for src, dst in pairs:
        if dst.exists():
            warn(f"{dst.name} already exists -- skipping (won't overwrite existing archive)")
            continue
        info(f"  {src.name} -> {dst.name}")
        if not dry_run:
            shutil.copy2(src, dst)


def strip_assets_from_csv(
    path: Path,
    asset_col: str,
    assets_to_strip: set[str],
    label: str,
    dry_run: bool,
) -> None:
    """Remove rows where the `asset` column matches any in `assets_to_strip`."""
    if not path.exists():
        info(f"{label}: file does not exist, nothing to strip")
        return

    kept = 0
    stripped = 0
    rows: list[dict] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if (row.get(asset_col) or "").strip().upper() in assets_to_strip:
                stripped += 1
            else:
                kept += 1
                rows.append(row)

    info(f"{label}: {stripped} rows stripped, {kept} rows kept")

    if dry_run:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_empty_csv(path: Path, header: list[str], label: str, dry_run: bool) -> None:
    if path.exists():
        info(f"{label}: already exists -- not overwriting")
        return
    info(f"{label}: creating with header ({len(header)} cols)")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def reset_account_state(
    assets: list[str], balance: float, dry_run: bool
) -> None:
    """Reset specified assets to fresh balance/pnl/reserved in account_state.json."""
    if not ACCOUNT_STATE.exists():
        warn(f"{ACCOUNT_STATE.name} does not exist -- nothing to reset")
        return

    with open(ACCOUNT_STATE, "r") as f:
        state = json.load(f)

    info(f"Resetting {len(assets)} sub-account(s) to ${balance:.2f}:")
    for asset in assets:
        series = ASSET_TO_SERIES[asset]
        old = state.get(series, {}).get("balance_dollars", 0.0)
        info(f"  {asset} ({series}): ${old:.2f} -> ${balance:.2f}, pnl=$0.00, reserved=$0.00")
        if not dry_run:
            state[series] = {
                "name": series,
                "balance_dollars": float(balance),
                "reserved_dollars": 0.0,
                "pnl_dollars": 0.0,
            }

    if not dry_run:
        with open(ACCOUNT_STATE, "w") as f:
            json.dump(state, f, indent=2)


def update_config(assets: list[str], balance: float, dry_run: bool) -> None:
    """Set dry_run=false and initial_balance=<balance> for the assets in config."""
    if not CONFIG_JSON.exists():
        err(f"{CONFIG_JSON} not found -- cannot update config")
        sys.exit(1)

    with open(CONFIG_JSON, "r") as f:
        cfg = json.load(f)

    info("Updating config/trading.json:")
    for asset in assets:
        asset_cfg = cfg.setdefault("assets", {}).setdefault(asset, {})
        old_dry = asset_cfg.get("dry_run", "(unset)")
        old_bal = asset_cfg.get("initial_balance", "(unset)")
        info(f"  {asset}: dry_run={old_dry} -> false, initial_balance={old_bal} -> {balance}")
        if not dry_run:
            asset_cfg["dry_run"] = False
            asset_cfg["initial_balance"] = float(balance)

    if not dry_run:
        with open(CONFIG_JSON, "w") as f:
            json.dump(cfg, f, indent=2)


def main() -> int:
    args = parse_args()
    assets = validate_assets(args.assets)
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total_balance = args.balance * len(assets)

    print()
    print("=" * 72)
    print("GO LIVE")
    print("=" * 72)
    info(f"Assets to promote: {', '.join(assets)}")
    info(f"Starting balance per asset: ${args.balance:.2f}")
    info(f"Total starting balance: ${total_balance:.2f}")
    info(f"Date tag: {date_tag}")
    if args.dry_run:
        warn("DRY RUN MODE: no files will be modified")
    print()

    if not args.dry_run:
        warn("This script does NOT stop the bot. Stop the bot first!")
        warn("If the bot is running, it will overwrite your changes.")
        try:
            answer = input("  Stop the bot, then type 'yes' to continue: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            return 1
        if answer != "yes":
            print("  Aborted.")
            return 1
        print()

    # 1. Snapshot
    print("[1/5] Snapshotting current state...")
    snapshot_files(date_tag, args.dry_run)
    print()

    # 2. Strip live assets out of dry CSVs
    print("[2/5] Stripping live assets from dry CSVs...")
    asset_set = set(assets)
    strip_assets_from_csv(TRADES_CSV, "asset", asset_set, "trades.csv", args.dry_run)
    strip_assets_from_csv(BALANCE_CSV, "asset", asset_set, "balance.csv", args.dry_run)
    print()

    # 3. Create empty live CSVs
    print("[3/5] Creating live CSVs (if missing)...")
    create_empty_csv(TRADES_LIVE_CSV, TRADES_LIVE_HEADER, "trades_live.csv", args.dry_run)
    create_empty_csv(BALANCE_LIVE_CSV, BALANCE_HEADER, "balance_live.csv", args.dry_run)
    print()

    # 4. Reset account state
    print("[4/5] Resetting account_state.json for live assets...")
    reset_account_state(assets, args.balance, args.dry_run)
    print()

    # 5. Update config
    print("[5/5] Updating config/trading.json...")
    update_config(assets, args.balance, args.dry_run)
    print()

    print("=" * 72)
    if args.dry_run:
        print("DRY RUN COMPLETE -- no files modified")
    else:
        print("CUTOVER COMPLETE")
    print("=" * 72)
    print()
    print("Next steps:")
    print(f"  1. Verify your Kalshi account has at least ${total_balance:.2f} in cash")
    print(f"  2. Restart the bot (it should pick up the new config on startup)")
    print(f"  3. On startup, the bot will run startup_reconcile() and log:")
    print(f"     'LIVE TRADING enabled for SOL/XRP'")
    print(f"  4. Check the manager UI -- you should see:")
    print(f"     - LIVE Trading table (SOL, XRP at ${args.balance:.2f} each)")
    print(f"     - Dry Run table (other 5 assets unchanged)")
    print(f"     - Reconciliation panel showing the new LIVE row")
    print()
    print("To roll back, run:  python scripts/go_dry.py SOL XRP")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
