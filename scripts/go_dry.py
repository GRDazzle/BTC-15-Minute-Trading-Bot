"""Roll one or more LIVE assets back to dry-run mode.

This is the rollback script. It does NOT touch your money in Kalshi --
it only changes how the bot accounts for things going forward.

Default behavior (--keep-balance):
  The asset's current live balance becomes its new dry-run starting
  balance. Useful when you want to take a break from live trading but
  preserve what the live experiment earned/lost.

Alternative (--reset):
  The asset is reset to a fresh dry-run balance ($25 by default).
  Useful for a clean slate.

What this script does:
  1. CHECK: refuse to run if there are pending verifications for any
     of the assets (wait for in-flight live trades to settle and
     verify before rolling back). Use --force to override.
  2. Archive trades_live.csv and balance_live.csv to date-tagged files.
  3. Update account_state.json for the rolled-back assets:
       --keep-balance: keep current balance, reset reserved to 0
       --reset:        balance = --reset-balance (default $25), pnl=$0
  4. Edit config/trading.json: set dry_run=true (or remove the override)
     for the specified assets, restore initial_balance to default.

After running, you MUST:
  - Stop the bot if it's running. This script does NOT stop the bot.
  - Restart the bot. The dry-run state takes effect on startup.

Important: this script does NOT withdraw your money from Kalshi.
The cash stays in your Kalshi account; you have to withdraw it
manually through Kalshi's interface if you want it back in your bank.

Usage:
    python scripts/go_dry.py SOL XRP                       # keep balance
    python scripts/go_dry.py SOL --reset                   # reset to $25
    python scripts/go_dry.py SOL --reset --reset-balance 25
    python scripts/go_dry.py SOL XRP --dry-run             # show what would happen
    python scripts/go_dry.py SOL XRP --force               # bypass pending check
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

TRADES_LIVE_CSV = PROJECT_ROOT / "output" / "trades_live.csv"
BALANCE_LIVE_CSV = PROJECT_ROOT / "output" / "balance_live.csv"
ACCOUNT_STATE = PROJECT_ROOT / "data" / "account_state.json"
CONFIG_JSON = PROJECT_ROOT / "config" / "trading.json"
RUNTIME_STATE = PROJECT_ROOT / "data" / "runtime_state.json"

ASSET_TO_SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
    "HYPE": "KXHYPE15M",
    "BNB": "KXBNB15M",
    "DOGE": "KXDOGE15M",
}


def info(msg: str) -> None:
    print(f"  {msg}")


def warn(msg: str) -> None:
    print(f"  ! {msg}")


def err(msg: str) -> None:
    print(f"  X {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll live assets back to dry-run")
    parser.add_argument("assets", nargs="+", help="Assets to roll back (e.g. SOL XRP)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--keep-balance",
        action="store_true",
        default=True,
        help="Keep current live balance as new dry-run starting balance (default)",
    )
    mode.add_argument(
        "--reset",
        action="store_true",
        help="Reset to a fresh dry-run balance instead of keeping current",
    )
    parser.add_argument(
        "--reset-balance",
        type=float,
        default=25.0,
        help="Fresh balance for --reset mode (default: 25)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the pending-verification check (NOT RECOMMENDED)",
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


def check_pending_verifications(assets: list[str], force: bool) -> bool:
    """Refuse to roll back if pending verifications exist."""
    if not RUNTIME_STATE.exists():
        info("No runtime_state.json found -- bot may not be running")
        return True
    try:
        with open(RUNTIME_STATE, "r") as f:
            state = json.load(f)
    except Exception as e:
        warn(f"Could not read runtime_state.json: {e}")
        return True

    pending = state.get("verification", {}).get("pending_count", 0)
    stale = state.get("verification", {}).get("stale_count", 0)
    if pending == 0 and stale == 0:
        info("No pending or stale verifications -- safe to roll back")
        return True

    err(
        f"Pending verifications: {pending}, stale: {stale}. "
        f"In-flight live trades exist."
    )
    if force:
        warn("--force set: bypassing pending check (NOT RECOMMENDED)")
        return True
    err(
        "Wait for live trades to settle and verify, then re-run.\n"
        "  Or use --force to abandon them (you may end up with state drift).\n"
        "  Or stop the bot first, then in-flight trades will not progress."
    )
    return False


def check_assets_are_live(assets: list[str]) -> list[str]:
    """Filter to only assets that are currently live in config."""
    if not CONFIG_JSON.exists():
        return []
    with open(CONFIG_JSON, "r") as f:
        cfg = json.load(f)
    live: list[str] = []
    for asset in assets:
        cfg_asset = cfg.get("assets", {}).get(asset, {})
        if cfg_asset.get("dry_run") is False:
            live.append(asset)
        else:
            warn(f"{asset} is not currently in live mode -- skipping")
    return live


def archive_live_files(date_tag: str, dry_run: bool) -> None:
    """Move live CSVs to date-tagged archive files."""
    pairs = []
    if TRADES_LIVE_CSV.exists():
        pairs.append((TRADES_LIVE_CSV, TRADES_LIVE_CSV.with_name(f"trades_live_archive_{date_tag}.csv")))
    if BALANCE_LIVE_CSV.exists():
        pairs.append((BALANCE_LIVE_CSV, BALANCE_LIVE_CSV.with_name(f"balance_live_archive_{date_tag}.csv")))

    if not pairs:
        info("No live CSVs to archive")
        return

    info(f"Archiving {len(pairs)} live CSV(s):")
    for src, dst in pairs:
        if dst.exists():
            warn(f"{dst.name} already exists -- skipping (won't overwrite)")
            continue
        info(f"  {src.name} -> {dst.name}")
        if not dry_run:
            shutil.copy2(src, dst)
            # Then truncate the live file to just the header (so the bot
            # has a clean slate if any asset is still live)
            with open(src, "r", newline="") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    header = []
            with open(src, "w", newline="") as f:
                if header:
                    csv.writer(f).writerow(header)


def update_account_state(
    assets: list[str], reset: bool, reset_balance: float, dry_run: bool
) -> None:
    """Update account_state.json for the rolled-back assets."""
    if not ACCOUNT_STATE.exists():
        warn(f"{ACCOUNT_STATE.name} not found")
        return

    with open(ACCOUNT_STATE, "r") as f:
        state = json.load(f)

    if reset:
        info(f"Resetting {len(assets)} sub-account(s) to ${reset_balance:.2f}:")
    else:
        info(f"Keeping current balance for {len(assets)} sub-account(s):")

    for asset in assets:
        series = ASSET_TO_SERIES[asset]
        current = state.get(series, {}).get("balance_dollars", 0.0)
        current_pnl = state.get(series, {}).get("pnl_dollars", 0.0)
        if reset:
            info(
                f"  {asset} ({series}): ${current:.2f} -> ${reset_balance:.2f}, "
                f"pnl=${current_pnl:+.2f} -> $0.00"
            )
            if not dry_run:
                state[series] = {
                    "name": series,
                    "balance_dollars": float(reset_balance),
                    "reserved_dollars": 0.0,
                    "pnl_dollars": 0.0,
                }
        else:
            info(
                f"  {asset} ({series}): balance=${current:.2f} kept, "
                f"reserved set to $0.00 (in-flight trades cleared)"
            )
            if not dry_run:
                state[series] = {
                    "name": series,
                    "balance_dollars": float(current),
                    "reserved_dollars": 0.0,
                    "pnl_dollars": float(current_pnl),
                }

    if not dry_run:
        with open(ACCOUNT_STATE, "w") as f:
            json.dump(state, f, indent=2)


def update_config(assets: list[str], reset: bool, reset_balance: float, dry_run: bool) -> None:
    """Set dry_run=true and reset initial_balance for rolled-back assets."""
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
        new_bal = reset_balance if reset else asset_cfg.get("initial_balance", 25.0)
        info(f"  {asset}: dry_run={old_dry} -> true, initial_balance={old_bal} -> {new_bal}")
        if not dry_run:
            asset_cfg["dry_run"] = True
            asset_cfg["initial_balance"] = float(new_bal)

    if not dry_run:
        with open(CONFIG_JSON, "w") as f:
            json.dump(cfg, f, indent=2)


def main() -> int:
    args = parse_args()
    assets = validate_assets(args.assets)
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    mode = "RESET" if args.reset else "KEEP-BALANCE"

    print()
    print("=" * 72)
    print("GO DRY (rollback)")
    print("=" * 72)
    info(f"Assets to roll back: {', '.join(assets)}")
    info(f"Mode: {mode}")
    if args.reset:
        info(f"Reset balance: ${args.reset_balance:.2f}")
    info(f"Date tag: {date_tag}")
    if args.dry_run:
        warn("DRY RUN MODE: no files will be modified")
    print()

    # Filter to only currently-live assets
    live_assets = check_assets_are_live(assets)
    if not live_assets:
        err("No assets to roll back (none are currently live)")
        return 1
    print()

    # Pending verifications check
    if not check_pending_verifications(live_assets, args.force):
        return 1
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

    # 1. Archive live CSVs
    print("[1/3] Archiving live CSVs...")
    archive_live_files(date_tag, args.dry_run)
    print()

    # 2. Update account state
    print("[2/3] Updating account_state.json...")
    update_account_state(live_assets, args.reset, args.reset_balance, args.dry_run)
    print()

    # 3. Update config
    print("[3/3] Updating config/trading.json...")
    update_config(live_assets, args.reset, args.reset_balance, args.dry_run)
    print()

    print("=" * 72)
    if args.dry_run:
        print("DRY RUN COMPLETE -- no files modified")
    else:
        print("ROLLBACK COMPLETE")
    print("=" * 72)
    print()
    print("Next steps:")
    print("  1. Restart the bot (it should pick up the new config on startup)")
    print(f"  2. The {len(live_assets)} rolled-back asset(s) will run as dry-run again")
    print("  3. The live data is preserved in:")
    print(f"     - output/trades_live_archive_{date_tag}.csv")
    print(f"     - output/balance_live_archive_{date_tag}.csv")
    print("  4. Your money is STILL IN YOUR KALSHI ACCOUNT.")
    print("     Withdraw it manually through Kalshi if you want it back.")
    print()
    print("To go live again, run:  python scripts/go_live.py SOL XRP --balance 50")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
