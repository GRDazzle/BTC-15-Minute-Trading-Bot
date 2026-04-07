"""Ledger reconciliation: rebuild per-asset balances from trade history.

Compares three sources of truth for each sub-account:
  1. Ledger     -- rebuilt from initial_balance + Σ pnl + Σ deposits
  2. In-memory  -- data/account_state.json (the bot's live working state)
  3. Kalshi     -- live API balance (only fetched with --kalshi flag)

Reports per-asset and total discrepancies. Read-only -- never modifies
state.

Usage:
    python scripts/reconcile.py
    python scripts/reconcile.py --kalshi
    python scripts/reconcile.py --state-file data/account_state.json --trades output/trades.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_STATE_PATH = PROJECT_ROOT / "data" / "account_state.json"
DEFAULT_TRADES_PATH = PROJECT_ROOT / "output" / "trades.csv"
DEFAULT_DEPOSITS_LOG = PROJECT_ROOT / "data" / "deposits_log.csv"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"

ASSET_TO_SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
    "HYPE": "KXHYPE15M",
    "BNB": "KXBNB15M",
    "DOGE": "KXDOGE15M",
}
SERIES_TO_ASSET = {v: k for k, v in ASSET_TO_SERIES.items()}


def load_initial_balances(config_path: Path) -> dict[str, float]:
    """Read initial_balance per asset from config/trading.json."""
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        cfg = json.load(f)
    defaults = cfg.get("defaults", {})
    default_balance = defaults.get("initial_balance", 25.0)
    out: dict[str, float] = {}
    for asset, asset_cfg in cfg.get("assets", {}).items():
        out[asset.upper()] = float(asset_cfg.get("initial_balance", default_balance))
    return out


def sum_pnl_per_asset(trades_path: Path) -> tuple[dict[str, float], dict[str, int]]:
    """Walk trades.csv and sum PnL per asset.

    Returns (pnl_by_asset, count_by_asset). Skips rows with empty/pending pnl.
    """
    pnl: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    if not trades_path.exists():
        return dict(pnl), dict(counts)
    with open(trades_path, "r") as f:
        for row in csv.DictReader(f):
            asset = (row.get("asset") or "").strip()
            if not asset:
                continue
            pnl_str = (row.get("pnl") or "").strip()
            if not pnl_str or pnl_str.lower() == "pending":
                continue
            try:
                pnl[asset] += float(pnl_str)
                counts[asset] += 1
            except ValueError:
                continue
    return dict(pnl), dict(counts)


def sum_deposits_per_asset(deposits_log: Path) -> dict[str, float]:
    """Walk deposits_log.csv and sum applied deposits per asset.

    Returns empty dict if the log doesn't exist (deposits weren't tracked
    yet -- expected for older state).
    """
    deposits: dict[str, float] = defaultdict(float)
    if not deposits_log.exists():
        return dict(deposits)
    with open(deposits_log, "r") as f:
        for row in csv.DictReader(f):
            asset = (row.get("asset") or "").strip()
            amt_str = (row.get("amount_dollars") or "").strip()
            if not asset or not amt_str:
                continue
            try:
                deposits[asset] += float(amt_str)
            except ValueError:
                continue
    return dict(deposits)


def load_in_memory_balances(state_path: Path) -> dict[str, float]:
    """Read each sub-account's current balance from account_state.json."""
    if not state_path.exists():
        return {}
    with open(state_path, "r") as f:
        state = json.load(f)
    out: dict[str, float] = {}
    for series, acct in state.items():
        asset = SERIES_TO_ASSET.get(series, series)
        out[asset] = float(acct.get("balance_dollars", 0.0))
    return out


def fetch_kalshi_balance() -> Optional[float]:
    """Fetch the live Kalshi account balance. Returns None on failure.

    Imports the SDK lazily so the script still runs in environments
    without Kalshi credentials configured.
    """
    try:
        from sdk.kalshi.client import KalshiClient, load_config
        from sdk.kalshi.orders import fetch_balance
    except Exception as e:
        print(f"  [warn] Could not import Kalshi SDK: {e}")
        return None
    try:
        cfg = load_config()
        client = KalshiClient(cfg)
        return fetch_balance(client)
    except Exception as e:
        print(f"  [warn] Kalshi balance fetch failed: {e}")
        return None


def reconcile(
    state_path: Path,
    trades_path: Path,
    deposits_log: Path,
    config_path: Path,
    fetch_kalshi: bool = False,
) -> int:
    """Run the reconciliation and print a report.

    Returns exit code: 0 if all assets agree within $0.50, 1 otherwise.
    """
    initial = load_initial_balances(config_path)
    pnl, counts = sum_pnl_per_asset(trades_path)
    deposits = sum_deposits_per_asset(deposits_log)
    in_memory = load_in_memory_balances(state_path)

    # Universe of assets = anything that appears in any source
    all_assets = sorted(
        set(initial.keys()) | set(pnl.keys()) | set(deposits.keys()) | set(in_memory.keys())
    )

    print("=" * 78)
    print("LEDGER RECONCILIATION")
    print("=" * 78)
    print(f"  state file  : {state_path}")
    print(f"  trades csv  : {trades_path}")
    print(f"  deposits log: {deposits_log}{' (missing)' if not deposits_log.exists() else ''}")
    print(f"  config      : {config_path}")
    print()
    print(
        f"{'Asset':<7} {'Initial':>10} {'+ PnL':>10} {'+ Deposits':>12} "
        f"{'= Ledger':>10} {'In-Memory':>11} {'Diff':>10} {'Trades':>8}"
    )
    print("-" * 78)

    total_ledger = 0.0
    total_in_memory = 0.0
    biggest_drift = 0.0
    drifted: list[str] = []

    for asset in all_assets:
        init = initial.get(asset, 0.0)
        p = pnl.get(asset, 0.0)
        d = deposits.get(asset, 0.0)
        ledger = init + p + d
        memory = in_memory.get(asset, 0.0)
        diff = memory - ledger
        n = counts.get(asset, 0)

        total_ledger += ledger
        total_in_memory += memory
        if abs(diff) > abs(biggest_drift):
            biggest_drift = diff
        if abs(diff) > 0.50:
            drifted.append(asset)

        flag = "" if abs(diff) < 0.50 else " *"
        print(
            f"{asset:<7} ${init:>9.2f} ${p:>+9.2f} ${d:>+11.2f} "
            f"${ledger:>9.2f} ${memory:>10.2f} ${diff:>+9.2f}{flag} {n:>7}"
        )

    print("-" * 78)
    total_diff = total_in_memory - total_ledger
    print(
        f"{'TOTAL':<7} {'':>10} {'':>10} {'':>12} "
        f"${total_ledger:>9.2f} ${total_in_memory:>10.2f} ${total_diff:>+9.2f}"
    )
    print()

    # Optional Kalshi 3-way check
    if fetch_kalshi:
        print("Fetching live Kalshi balance...")
        kalshi_bal = fetch_kalshi_balance()
        if kalshi_bal is not None:
            kalshi_diff = kalshi_bal - total_in_memory
            print(
                f"  Kalshi total : ${kalshi_bal:.2f}  "
                f"(in-memory: ${total_in_memory:.2f}, diff: ${kalshi_diff:+.2f})"
            )
            if abs(kalshi_diff) > 0.50:
                print("  WARNING: Kalshi balance differs from in-memory total")
                print(
                    "  Note: Kalshi balance reflects ALL of your account, including "
                    "manual trades and any other bots. Local sub-accounts only track "
                    "what this bot has done."
                )
        print()

    if drifted:
        print(f"DRIFT DETECTED for {len(drifted)} asset(s): {', '.join(drifted)}")
        print("Possible causes:")
        print("  - Deposits made before deposits_log.csv tracking was added")
        print("  - Manual edits to account_state.json")
        print("  - Trades in flight (pending settlement, not in trades.csv yet)")
        print("  - Bot state divergence (bug)")
        return 1

    print("OK: all sub-accounts match within tolerance")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Ledger reconciliation report")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--trades", default=str(DEFAULT_TRADES_PATH))
    parser.add_argument("--deposits-log", default=str(DEFAULT_DEPOSITS_LOG))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--kalshi",
        action="store_true",
        help="Also fetch live Kalshi balance for 3-way check",
    )
    args = parser.parse_args()

    return reconcile(
        Path(args.state_file),
        Path(args.trades),
        Path(args.deposits_log),
        Path(args.config),
        fetch_kalshi=args.kalshi,
    )


if __name__ == "__main__":
    sys.exit(main())
