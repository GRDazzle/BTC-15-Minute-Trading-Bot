"""Add balance to a sub-account by queueing a deposit.

The bot polls data/deposits.json at each window boundary and applies pending
deposits to the in-memory account state. Direct edits to account_state.json
get overwritten because the bot writes its own state periodically.

Usage:
    python scripts/add_balance.py BTC 25
    python scripts/add_balance.py SOL 10.50
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPOSITS_PATH = PROJECT_ROOT / "data" / "deposits.json"

ASSET_TO_SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
    "HYPE": "KXHYPE15M",
    "BNB": "KXBNB15M",
    "DOGE": "KXDOGE15M",
}


def main():
    parser = argparse.ArgumentParser(description="Queue a balance deposit for the live bot")
    parser.add_argument("asset", help="Asset name (e.g. BTC, ETH)")
    parser.add_argument("amount", type=float, help="Amount in dollars")
    args = parser.parse_args()

    asset = args.asset.upper()
    if asset not in ASSET_TO_SERIES:
        print(f"Unknown asset: {asset}. Valid: {list(ASSET_TO_SERIES.keys())}")
        sys.exit(1)

    if args.amount <= 0:
        print("Amount must be positive")
        sys.exit(1)

    series = ASSET_TO_SERIES[asset]

    # Load existing queue
    DEPOSITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEPOSITS_PATH.exists():
        with open(DEPOSITS_PATH, "r") as f:
            queue = json.load(f)
    else:
        queue = {"pending": []}

    # Add deposit
    deposit = {
        "asset": asset,
        "series": series,
        "amount": args.amount,
        "queued_at": time.time(),
    }
    queue["pending"].append(deposit)

    with open(DEPOSITS_PATH, "w") as f:
        json.dump(queue, f, indent=2)

    print(f"Queued deposit: {asset} +${args.amount:.2f}")
    print(f"Bot will apply at next window boundary (within 15 min)")


if __name__ == "__main__":
    main()
