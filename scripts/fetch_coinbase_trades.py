"""
Download historical trade data from Coinbase Exchange REST API.

Saves in the same aggTrade CSV format as Binance Data Vision for
compatibility with the training pipeline.

No authentication required — public endpoint.

Usage:
    python scripts/fetch_coinbase_trades.py --assets BTC,ETH,SOL,XRP --days 30
    python scripts/fetch_coinbase_trades.py --assets BTC --days 7
"""
import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
API_URL = "https://api.exchange.coinbase.com"

ASSET_TO_PRODUCT = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
}

ASSET_TO_SYMBOL = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}


def fetch_trades_page(product_id: str, after: int | None = None) -> list[dict]:
    """Fetch one page of trades (up to 1000). Returns newest first."""
    url = f"{API_URL}/products/{product_id}/trades"
    params = {"limit": 1000}
    if after is not None:
        params["after"] = after  # Get trades older than this ID

    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < 4:
                time.sleep(2)
            else:
                print(f"    Failed after 5 attempts: {e}")
                return []
    return []


def fetch_asset(asset: str, days: int) -> None:
    """Download trades for one asset, save to daily CSVs."""
    product_id = ASSET_TO_PRODUCT.get(asset.upper())
    symbol = ASSET_TO_SYMBOL.get(asset.upper())
    if not product_id:
        print(f"  Unknown asset: {asset}")
        return

    asset_dir = DATA_DIR / asset.upper()
    asset_dir.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    print(f"\n=== Fetching {asset} ({product_id}) last {days} days ===")
    print(f"  Cutoff: {cutoff.strftime('%Y-%m-%d %H:%M UTC')}")

    # Open file handles per day
    files: dict[str, tuple] = {}
    total_trades = 0
    after_id = None
    done = False
    page = 0

    while not done:
        page += 1
        trades = fetch_trades_page(product_id, after=after_id)

        if not trades:
            break

        for trade in trades:
            # Parse timestamp
            ts_str = trade["time"]
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

            if ts < cutoff:
                done = True
                break

            # Determine daily file
            date_str = ts.strftime("%Y-%m-%d")
            if date_str not in files:
                csv_path = asset_dir / f"{symbol}-aggTrades-{date_str}.csv"
                # Check if file exists and has data (don't overwrite)
                if csv_path.exists() and csv_path.stat().st_size > 100:
                    # Skip this date — already have data
                    files[date_str] = None
                else:
                    fh = open(csv_path, "w", newline="", encoding="utf-8")
                    files[date_str] = (fh, csv.writer(fh))

            writer_entry = files.get(date_str)
            if writer_entry is None:
                continue  # Skip, file already exists

            fh, writer = writer_entry

            # Convert to aggTrade format
            ts_us = int(ts.timestamp() * 1_000_000)
            is_buyer_maker = trade["side"] != "buy"

            writer.writerow([
                trade["trade_id"],
                trade["price"],
                trade["size"],
                0,  # first_id (N/A for Coinbase)
                0,  # last_id (N/A for Coinbase)
                ts_us,
                is_buyer_maker,
                True,  # best_price_match
            ])
            total_trades += 1

        # Pagination: get next page (older trades)
        if trades:
            # Smallest trade_id in this page = oldest
            min_id = min(t["trade_id"] for t in trades)
            after_id = min_id

        if page % 50 == 0:
            print(f"  Page {page}: {total_trades:,} trades so far...")

        # Rate limit: 10 req/s — minimal delay, let 429 handling manage bursts
        time.sleep(0.05)

    # Close files and sort each one (trades come newest-first, need oldest-first)
    for date_str, entry in files.items():
        if entry is None:
            continue
        fh, _ = entry
        fh.close()

    # Re-sort files (Coinbase returns newest first, we need oldest first)
    print(f"  Sorting {len([e for e in files.values() if e is not None])} files...")
    for date_str, entry in files.items():
        if entry is None:
            continue
        csv_path = asset_dir / f"{symbol}-aggTrades-{date_str}.csv"
        if not csv_path.exists():
            continue
        # Read, sort by timestamp, rewrite
        rows = []
        with open(csv_path, "r") as f:
            for row in csv.reader(f):
                if row:
                    rows.append(row)
        rows.sort(key=lambda r: int(r[5]) if len(r) > 5 else 0)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    print(f"  {asset}: {total_trades:,} trades written across {len([e for e in files.values() if e is not None])} files")


def main():
    parser = argparse.ArgumentParser(description="Download Coinbase historical trades")
    parser.add_argument("--assets", default="BTC,ETH,SOL,XRP")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")]
    for asset in assets:
        fetch_asset(asset, args.days)

    print("\nDone.")


if __name__ == "__main__":
    main()
