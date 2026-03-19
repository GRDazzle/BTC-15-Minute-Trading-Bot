#!/usr/bin/env python3
"""Download Coinbase 1-minute candles for backtesting.

Usage:
    python scripts/download_coinbase_candles.py
    python scripts/download_coinbase_candles.py --days 30 --assets BTC

Output: data/historical/coinbase/{product}_1m.csv
Columns: timestamp, open, high, low, close, volume

Coinbase returns max 300 candles per request.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.exchange.coinbase.com/products"

ASSET_PRODUCTS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
}

COLUMNS = ["timestamp", "low", "high", "open", "close", "volume"]
OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

MAX_CANDLES = 300  # Coinbase limit per request
GRANULARITY = 60   # 1 minute in seconds


def fetch_candles(product: str, start_iso: str, end_iso: str) -> list[list]:
    """Fetch candles from Coinbase. Returns newest-first order."""
    url = f"{BASE_URL}/{product}/candles"
    params = {
        "start": start_iso,
        "end": end_iso,
        "granularity": GRANULARITY,
    }
    headers = {"Accept": "application/json"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_product(product: str, days: int, output_dir: Path) -> Path:
    """Download all 1m candles for a product and save to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{product.replace('-', '')}_1m.csv"

    now = datetime.now(timezone.utc)
    end_dt = now
    start_dt = now - timedelta(days=days)

    all_rows: list[list] = []
    cursor = start_dt
    request_count = 0
    # Each request covers MAX_CANDLES minutes
    chunk = timedelta(minutes=MAX_CANDLES)

    print(f"  Downloading {product} ({days} days)...")

    while cursor < end_dt:
        chunk_end = min(cursor + chunk, end_dt)
        try:
            batch = fetch_candles(
                product,
                cursor.isoformat(),
                chunk_end.isoformat(),
            )
        except requests.HTTPError as e:
            print(f"    HTTP error at {cursor}: {e}, retrying in 5s...")
            time.sleep(5)
            continue

        if batch:
            all_rows.extend(batch)

        cursor = chunk_end
        request_count += 1

        if request_count % 20 == 0:
            pct = min(100, (cursor - start_dt) / (end_dt - start_dt) * 100)
            print(f"    {request_count} requests, {len(all_rows)} candles ({pct:.0f}%)")

        # Coinbase rate limit: 10 req/sec
        time.sleep(0.15)

    # Coinbase returns [timestamp, low, high, open, close, volume]
    # Sort by timestamp ascending (Coinbase returns newest first)
    all_rows.sort(key=lambda r: r[0])

    # Deduplicate by timestamp
    seen = set()
    unique_rows = []
    for row in all_rows:
        if row[0] not in seen:
            seen.add(row[0])
            unique_rows.append(row)

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        for row in unique_rows:
            ts_unix, low, high, opn, close, volume = row
            ts = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
            writer.writerow([ts.isoformat(), opn, high, low, close, volume])

    print(f"  Saved {len(unique_rows)} candles -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download Coinbase 1m candles")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    parser.add_argument(
        "--assets", type=str, default="BTC,ETH,SOL,XRP",
        help="Comma-separated assets",
    )
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")]
    output_dir = Path("data/historical/coinbase")

    print(f"Coinbase 1m Candle Downloader")
    print(f"  Days: {args.days}")
    print(f"  Assets: {', '.join(assets)}")
    print(f"  Output: {output_dir}/")
    print()

    for asset in assets:
        product = ASSET_PRODUCTS.get(asset)
        if not product:
            print(f"  Unknown asset: {asset}, skipping")
            continue
        download_product(product, args.days, output_dir)
        time.sleep(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
