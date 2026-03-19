#!/usr/bin/env python3
"""Download Binance 1-minute klines for backtesting.

Usage:
    python scripts/download_binance_klines.py
    python scripts/download_binance_klines.py --days 30 --assets BTC
    python scripts/download_binance_klines.py --days 90 --assets BTC,ETH,SOL,XRP

Output: data/historical/binance/{symbol}_1m.csv
Columns: timestamp, open, high, low, close, volume, quote_volume,
         trades, taker_buy_volume, taker_buy_quote_volume
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.binance.us/api/v3/klines"

ASSET_SYMBOLS = {
    "BTC": "BTCUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
    "XRP": "XRPUSD",
}

COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

OUTPUT_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]

LIMIT = 1000  # max per request


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """Fetch up to 1000 klines from Binance."""
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_symbol(symbol: str, days: int, output_dir: Path) -> Path:
    """Download all 1m klines for a symbol and save to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol}_1m.csv"

    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - timedelta(days=days)).timestamp() * 1000)

    all_rows: list[list] = []
    cursor = start_ms
    request_count = 0

    print(f"  Downloading {symbol} ({days} days)...")

    while cursor < end_ms:
        batch = fetch_klines(symbol, cursor, end_ms)
        if not batch:
            break

        all_rows.extend(batch)
        request_count += 1

        # Move cursor past the last candle's open time
        last_open_ms = batch[-1][0]
        cursor = last_open_ms + 60_000  # next minute

        if request_count % 10 == 0:
            pct = min(100, (cursor - start_ms) / (end_ms - start_ms) * 100)
            print(f"    {request_count} requests, {len(all_rows)} candles ({pct:.0f}%)")

        # Respect rate limits: 1200 req/min for Binance
        if request_count % 50 == 0:
            time.sleep(1)

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        for row in all_rows:
            mapped = dict(zip(COLUMNS, row))
            # Convert timestamp from ms to ISO8601
            ts = datetime.fromtimestamp(mapped["timestamp"] / 1000, tz=timezone.utc)
            writer.writerow([
                ts.isoformat(),
                mapped["open"],
                mapped["high"],
                mapped["low"],
                mapped["close"],
                mapped["volume"],
                mapped["quote_volume"],
                mapped["trades"],
                mapped["taker_buy_volume"],
                mapped["taker_buy_quote_volume"],
            ])

    print(f"  Saved {len(all_rows)} candles -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download Binance 1m klines")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    parser.add_argument(
        "--assets", type=str, default="BTC,ETH,SOL,XRP",
        help="Comma-separated assets",
    )
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")]
    output_dir = Path("data/historical/binance")

    print(f"Binance 1m Kline Downloader")
    print(f"  Days: {args.days}")
    print(f"  Assets: {', '.join(assets)}")
    print(f"  Output: {output_dir}/")
    print()

    for asset in assets:
        symbol = ASSET_SYMBOLS.get(asset)
        if not symbol:
            print(f"  Unknown asset: {asset}, skipping")
            continue
        download_symbol(symbol, args.days, output_dir)
        time.sleep(0.5)  # brief pause between assets

    print("\nDone.")


if __name__ == "__main__":
    main()
