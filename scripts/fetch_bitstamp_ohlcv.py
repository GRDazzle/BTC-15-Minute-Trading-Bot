"""Download historical 1-minute OHLCV candles from Bitstamp.

Paginates with the 'start' parameter. Saves daily CSV files.

Usage:
    python scripts/fetch_bitstamp_ohlcv.py --assets BTC,ETH,SOL,XRP --days 45
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "ohlcv_bitstamp"

ASSET_TO_PAIR = {
    "BTC": "btcusd",
    "ETH": "ethusd",
    "SOL": "solusd",
    "XRP": "xrpusd",
    "HYPE": "hypeusd",
    "BNB": "bnbusd",
    "DOGE": "dogeusd",
}


def fetch_ohlcv(pair, start_ts, step=60, limit=1000):
    """Fetch OHLCV candles. Returns list of candle dicts."""
    url = (
        f"https://www.bitstamp.net/api/v2/ohlc/{pair}/"
        f"?step={step}&limit={limit}&start={start_ts}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    for attempt in range(5):
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            ohlc = data.get("data", {}).get("ohlc", [])
            return ohlc
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP error: {e.code}")
                return []
        except Exception as e:
            if attempt < 4:
                time.sleep(3)
            else:
                print(f"  Error: {e}")
                return []
    return []


def download_asset(asset, days):
    """Download OHLCV for one asset."""
    pair = ASSET_TO_PAIR.get(asset)
    if not pair:
        print(f"Unknown asset: {asset}")
        return

    asset_dir = OUTPUT_DIR / asset
    asset_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    current_ts = int(start_date.timestamp())
    end_ts = int(time.time())

    print(f"\n{asset} ({pair}): downloading from {start_date.strftime('%Y-%m-%d')}")

    # Collect all candles
    daily_candles = {}
    total_candles = 0

    while current_ts < end_ts:
        candles = fetch_ohlcv(pair, current_ts)
        if not candles:
            break

        for c in candles:
            ts = int(c["timestamp"])
            if ts > end_ts:
                break
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")

            if date_str not in daily_candles:
                daily_candles[date_str] = []
            daily_candles[date_str].append({
                "timestamp": ts,
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c["volume"]),
            })
            total_candles += 1

        # Advance cursor past the last candle
        last_ts = int(candles[-1]["timestamp"])
        if last_ts <= current_ts:
            break  # No progress
        current_ts = last_ts + 60  # Next minute

        if total_candles % 5000 == 0:
            current_date = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  {total_candles:,} candles, at {current_date}...")

        time.sleep(6.5)  # Rate limit: ~10 req/min

    # Write daily CSVs
    files_written = 0
    for date_str in sorted(daily_candles.keys()):
        out_path = asset_dir / f"{asset}-ohlcv-{date_str}.csv"
        if out_path.exists():
            continue
        candles_day = daily_candles[date_str]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
            writer.writeheader()
            for c in sorted(candles_day, key=lambda x: x["timestamp"]):
                writer.writerow(c)
        files_written += 1

    print(f"  Done: {total_candles:,} candles, {files_written} new files, {len(daily_candles)} dates")


def main():
    parser = argparse.ArgumentParser(description="Download Bitstamp historical OHLCV")
    parser.add_argument("--assets", default="BTC,ETH,SOL,XRP")
    parser.add_argument("--days", type=int, default=45)
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")]
    for asset in assets:
        download_asset(asset, args.days)

    print("\nDone.")


if __name__ == "__main__":
    main()
