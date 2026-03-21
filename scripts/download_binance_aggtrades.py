"""
Download daily aggTrades CSVs from Binance Data Vision.

Usage:
  python scripts/download_binance_aggtrades.py --days 30 --assets BTC
  python scripts/download_binance_aggtrades.py --days 30 --assets BTC,ETH,SOL,XRP

Source: https://data.binance.vision/data/spot/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM-DD}.zip
CSV columns (no header): agg_trade_id, price, qty, first_id, last_id, timestamp, is_buyer_maker, best_price_match
Note: Since Jan 2025, timestamps are in microseconds (not milliseconds).
"""
import argparse
import hashlib
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades"

# Binance symbol mapping
ASSET_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}

BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"


def download_file(url: str, timeout: int = 120) -> bytes:
    """Download a file and return its bytes."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def verify_checksum(data: bytes, checksum_url: str) -> bool:
    """Verify SHA256 checksum against Binance-provided .CHECKSUM file."""
    try:
        resp = requests.get(checksum_url, timeout=30)
        resp.raise_for_status()
        # Format: "<sha256>  <filename>"
        expected_hash = resp.text.strip().split()[0].lower()
        actual_hash = hashlib.sha256(data).hexdigest().lower()
        return actual_hash == expected_hash
    except Exception as e:
        print(f"  Warning: Could not verify checksum ({e}), proceeding anyway")
        return True  # Don't block on checksum failure


def download_day(symbol: str, date: datetime, output_dir: Path) -> bool:
    """Download and extract aggTrades for one symbol/day.

    Returns True if file exists (downloaded or already present), False on failure.
    """
    date_str = date.strftime("%Y-%m-%d")
    csv_name = f"{symbol}-aggTrades-{date_str}.csv"
    csv_path = output_dir / csv_name

    # Skip if already exists
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return True

    zip_name = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{zip_name}"
    checksum_url = f"{url}.CHECKSUM"

    try:
        data = download_file(url)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print(f"  {date_str}: Not available (404)")
        else:
            print(f"  {date_str}: HTTP error: {e}")
        return False
    except Exception as e:
        print(f"  {date_str}: Download error: {e}")
        return False

    # Verify checksum
    if not verify_checksum(data, checksum_url):
        print(f"  {date_str}: CHECKSUM MISMATCH - skipping")
        return False

    # Extract CSV from ZIP
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            # Find the CSV file in the archive
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_files:
                print(f"  {date_str}: No CSV found in ZIP")
                return False
            zf.extract(csv_files[0], output_dir)
            # Rename if needed
            extracted = output_dir / csv_files[0]
            if extracted != csv_path:
                extracted.rename(csv_path)
    except zipfile.BadZipFile:
        print(f"  {date_str}: Bad ZIP file")
        return False

    size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"  {date_str}: OK ({size_mb:.1f} MB)")
    return True


def download_asset(asset: str, days: int) -> None:
    """Download N days of aggTrades for one asset."""
    symbol = ASSET_SYMBOLS.get(asset.upper())
    if symbol is None:
        print(f"Unknown asset: {asset}. Available: {list(ASSET_SYMBOLS.keys())}")
        return

    output_dir = DATA_DIR / asset.upper()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Downloading {symbol} aggTrades ({days} days) ===")
    print(f"Output: {output_dir}")

    today = datetime.now(timezone.utc).date()
    success = 0
    failed = 0

    for i in range(days, 0, -1):
        date = datetime.combine(today - timedelta(days=i), datetime.min.time())
        ok = download_day(symbol, date, output_dir)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{asset}: {success} downloaded, {failed} failed/skipped")


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance aggTrades data from data.binance.vision"
    )
    parser.add_argument(
        "--assets",
        required=True,
        help="Asset(s) to download, comma-separated (e.g. BTC or BTC,ETH,SOL,XRP)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to download (default: 30)",
    )
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")]
    for asset in assets:
        download_asset(asset, args.days)

    print("\nDone.")


if __name__ == "__main__":
    main()
