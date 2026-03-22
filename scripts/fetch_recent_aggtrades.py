"""
Fetch recent aggTrades from Binance.US REST API to fill the gap between
Data Vision daily CSVs (always 1+ day stale) and live Kalshi polling data.

Usage:
  python scripts/fetch_recent_aggtrades.py --assets BTC --hours 12
  python scripts/fetch_recent_aggtrades.py --assets BTC,ETH,SOL,XRP --hours 48

Endpoint: GET https://api.binance.us/api/v3/aggTrades
CSV columns (no header): agg_trade_id, price, qty, first_id, last_id, timestamp, is_buyer_maker, best_price_match
Note: Timestamps are written in microseconds to match Data Vision format (since Jan 2025).
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

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades"

# Binance symbol mapping (matches download_binance_aggtrades.py)
ASSET_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}

# Fallback symbols if USDT pair is unavailable on Binance.US
ASSET_SYMBOLS_FALLBACK = {
    "BTC": "BTCUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
    "XRP": "XRPUSD",
}

BASE_URL = "https://api.binance.us/api/v3/aggTrades"
MAX_LIMIT = 1000
RATE_SLEEP = 0.06  # ~16 req/sec, well under 1200/min limit
MAX_RETRIES = 3


def fetch_page(symbol: str, from_id: int | None = None,
               start_time: int | None = None) -> list[dict]:
    """Fetch one page of aggTrades from Binance.US REST API.

    Args:
        symbol: Trading pair (e.g. BTCUSDT)
        from_id: Fetch trades starting from this agg_trade_id (exclusive pagination)
        start_time: Fetch trades starting from this timestamp in ms (only for first request)

    Returns:
        List of trade dicts from the API.
    """
    params = {"symbol": symbol, "limit": MAX_LIMIT}
    if from_id is not None:
        params["fromId"] = from_id
    elif start_time is not None:
        params["startTime"] = start_time

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                raise  # Don't retry 400s (bad symbol)
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"  Retry {attempt}/{MAX_RETRIES} after HTTP {e.response.status_code if e.response else '?'} (wait {wait}s)")
                time.sleep(wait)
            else:
                raise
        except requests.RequestException as e:
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"  Retry {attempt}/{MAX_RETRIES} after {type(e).__name__} (wait {wait}s)")
                time.sleep(wait)
            else:
                raise


def resolve_symbol(asset: str) -> str:
    """Try primary symbol (USDT), fallback to USD if Binance.US returns 400."""
    primary = ASSET_SYMBOLS.get(asset)
    if primary is None:
        raise ValueError(f"Unknown asset: {asset}")

    # Quick probe: fetch 1 trade
    try:
        resp = requests.get(BASE_URL, params={"symbol": primary, "limit": 1}, timeout=10)
        resp.raise_for_status()
        return primary
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 400:
            fallback = ASSET_SYMBOLS_FALLBACK.get(asset)
            if fallback:
                print(f"  {primary} returned 400, trying {fallback}...")
                try:
                    resp2 = requests.get(BASE_URL, params={"symbol": fallback, "limit": 1}, timeout=10)
                    resp2.raise_for_status()
                    return fallback
                except Exception:
                    pass
        raise


def find_last_timestamp_in_csvs(asset: str) -> int | None:
    """Find the last timestamp (in ms) from existing CSV files for this asset.

    Reads the last line of the most recent CSV. Handles both Data Vision
    (Binance.com, IDs ~3.9B) and Binance.US REST (IDs ~29M) CSVs.
    Timestamps in CSVs are in microseconds; we convert to ms for the API.
    """
    asset_dir = DATA_DIR / asset.upper()
    if not asset_dir.exists():
        return None

    # Find all aggTrades CSVs regardless of symbol prefix
    csvs = sorted(asset_dir.glob("*-aggTrades-*.csv"))
    if not csvs:
        return None

    # Read last line of most recent CSV
    last_csv = csvs[-1]
    try:
        with open(last_csv, "rb") as f:
            # Seek to near end for efficiency
            f.seek(0, 2)
            size = f.tell()
            seek_pos = max(0, size - 4096)
            f.seek(seek_pos)
            lines = f.read().decode("utf-8", errors="replace").strip().splitlines()
            if lines:
                last_line = lines[-1].strip()
                if last_line:
                    fields = last_line.split(",")
                    ts_raw = int(fields[5])  # timestamp column
                    # Data Vision uses microseconds (since Jan 2025)
                    # Heuristic: if > 1e15, it's microseconds; convert to ms
                    if ts_raw > 1_000_000_000_000_000:
                        ts_ms = ts_raw // 1000
                    else:
                        ts_ms = ts_raw
                    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                    print(f"  Last timestamp in {last_csv.name}: {dt.isoformat()} ({ts_ms}ms)")
                    return ts_ms
    except Exception as e:
        print(f"  Warning: Could not read last timestamp from {last_csv.name}: {e}")

    return None


def trade_to_csv_row(trade: dict) -> list:
    """Convert API trade dict to CSV row matching Data Vision format.

    API returns timestamps in milliseconds; we convert to microseconds
    to match Data Vision format (since Jan 2025).
    """
    ts_ms = trade["T"]
    ts_us = ts_ms * 1000  # ms -> us
    return [
        trade["a"],       # agg_trade_id
        trade["p"],       # price
        trade["q"],       # qty
        trade["f"],       # first_id
        trade["l"],       # last_id
        ts_us,            # timestamp (microseconds)
        trade["m"],       # is_buyer_maker
        trade["M"],       # best_price_match
    ]


def fetch_asset(asset: str, hours: int) -> int:
    """Fetch recent aggTrades for one asset. Returns number of trades written."""
    asset = asset.upper()
    print(f"\n=== Fetching {asset} aggTrades (last {hours}h) ===")

    try:
        symbol = resolve_symbol(asset)
    except Exception as e:
        print(f"  ERROR: Could not resolve symbol for {asset}: {e}")
        return 0

    print(f"  Symbol: {symbol}")

    asset_dir = DATA_DIR / asset
    asset_dir.mkdir(parents=True, exist_ok=True)

    # Determine starting point
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    cutoff_ms = now_ms - (hours * 3600 * 1000)

    # Check existing CSVs for the latest timestamp (works across ID spaces)
    last_ts_ms = find_last_timestamp_in_csvs(asset)
    if last_ts_ms is not None and last_ts_ms > cutoff_ms:
        # Resume from where existing data ends (+1ms to avoid overlap)
        start_time = last_ts_ms + 1
        print(f"  Resuming from existing data end + 1ms")
    else:
        start_time = cutoff_ms
        print(f"  Starting from {hours}h ago")

    from_id = None  # First request always uses startTime

    # Pagination loop
    total_written = 0
    # Track open file handles per day to avoid re-opening
    open_writers: dict[str, tuple] = {}  # date_str -> (file_handle, csv_writer)

    try:
        page_num = 0
        while True:
            page_num += 1
            trades = fetch_page(symbol, from_id=from_id, start_time=start_time)

            if not trades:
                print(f"  No more trades (page {page_num})")
                break

            # After first request, switch to fromId pagination
            start_time = None

            for trade in trades:
                ts_ms = trade["T"]

                # Stop if we've gone past now
                if ts_ms > now_ms:
                    break

                # Determine which daily CSV to write to
                trade_date = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                date_str = trade_date.strftime("%Y-%m-%d")
                csv_name = f"{symbol}-aggTrades-{date_str}.csv"

                if date_str not in open_writers:
                    csv_path = asset_dir / csv_name
                    fh = open(csv_path, "a", newline="", encoding="utf-8")
                    writer = csv.writer(fh)
                    open_writers[date_str] = (fh, writer)

                _, writer = open_writers[date_str]
                writer.writerow(trade_to_csv_row(trade))
                total_written += 1

            # Check if last trade is beyond now
            last_trade_ts = trades[-1]["T"]
            if last_trade_ts > now_ms:
                break

            # Set up next page
            from_id = trades[-1]["a"] + 1

            # Rate limiting
            time.sleep(RATE_SLEEP)

            # Progress logging every 50 pages
            if page_num % 50 == 0:
                pct_done = min(100, (last_trade_ts - cutoff_ms) / max(1, now_ms - cutoff_ms) * 100)
                print(f"  Page {page_num}: {total_written} trades written ({pct_done:.0f}%)")

    finally:
        # Close all open file handles
        for fh, _ in open_writers.values():
            fh.close()

    print(f"  {asset}: {total_written} trades written across {len(open_writers)} file(s)")
    return total_written


def main():
    parser = argparse.ArgumentParser(
        description="Fetch recent aggTrades from Binance.US REST API"
    )
    parser.add_argument(
        "--assets",
        required=True,
        help="Asset(s) to fetch, comma-separated (e.g. BTC or BTC,ETH,SOL,XRP)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=12,
        help="Hours of history to fetch (default: 12)",
    )
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")]
    total = 0
    for asset in assets:
        try:
            n = fetch_asset(asset, args.hours)
            total += n
        except Exception as e:
            print(f"  ERROR fetching {asset}: {e}")

    print(f"\nDone. {total} total trades written.")


if __name__ == "__main__":
    main()
