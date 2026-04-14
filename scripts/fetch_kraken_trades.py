"""Download historical tick-level trades from Kraken.

Paginates with the 'since' cursor to fetch full trade history.
Saves daily CSV files in the same format as Coinbase aggTrades.

Usage:
    python scripts/fetch_kraken_trades.py --assets BTC,ETH,SOL,XRP --days 45
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "aggtrades_kraken"

ASSET_TO_PAIR = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
    "XRP": "XRPUSD",
    "HYPE": "HYPEUSD",
    "BNB": "BNBUSD",
    # DOGE not listed on Kraken
}

ASSET_TO_SYMBOL = {
    "BTC": "BTCUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
    "XRP": "XRPUSD",
    "HYPE": "HYPEUSD",
    "BNB": "BNBUSD",
}


def fetch_trades(pair, since_ns):
    """Fetch up to 1000 trades from Kraken. Returns (trades, last_cursor)."""
    url = f"https://api.kraken.com/0/public/Trades?pair={pair}&since={since_ns}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    for attempt in range(5):
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            if "error" in data and data["error"]:
                if any("EGeneral" in e or "rate" in e.lower() for e in data["error"]):
                    time.sleep(3)
                    continue
                print(f"  API error: {data['error']}")
                return [], since_ns
            result = data.get("result", {})
            last = result.get("last", since_ns)
            # Find trades (key varies: XXBTZUSD, XBTUSD, etc.)
            trades = []
            for key, val in result.items():
                if isinstance(val, list) and val and isinstance(val[0], list):
                    trades = val
                    break
            return trades, int(last)
        except Exception as e:
            if attempt < 4:
                time.sleep(2)
            else:
                print(f"  Error: {e}")
                return [], since_ns
    return [], since_ns


def download_asset(asset, days):
    """Download trades for one asset."""
    pair = ASSET_TO_PAIR.get(asset)
    symbol = ASSET_TO_SYMBOL.get(asset)
    if not pair:
        print(f"Unknown asset: {asset}")
        return

    asset_dir = OUTPUT_DIR / asset
    asset_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    since_ns = int(start_date.timestamp() * 1e9)
    end_ts = time.time()

    print(f"\n{asset} ({pair}): downloading from {start_date.strftime('%Y-%m-%d')}")

    # Group trades by date
    daily_trades = {}
    total_trades = 0
    last_print = time.time()

    while True:
        trades, since_ns = fetch_trades(pair, since_ns)
        if not trades:
            break

        for trade in trades:
            # [price, volume, timestamp, buy/sell, market/limit, misc, trade_id]
            price = float(trade[0])
            qty = float(trade[1])
            ts = float(trade[2])
            side = trade[3]  # 'b' or 's'

            if ts > end_ts:
                break

            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")

            if date_str not in daily_trades:
                daily_trades[date_str] = []
            daily_trades[date_str].append({
                "price": price,
                "qty": qty,
                "timestamp": int(ts * 1000),  # ms
                "is_buyer_maker": side == "b",
            })
            total_trades += 1

        # Check if we've gone past the end
        if trades:
            last_ts = float(trades[-1][2])
            if last_ts > end_ts:
                break

        if time.time() - last_print > 10:
            current_date = datetime.fromtimestamp(float(trades[-1][2]) if trades else 0, tz=timezone.utc).strftime("%Y-%m-%d") if trades else "?"
            print(f"  {total_trades:,} trades, at {current_date}...")
            last_print = time.time()

        time.sleep(1.1)  # Rate limit: 1 req/sec

    # Write daily CSVs
    files_written = 0
    for date_str in sorted(daily_trades.keys()):
        out_path = asset_dir / f"{symbol}-aggTrades-{date_str}.csv"
        if out_path.exists():
            continue  # Skip existing
        trades_day = daily_trades[date_str]
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            for t in sorted(trades_day, key=lambda x: x["timestamp"]):
                writer.writerow([
                    0,  # agg_trade_id (not available from Kraken)
                    t["price"],
                    t["qty"],
                    0, 0,  # first_id, last_id (N/A)
                    t["timestamp"],
                    t["is_buyer_maker"],
                    True,  # best_price_match
                ])
        files_written += 1

    print(f"  Done: {total_trades:,} trades, {files_written} new files, {len(daily_trades)} dates")


def main():
    parser = argparse.ArgumentParser(description="Download Kraken historical trades")
    parser.add_argument("--assets", default="BTC,ETH,SOL,XRP")
    parser.add_argument("--days", type=int, default=45)
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")]
    for asset in assets:
        download_asset(asset, args.days)

    print("\nDone.")


if __name__ == "__main__":
    main()
