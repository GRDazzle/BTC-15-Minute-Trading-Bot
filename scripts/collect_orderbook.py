"""Collect L2 order book depth snapshots from Coinbase + Kraken.

Polls REST endpoints every 10 seconds, saves daily CSVs with
depth summary stats. Run alongside the bot as a separate process.

Usage:
    python scripts/collect_orderbook.py --assets BTC,ETH,SOL,XRP
    python scripts/collect_orderbook.py --assets BTC,ETH,SOL,XRP,HYPE,BNB,DOGE --interval 10

Output:
    data/orderbook/{ASSET}/orderbook-{YYYY-MM-DD}.csv
"""
import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "orderbook"

COINBASE_PAIRS = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
    "XRP": "XRP-USD", "HYPE": "HYPE-USD", "BNB": "BNB-USD", "DOGE": "DOGE-USD",
}
KRAKEN_PAIRS = {
    "BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD",
    "XRP": "XRPUSD", "HYPE": "HYPEUSD", "BNB": "BNBUSD",
    # DOGE not on Kraken
}

CSV_FIELDS = [
    "timestamp", "asset", "exchange", "mid_price",
    "bid_depth_0.5pct", "ask_depth_0.5pct",
    "bid_depth_1.0pct", "ask_depth_1.0pct",
    "bid_depth_2.0pct", "ask_depth_2.0pct",
    "depth_imbalance_1pct", "spread_bps",
    "n_bid_levels", "n_ask_levels",
    "best_bid", "best_ask",
]


def fetch_coinbase_book(pair: str) -> dict | None:
    """Fetch Coinbase L2 order book (top 50 levels)."""
    url = f"https://api.exchange.coinbase.com/products/{pair}/book?level=2"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def fetch_kraken_book(pair: str) -> dict | None:
    """Fetch Kraken order book (top 100 levels)."""
    url = f"https://api.kraken.com/0/public/Depth?pair={pair}&count=100"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        if data.get("error"):
            return None
        result = data.get("result", {})
        # Kraken returns {pair_key: {bids: [...], asks: [...]}}
        for key in result:
            return result[key]
    except Exception:
        return None


def compute_depth_stats(bids: list, asks: list) -> dict:
    """Compute depth summary stats from bid/ask arrays.

    bids/asks: list of [price_str, size_str, ...] sorted by price
    """
    if not bids or not asks:
        return {}

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    if mid == 0:
        return {}

    spread_bps = (best_ask - best_bid) / mid * 10000

    # Compute depth at various % levels from mid
    stats = {"mid_price": mid, "spread_bps": round(spread_bps, 1),
             "best_bid": best_bid, "best_ask": best_ask,
             "n_bid_levels": len(bids), "n_ask_levels": len(asks)}

    for pct in [0.5, 1.0, 2.0]:
        bid_threshold = mid * (1 - pct / 100)
        ask_threshold = mid * (1 + pct / 100)

        bid_depth = sum(float(b[1]) for b in bids if float(b[0]) >= bid_threshold)
        ask_depth = sum(float(a[1]) for a in asks if float(a[0]) <= ask_threshold)

        pct_label = f"{pct}pct"  # "0.5pct", "1.0pct", "2.0pct"
        stats[f"bid_depth_{pct_label}"] = round(bid_depth, 4)
        stats[f"ask_depth_{pct_label}"] = round(ask_depth, 4)

    # Depth imbalance at 1%
    bd1 = stats.get("bid_depth_1.0pct", 0)
    ad1 = stats.get("ask_depth_1.0pct", 0)
    total = bd1 + ad1
    stats["depth_imbalance_1pct"] = round((bd1 - ad1) / total, 4) if total > 0 else 0.0

    return stats


def get_csv_writer(asset: str, date_str: str) -> tuple:
    """Get or create CSV writer for this asset/date."""
    asset_dir = OUTPUT_DIR / asset
    asset_dir.mkdir(parents=True, exist_ok=True)
    path = asset_dir / f"orderbook-{date_str}.csv"
    is_new = not path.exists()
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if is_new:
        writer.writeheader()
    return fh, writer


def main():
    ap = argparse.ArgumentParser(description="Collect L2 order book snapshots")
    ap.add_argument("--assets", default="BTC,ETH,SOL,XRP", help="Assets to collect")
    ap.add_argument("--interval", type=int, default=10, help="Seconds between snapshots")
    args = ap.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")]
    interval = args.interval

    print(f"Collecting L2 order book: {assets} every {interval}s")
    print(f"Output: {OUTPUT_DIR}/{{ASSET}}/orderbook-YYYY-MM-DD.csv")
    print(f"Press Ctrl+C to stop\n")

    open_files: dict[str, tuple] = {}
    current_date = ""

    try:
        while True:
            now = datetime.now(timezone.utc)
            date_str = now.strftime("%Y-%m-%d")

            # Rotate files on day change
            if date_str != current_date:
                for fh, _ in open_files.values():
                    fh.close()
                open_files.clear()
                current_date = date_str

            ts_iso = now.isoformat()

            for asset in assets:
                # Coinbase
                cb_pair = COINBASE_PAIRS.get(asset)
                if cb_pair:
                    book = fetch_coinbase_book(cb_pair)
                    if book and book.get("bids") and book.get("asks"):
                        stats = compute_depth_stats(book["bids"], book["asks"])
                        if stats:
                            key = f"{asset}/{date_str}"
                            if key not in open_files:
                                open_files[key] = get_csv_writer(asset, date_str)
                            _, writer = open_files[key]
                            row = {"timestamp": ts_iso, "asset": asset, "exchange": "coinbase"}
                            for field in CSV_FIELDS:
                                if field not in row:
                                    row[field] = stats.get(field, "")
                            writer.writerow(row)
                            open_files[key][0].flush()

                # Kraken
                kr_pair = KRAKEN_PAIRS.get(asset)
                if kr_pair:
                    book = fetch_kraken_book(kr_pair)
                    if book and book.get("bids") and book.get("asks"):
                        stats = compute_depth_stats(book["bids"], book["asks"])
                        if stats:
                            key = f"{asset}/{date_str}"
                            if key not in open_files:
                                open_files[key] = get_csv_writer(asset, date_str)
                            _, writer = open_files[key]
                            row = {"timestamp": ts_iso, "asset": asset, "exchange": "kraken"}
                            for field in CSV_FIELDS:
                                if field not in row:
                                    row[field] = stats.get(field, "")
                            writer.writerow(row)
                            open_files[key][0].flush()

                # Small delay between assets to avoid rate limits
                time.sleep(0.2)

            # Log progress every 5 min
            if int(now.timestamp()) % 300 < interval:
                sizes = {}
                for asset in assets:
                    p = OUTPUT_DIR / asset / f"orderbook-{date_str}.csv"
                    if p.exists():
                        sizes[asset] = p.stat().st_size
                size_str = ", ".join(f"{a}:{s//1024}KB" for a, s in sizes.items())
                print(f"[{now.strftime('%H:%M:%S')}] Collecting... {size_str}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopping collector...")
    finally:
        for fh, _ in open_files.values():
            fh.close()
        print("Done. Files saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
