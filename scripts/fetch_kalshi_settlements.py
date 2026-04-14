"""Fetch historical Kalshi settlement results for all assets.

Downloads settled market results from the Kalshi API and saves them
as a JSON file for use as training labels. This replaces the Coinbase
price-based labels with Kalshi's actual CF Benchmarks settlement.

Usage:
    python scripts/fetch_kalshi_settlements.py
    python scripts/fetch_kalshi_settlements.py --assets BTC,ETH,SOL,XRP
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sdk.kalshi.client import KalshiClient, load_config

OUTPUT_DIR = PROJECT_ROOT / "data" / "kalshi_settlements"

SERIES_MAP = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
    "HYPE": "KXHYPE15M",
    "BNB": "KXBNB15M",
    "DOGE": "KXDOGE15M",
}


def fetch_all_settlements(client, series_ticker, max_pages=100):
    """Fetch all settled markets for a series, handling pagination."""
    all_markets = []
    cursor = None

    for page in range(max_pages):
        params = {
            "series_ticker": series_ticker,
            "status": "settled",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        st, data = client.request("GET", "/markets", params=params, auth=False)
        if st == 429:
            print(f"  Rate limited on page {page}, waiting 5s...")
            time.sleep(5)
            st, data = client.request("GET", "/markets", params=params, auth=False)
        if st != 200:
            print(f"  Error on page {page}: status={st}")
            break

        markets = data.get("markets", [])
        if not markets:
            break

        all_markets.extend(markets)
        cursor = data.get("cursor")
        if not cursor:
            break

        if (page + 1) % 10 == 0:
            print(f"  Page {page + 1}: {len(all_markets)} markets so far...")
        time.sleep(0.1)  # rate limit

    return all_markets


def main():
    parser = argparse.ArgumentParser(description="Fetch Kalshi historical settlements")
    parser.add_argument("--assets", default="BTC,ETH,SOL,XRP,HYPE,BNB,DOGE")
    args = parser.parse_args()

    cfg = load_config()
    client = KalshiClient(cfg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assets = [a.strip().upper() for a in args.assets.split(",")]

    for asset in assets:
        series = SERIES_MAP.get(asset)
        if not series:
            print(f"Unknown asset: {asset}")
            continue

        print(f"\n{asset} ({series}):")
        markets = fetch_all_settlements(client, series)
        print(f"  Fetched {len(markets)} settled markets")

        if not markets:
            continue

        # Extract just what we need: event_ticker -> result, close_time
        settlements = {}
        for m in markets:
            event_ticker = m.get("event_ticker", "")
            result = m.get("result", "")
            close_time = m.get("close_time", "")

            if event_ticker and result:
                # Group by event_ticker (multiple markets per event)
                if event_ticker not in settlements:
                    settlements[event_ticker] = {
                        "result": result,
                        "close_time": close_time,
                    }

        # Also build a simple lookup: close_time_utc -> result
        by_close_time = {}
        for et, info in settlements.items():
            by_close_time[info["close_time"]] = info["result"]

        # Get date range
        close_times = sorted(by_close_time.keys())
        print(f"  Unique events: {len(settlements)}")
        print(f"  Date range: {close_times[0]} to {close_times[-1]}")

        # Label balance
        n_yes = sum(1 for v in settlements.values() if v["result"] == "yes")
        n_no = len(settlements) - n_yes
        print(f"  Balance: {n_yes} yes / {n_no} no ({n_yes/len(settlements)*100:.1f}% yes)")

        # Save
        out = {
            "asset": asset,
            "series": series,
            "n_events": len(settlements),
            "date_range": [close_times[0], close_times[-1]],
            "settlements": settlements,
            "by_close_time": by_close_time,
        }

        out_path = OUTPUT_DIR / f"{asset}_settlements.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
