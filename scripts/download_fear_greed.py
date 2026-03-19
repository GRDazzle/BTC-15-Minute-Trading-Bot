#!/usr/bin/env python3
"""Download historical Crypto Fear & Greed Index from alternative.me.

Usage:
    python scripts/download_fear_greed.py

Output: data/historical/fear_greed.csv
Columns: timestamp, date, value, classification

One API call — returns all history (daily granularity since 2018).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import requests

API_URL = "https://api.alternative.me/fng/"


def download_fear_greed(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fear_greed.csv"

    print("Downloading Fear & Greed Index (all history)...")
    resp = requests.get(API_URL, params={"limit": 0, "format": "json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    if not data:
        print("  No data returned!")
        return out_path

    # Sort oldest first
    data.sort(key=lambda d: int(d["timestamp"]))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "date", "value", "classification"])
        for entry in data:
            ts = int(entry["timestamp"])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            writer.writerow([
                dt.isoformat(),
                dt.strftime("%Y-%m-%d"),
                entry["value"],
                entry["value_classification"],
            ])

    print(f"  Saved {len(data)} days -> {out_path}")
    first_dt = datetime.fromtimestamp(int(data[0]["timestamp"]), tz=timezone.utc)
    last_dt = datetime.fromtimestamp(int(data[-1]["timestamp"]), tz=timezone.utc)
    print(f"  Range: {first_dt.date()} -> {last_dt.date()}")
    return out_path


def main():
    output_dir = Path("data/historical")
    print("Fear & Greed Index Downloader")
    print(f"  Output: {output_dir}/")
    print()
    download_fear_greed(output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
