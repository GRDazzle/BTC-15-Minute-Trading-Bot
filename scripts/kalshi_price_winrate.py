"""Kalshi market price as predictor: when one side is priced >Xc, how often does it win?

Analyzes the non-trading window (mtc 10-7, i.e. before the m4+ gate) to see
if Kalshi's own market pricing is a reliable signal.

Usage:
  python scripts/kalshi_price_winrate.py
  python scripts/kalshi_price_winrate.py --asset ETH --threshold 65
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

KALSHI_DATA = Path("G:/workspace/BuzzTheGambler/Kalshi/15-min/TradingBot/data")

SERIES_MAP = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
}


def load_data(asset: str):
    """Load poll snapshots and outcomes."""
    series = SERIES_MAP.get(asset)
    if not series:
        print(f"Unknown asset: {asset}")
        sys.exit(1)

    data_dir = KALSHI_DATA / series
    if not data_dir.exists():
        print(f"No data at {data_dir}")
        sys.exit(1)

    events: dict[str, dict[int, dict]] = {}
    outcomes: dict[str, str] = {}

    files = sorted(data_dir.glob("*.jsonl"))
    print(f"Loading {len(files)} JSONL files from {data_dir.name}...")

    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                et = d["event_ticker"]

                if d["type"] == "outcome":
                    outcomes[et] = d["outcome"]
                    continue
                if d["type"] != "poll":
                    continue

                mtc = d["mins_to_close"]
                mtc_int = round(mtc)
                if mtc_int < 0 or mtc_int > 14:
                    continue
                if abs(mtc - mtc_int) > 0.2:
                    continue

                if et not in events:
                    events[et] = {}
                if mtc_int not in events[et]:
                    events[et][mtc_int] = {
                        "yes_ask": d["yes_ask"],
                        "no_ask": d["no_ask"],
                        "yes_bid": d.get("yes_bid"),
                        "no_bid": d.get("no_bid"),
                    }

    matched = [et for et in events if et in outcomes]
    print(f"  Events: {len(events)}  Outcomes: {len(outcomes)}  Matched: {len(matched)}")
    return events, outcomes, matched


def analyze(asset: str, threshold: int):
    events, outcomes, matched = load_data(asset)
    if not matched:
        print("No matched events.")
        return

    # Analyze each mtc
    print(f"\n{'='*85}")
    print(f"  {asset} -- When one side is priced >= {threshold}c, does it win?")
    print(f"{'='*85}")
    print(
        f"{'mtc':>4} | {'dm':>3} | {'yes>={t}c':>10} | {'yes_win':>8} | {'yes_%':>6} | "
        f"{'no>={t}c':>10} | {'no_win':>8} | {'no_%':>6} | {'total':>6} | {'overall_%':>9}".format(t=threshold)
    )
    print("-" * 85)

    # Also track by threshold buckets
    bucket_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "wins": 0})

    for mtc in range(14, -1, -1):
        yes_total = 0
        yes_wins = 0
        no_total = 0
        no_wins = 0

        for et in matched:
            if mtc not in events[et]:
                continue
            e = events[et][mtc]
            outcome = outcomes[et]

            # Check YES side >= threshold
            if e["yes_ask"] is not None and e["yes_ask"] >= threshold:
                yes_total += 1
                if outcome == "yes":
                    yes_wins += 1

            # Check NO side >= threshold
            if e["no_ask"] is not None and e["no_ask"] >= threshold:
                no_total += 1
                if outcome == "no":
                    no_wins += 1

        total = yes_total + no_total
        wins = yes_wins + no_wins
        if total == 0:
            continue

        dm = 10 - mtc
        dm_str = str(dm) if 0 <= dm <= 9 else "-"
        yes_pct = f"{yes_wins/yes_total*100:.1f}%" if yes_total else "-"
        no_pct = f"{no_wins/no_total*100:.1f}%" if no_total else "-"
        overall_pct = f"{wins/total*100:.1f}%" if total else "-"

        gate = " <-- gate" if 4 <= dm <= 9 else ""
        print(
            f"{mtc:4d} | {dm_str:>3} | {yes_total:10d} | {yes_wins:8d} | {yes_pct:>6} | "
            f"{no_total:10d} | {no_wins:8d} | {no_pct:>6} | {total:6d} | {overall_pct:>9}{gate}"
        )

    # --- Threshold sweep ---
    print(f"\n{'='*85}")
    print(f"  {asset} -- Win rate by price threshold (all mtc combined)")
    print(f"{'='*85}")
    print(f"{'threshold':>10} | {'total':>6} | {'wins':>6} | {'win%':>6} | {'coverage':>8}")
    print("-" * 50)

    total_events = len(matched)
    for thresh in [50, 55, 60, 65, 70, 75, 80, 85, 90]:
        total = 0
        wins = 0
        events_with_signal = set()

        for et in matched:
            for mtc in range(7, 11):  # non-trading window: mtc 7-10 (dm 0-3)
                if mtc not in events[et]:
                    continue
                e = events[et][mtc]
                outcome = outcomes[et]

                if e["yes_ask"] is not None and e["yes_ask"] >= thresh:
                    total += 1
                    events_with_signal.add(et)
                    if outcome == "yes":
                        wins += 1

                if e["no_ask"] is not None and e["no_ask"] >= thresh:
                    total += 1
                    events_with_signal.add(et)
                    if outcome == "no":
                        wins += 1

        if total == 0:
            continue
        coverage = len(events_with_signal) / total_events * 100 if total_events else 0
        print(
            f"{thresh:>9}c | {total:6d} | {wins:6d} | {wins/total*100:5.1f}% | {coverage:6.1f}%"
        )

    # --- Non-trading window summary ---
    print(f"\n{'='*85}")
    print(f"  {asset} -- Non-trading window (mtc 7-10 / dm 0-3) with >={threshold}c")
    print(f"{'='*85}")

    total = 0
    wins = 0
    for et in matched:
        for mtc in range(7, 11):
            if mtc not in events[et]:
                continue
            e = events[et][mtc]
            outcome = outcomes[et]

            if e["yes_ask"] is not None and e["yes_ask"] >= threshold:
                total += 1
                if outcome == "yes":
                    wins += 1
            if e["no_ask"] is not None and e["no_ask"] >= threshold:
                total += 1
                if outcome == "no":
                    wins += 1

    if total:
        print(f"  Samples: {total}")
        print(f"  Wins:    {wins} ({wins/total*100:.1f}%)")
        print(f"  Losses:  {total-wins} ({(total-wins)/total*100:.1f}%)")
    else:
        print("  No samples found.")


def main():
    parser = argparse.ArgumentParser(description="Kalshi price-as-predictor analysis")
    parser.add_argument("--asset", default="BTC", help="Asset (default: BTC)")
    parser.add_argument("--threshold", type=int, default=70, help="Price threshold in cents (default: 70)")
    args = parser.parse_args()
    analyze(args.asset.upper(), args.threshold)


if __name__ == "__main__":
    main()
