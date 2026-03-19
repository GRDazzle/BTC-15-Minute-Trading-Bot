"""
Realized EV analysis: cross-references signal accuracy with actual Kalshi ask prices.

Loads all Kalshi polling JSONL data, samples prices at each mins_to_close,
and computes expected value per contract at observed win rates.

Usage:
  python scripts/ev_analysis.py
  python scripts/ev_analysis.py --asset ETH
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

# Backtester decision_minute X fires at window_start + (5+X) min
# Window closes at window_start + 15 min
# So mins_to_close = 15 - (5+X) = 10 - X
def dm_to_mtc(decision_minute: int) -> int:
    return 10 - decision_minute


def load_kalshi_data(asset: str):
    """Load all poll snapshots and outcomes for an asset."""
    series = SERIES_MAP.get(asset)
    if not series:
        print(f"Unknown asset: {asset}. Available: {list(SERIES_MAP.keys())}")
        sys.exit(1)

    data_dir = KALSHI_DATA / series
    if not data_dir.exists():
        print(f"No Kalshi data at {data_dir}")
        sys.exit(1)

    # Per event: sample one price snapshot per integer mins_to_close
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
                # Only sample near whole minutes
                if abs(mtc - mtc_int) > 0.2:
                    continue

                if et not in events:
                    events[et] = {}
                # First sample per integer minute wins
                if mtc_int not in events[et]:
                    events[et][mtc_int] = {
                        "yes_ask": d["yes_ask"],
                        "no_ask": d["no_ask"],
                        "yes_bid": d["yes_bid"],
                        "no_bid": d["no_bid"],
                    }

    matched = [et for et in events if et in outcomes]
    print(f"  Events polled: {len(events)}")
    print(f"  Outcomes:      {len(outcomes)}")
    print(f"  Matched:       {len(matched)}")
    return events, outcomes, matched


def analyze(asset: str, win_rate: float):
    events, outcomes, matched = load_kalshi_data(asset)
    if not matched:
        print("No matched events to analyze.")
        return

    loss_rate = 1.0 - win_rate

    # --- Table 1: Price distribution by mins_to_close ---
    print(f"\n{'='*80}")
    print(f"  {asset} — Entry prices for the WINNING side by mins_to_close")
    print(f"  (If outcome=yes -> yes_ask. If outcome=no -> no_ask.)")
    print(f"{'='*80}")
    print(f"{'mtc':>4} | {'dm':>3} | {'avg':>6} | {'p25':>5} | {'med':>5} | {'p75':>5} | {'min':>5} | {'max':>5} | {'n':>4}")
    print("-" * 65)

    mtc_avgs = {}
    for mtc in range(10, -1, -1):
        prices = []
        for et in matched:
            if mtc not in events[et]:
                continue
            e = events[et][mtc]
            if outcomes[et] == "yes":
                prices.append(e["yes_ask"])
            else:
                prices.append(e["no_ask"])
        if not prices:
            continue

        prices.sort()
        n = len(prices)
        avg = sum(prices) / n
        mtc_avgs[mtc] = avg
        med = prices[n // 2]
        p25 = prices[n // 4]
        p75 = prices[3 * n // 4]

        dm = 10 - mtc
        dm_str = str(dm) if 0 <= dm <= 9 else "-"
        print(
            f"{mtc:4d} | {dm_str:>3} | {avg:5.1f}c | {p25:4d}c | "
            f"{med:4d}c | {p75:4d}c | {prices[0]:4d}c | {prices[-1]:4d}c | {n:4d}"
        )

    # --- Table 2: What the LOSING side costs ---
    print(f"\n{'='*80}")
    print(f"  {asset} — Entry prices for the LOSING side by mins_to_close")
    print(f"  (If outcome=yes -> no_ask was wrong. If outcome=no -> yes_ask was wrong.)")
    print(f"{'='*80}")
    print(f"{'mtc':>4} | {'dm':>3} | {'avg':>6} | {'p25':>5} | {'med':>5} | {'p75':>5} | {'n':>4}")
    print("-" * 50)

    loser_avgs = {}
    for mtc in range(10, -1, -1):
        prices = []
        for et in matched:
            if mtc not in events[et]:
                continue
            e = events[et][mtc]
            # The WRONG side: if outcome=yes, buying NO was wrong
            if outcomes[et] == "yes":
                prices.append(e["no_ask"])
            else:
                prices.append(e["yes_ask"])
        if not prices:
            continue

        prices.sort()
        n = len(prices)
        avg = sum(prices) / n
        loser_avgs[mtc] = avg
        med = prices[n // 2]
        p25 = prices[n // 4]
        p75 = prices[3 * n // 4]

        dm = 10 - mtc
        dm_str = str(dm) if 0 <= dm <= 9 else "-"
        print(
            f"{mtc:4d} | {dm_str:>3} | {avg:5.1f}c | {p25:4d}c | "
            f"{med:4d}c | {p75:4d}c | {n:4d}"
        )

    # --- Table 3: EV per contract ---
    # When we're RIGHT (89%): we paid winner_ask, profit = 100 - winner_ask
    # When we're WRONG (11%): we paid loser_ask, loss = loser_ask
    # But actually, at entry time we don't know which side wins.
    # Our signal says BULLISH/BEARISH. We always buy YES or NO at the respective ask.
    #
    # Better model: at any given mtc, the yes_ask averages X.
    # If signal=BULLISH, we buy YES at yes_ask. If BEARISH, buy NO at no_ask.
    # Since yes_ask + no_ask ≈ 100 + spread, buying the "right" side costs
    # roughly the winner_ask average, and buying the "wrong" side costs the loser_ask avg.
    #
    # EV = win_rate * (100 - avg_entry) - loss_rate * avg_entry
    # where avg_entry ≈ average of winner_ask (when we're right) and loser_ask (when wrong)
    # weighted by probability.

    print(f"\n{'='*80}")
    print(f"  {asset} — EV per contract at {win_rate:.0%} accuracy")
    print(f"  EV = {win_rate:.0%} x (100 - entry) - {loss_rate:.0%} x entry")
    print(f"{'='*80}")
    print(
        f"{'mtc':>4} | {'dm':>3} | {'win_entry':>9} | {'lose_entry':>10} | "
        f"{'win_profit':>10} | {'lose_cost':>9} | {'EV':>8} | {'verdict':>8}"
    )
    print("-" * 80)

    for mtc in range(10, -1, -1):
        if mtc not in mtc_avgs or mtc not in loser_avgs:
            continue
        win_entry = mtc_avgs[mtc]  # what we pay when correct
        lose_entry = loser_avgs[mtc]  # what we pay when wrong

        win_profit = 100 - win_entry
        lose_cost = lose_entry

        ev = win_rate * win_profit - loss_rate * lose_cost

        dm = 10 - mtc
        dm_str = str(dm) if 0 <= dm <= 9 else "-"

        if ev > 5:
            verdict = "STRONG"
        elif ev > 0:
            verdict = "+EV"
        elif ev > -2:
            verdict = "~BREAK"
        else:
            verdict = "-EV"

        print(
            f"{mtc:4d} | {dm_str:>3} | {win_entry:8.1f}c | {lose_entry:9.1f}c | "
            f"{win_profit:9.1f}c | {lose_cost:8.1f}c | {ev:+7.1f}c | {verdict:>8}"
        )

    # --- Table 4: Blended strategy summary ---
    print(f"\n{'='*80}")
    print(f"  {asset} — Strategy summary (m4+ gate, {win_rate:.0%} win rate)")
    print(f"{'='*80}")

    # Only decision_minutes 4-9 (mtc 6 down to 1)
    total_ev = 0.0
    count = 0
    for dm in range(4, 10):
        mtc = dm_to_mtc(dm)
        if mtc in mtc_avgs and mtc in loser_avgs:
            win_entry = mtc_avgs[mtc]
            lose_entry = loser_avgs[mtc]
            ev = win_rate * (100 - win_entry) - loss_rate * lose_entry
            total_ev += ev
            count += 1
            print(f"  dm={dm} (mtc={mtc}): avg entry {win_entry:.0f}c win / {lose_entry:.0f}c lose -> EV {ev:+.1f}c")

    if count:
        avg_ev = total_ev / count
        print(f"\n  Average EV across dm 4-9: {avg_ev:+.1f}c per contract")
        print(f"  Per 10 contracts/session:  {avg_ev * 10:+.1f}c")
        print(f"  Per 96 sessions/day:       ${avg_ev * 10 * 96 / 100:+.2f}")


def main():
    parser = argparse.ArgumentParser(description="Kalshi realized EV analysis")
    parser.add_argument("--asset", default="BTC", help="Asset (default: BTC)")
    parser.add_argument(
        "--win-rate", type=float, default=0.89,
        help="Assumed win rate (default: 0.89)",
    )
    args = parser.parse_args()
    analyze(args.asset.upper(), args.win_rate)


if __name__ == "__main__":
    main()
