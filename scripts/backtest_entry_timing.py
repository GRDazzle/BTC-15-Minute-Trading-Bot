"""Backtest: does waiting for a better price within a window improve PnL?

For each window where we traded, look at ALL Kalshi prices from min_dm to close.
Compare: actual entry vs best-available entry vs latest-available entry.

Usage:
    python scripts/backtest_entry_timing.py --trades output/v11_archive_2026-04-17/trades.csv
"""
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLLS_DIR = PROJECT_ROOT / "data" / "kalshi_polls"


def get_window_polls(asset, market_ticker, window_start, window_end):
    series = f"KX{asset}15M"
    asset_dir = POLLS_DIR / series
    if not asset_dir.exists():
        return []
    ws = datetime.fromisoformat(window_start.replace("Z", "+00:00")) if isinstance(window_start, str) else window_start
    we = datetime.fromisoformat(window_end.replace("Z", "+00:00")) if isinstance(window_end, str) else window_end
    block_hr = (ws.hour // 4) * 4
    polls = []
    for try_hr in [block_hr, (block_hr + 4) % 24]:
        date_str = ws.strftime("%Y-%m-%d")
        if try_hr == 0 and block_hr == 20:
            date_str = (ws + timedelta(days=1)).strftime("%Y-%m-%d")
        path = asset_dir / f"{date_str}_{try_hr:02d}00_UTC.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if p.get("type") != "poll" or p.get("market_ticker") != market_ticker:
                    continue
                try:
                    pts = datetime.fromisoformat(p["ts"].replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    continue
                if pts < ws or pts > we:
                    continue
                polls.append({
                    "ts": pts,
                    "yes_ask": int(p.get("yes_ask", 100)),
                    "no_ask": int(p.get("no_ask", 100)),
                    "yes_bid": int(p.get("yes_bid", 0)),
                    "no_bid": int(p.get("no_bid", 0)),
                    "mins_to_close": float(p.get("mins_to_close", 0)),
                })
    polls.sort(key=lambda x: x["ts"])
    return polls


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=Path, default=PROJECT_ROOT / "output" / "trades.csv")
    args = ap.parse_args()

    rows = []
    with open(args.trades) as f:
        for row in csv.DictReader(f):
            if not row.get("outcome") or not row.get("pnl"):
                continue
            rows.append(row)
    print(f"Loaded {len(rows)} settled trades")

    stats = defaultdict(lambda: {
        "n": 0, "actual_pnl": 0.0, "best_pnl": 0.0, "worst_pnl": 0.0,
        "avg_actual_price": 0.0, "avg_best_price": 0.0, "avg_worst_price": 0.0,
        "price_savings": [],
    })

    for t in rows:
        asset = t["asset"]
        direction = t["direction"]
        entry_price = int(t["price_cents"])
        contracts = int(t["contracts"])
        actual_pnl = float(t["pnl"])
        outcome = t["outcome"]
        entry_ts = datetime.fromisoformat(t["timestamp"])
        window_end = t["window_id"]
        we = datetime.fromisoformat(window_end.replace("Z", "+00:00"))
        ws = we - timedelta(minutes=15)

        # Get all polls from entry to window end
        polls = get_window_polls(asset, t["market_ticker"], entry_ts, we)
        if not polls:
            continue

        # Find best and worst available prices AFTER our entry
        if direction == "BULLISH":
            # We want lowest yes_ask (cheapest entry)
            prices = [p["yes_ask"] for p in polls if p["yes_ask"] > 0]
        else:
            # We want lowest no_ask (cheapest entry)
            prices = [p["no_ask"] for p in polls if p["no_ask"] > 0]

        if not prices:
            continue

        best_price = min(prices)  # cheapest we could have entered
        worst_price = max(prices)  # most expensive
        # Last available price (entering near close)
        last_price = prices[-1]

        # Compute hypothetical PnL at best price
        won = (direction == "BULLISH" and outcome == "yes") or (direction == "BEARISH" and outcome == "no")
        fee = 0.02

        def calc_pnl(price, won, contracts):
            cost = (price / 100.0 + fee) * contracts
            return (contracts * 1.0 - cost) if won else -cost

        s = stats[asset]
        s["n"] += 1
        s["actual_pnl"] += actual_pnl
        s["best_pnl"] += calc_pnl(best_price, won, contracts)
        s["worst_pnl"] += calc_pnl(worst_price, won, contracts)
        s["avg_actual_price"] += entry_price
        s["avg_best_price"] += best_price
        s["avg_worst_price"] += worst_price
        s["price_savings"].append(entry_price - best_price)

    # Report
    print()
    print(f"{'Asset':<6} {'N':>4} {'Actual PnL':>11} {'Best PnL':>11} {'D':>9} {'Avg entry':>10} {'Avg best':>10} {'Avg save':>9}")
    print("-" * 80)
    total_actual = total_best = 0.0
    total_n = 0
    for asset in sorted(stats):
        s = stats[asset]
        if s["n"] == 0:
            continue
        avg_entry = s["avg_actual_price"] / s["n"]
        avg_best = s["avg_best_price"] / s["n"]
        avg_save = sum(s["price_savings"]) / s["n"]
        delta = s["best_pnl"] - s["actual_pnl"]
        total_actual += s["actual_pnl"]
        total_best += s["best_pnl"]
        total_n += s["n"]
        print(f"{asset:<6} {s['n']:>4} ${s['actual_pnl']:>+10.2f} ${s['best_pnl']:>+10.2f} ${delta:>+8.2f} {avg_entry:>9.1f}c {avg_best:>9.1f}c {avg_save:>8.1f}c")
    print("-" * 80)
    delta = total_best - total_actual
    print(f"{'TOTAL':<6} {total_n:>4} ${total_actual:>+10.2f} ${total_best:>+10.2f} ${delta:>+8.2f}")
    print()
    print(f"If we could perfectly time entry to the best price in each window:")
    print(f"  Extra PnL: ${delta:+.2f} over {total_n} trades")
    print(f"  Per trade: ${delta/total_n:+.3f}" if total_n else "")

    # Distribution of savings
    all_saves = []
    for s in stats.values():
        all_saves.extend(s["price_savings"])
    if all_saves:
        all_saves.sort()
        print(f"\nPrice savings distribution (actual_entry - best_available):")
        print(f"  0c (entered at best):  {sum(1 for x in all_saves if x == 0)} trades ({100*sum(1 for x in all_saves if x==0)/len(all_saves):.0f}%)")
        for thresh in [1, 2, 3, 5, 10]:
            n = sum(1 for x in all_saves if x >= thresh)
            print(f"  >={thresh}c savings possible: {n} trades ({100*n/len(all_saves):.0f}%)")
        print(f"  Median saving:         {all_saves[len(all_saves)//2]}c")
        print(f"  Mean saving:           {sum(all_saves)/len(all_saves):.1f}c")


if __name__ == "__main__":
    main()
