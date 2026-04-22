"""Backtest a take-profit (TP) rule against historical trade sessions.

Rule: For each filled trade with entry_price < `min_entry`,
walk through Kalshi polls from entry to settlement.
  - If BULLISH (bought YES at entry): TP when yes_bid >= entry + tp_offset
  - If BEARISH (bought NO at entry):  TP when no_bid  >= entry + tp_offset
TP'd trades realize (tp_price - entry) * contracts profit.
Other trades use actual settlement outcome from trades.csv.

Usage:
    python scripts/backtest_tp.py --trades output/trades.csv --tp 20 --max-entry 80
    python scripts/backtest_tp.py --trades output/v9_archive_2026-04-10/trades.csv
"""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLLS_DIR = PROJECT_ROOT / "data" / "kalshi_polls"


def poll_files_for_window(asset: str, window_end_iso: str) -> list[Path]:
    """Return the JSONL files that could contain polls for this window."""
    series = f"KX{asset}15M"
    asset_dir = POLLS_DIR / series
    if not asset_dir.exists():
        return []
    we = datetime.fromisoformat(window_end_iso.replace("Z", "+00:00"))
    # Window starts 15 min earlier
    ws = we - timedelta(minutes=15)
    # Files cover 4-hour blocks starting at 0000, 0400, 0800, ...
    block_start_hr = (ws.hour // 4) * 4
    files = []
    for try_hr in [block_start_hr, (block_start_hr + 4) % 24]:
        date_str = ws.strftime("%Y-%m-%d")
        # Handle day boundary
        if try_hr == 0 and block_start_hr == 20:
            date_str = (ws + timedelta(days=1)).strftime("%Y-%m-%d")
        fname = f"{date_str}_{try_hr:02d}00_UTC.jsonl"
        path = asset_dir / fname
        if path.exists():
            files.append(path)
    return files


def get_window_polls(asset: str, market_ticker: str, entry_ts: datetime, window_end_iso: str) -> list[dict]:
    """Return polls for this market ticker between entry_ts and window end, sorted by ts."""
    we = datetime.fromisoformat(window_end_iso.replace("Z", "+00:00"))
    files = poll_files_for_window(asset, window_end_iso)
    polls = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if p.get("type") != "poll":
                    continue
                if p.get("market_ticker") != market_ticker:
                    continue
                try:
                    pts = datetime.fromisoformat(p["ts"].replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    continue
                if pts < entry_ts or pts > we:
                    continue
                polls.append({
                    "ts": pts,
                    "yes_bid": int(p.get("yes_bid", 0)),
                    "yes_ask": int(p.get("yes_ask", 100)),
                    "no_bid": int(p.get("no_bid", 0)),
                    "no_ask": int(p.get("no_ask", 100)),
                })
    polls.sort(key=lambda x: x["ts"])
    return polls


def backtest(trades_path: Path, tp_offset: int, max_entry: int, fee_per_contract: float) -> None:
    rows = []
    with open(trades_path) as f:
        for row in csv.DictReader(f):
            if not row.get("outcome") or not row.get("pnl"):
                continue
            try:
                entry_ts = datetime.fromisoformat(row["timestamp"])
                rows.append({
                    "ts": entry_ts,
                    "asset": row["asset"],
                    "window_id": row["window_id"],
                    "market_ticker": row["market_ticker"],
                    "direction": row["direction"],
                    "entry_price": int(row["price_cents"]),
                    "contracts": int(row["contracts"]),
                    "cost": float(row["cost"]),
                    "outcome": row["outcome"],
                    "pnl": float(row["pnl"]),
                })
            except (ValueError, KeyError):
                continue
    print(f"Loaded {len(rows)} settled trades from {trades_path}")
    if not rows:
        return

    print(f"Backtest: TP=+{tp_offset}c when entry<{max_entry}c (fee={fee_per_contract}c/contract)")
    print()

    # Stats per asset
    stats = defaultdict(lambda: {
        "n": 0, "real_pnl": 0.0, "real_wins": 0,
        "tp_eligible": 0, "tp_triggered": 0, "tp_pnl": 0.0, "tp_wins": 0,
        "tp_avoided_loss": 0.0, "tp_lost_winnings": 0.0,
    })

    poll_cache: dict[str, list] = {}
    for i, t in enumerate(rows):
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(rows)} processed...")
        s = stats[t["asset"]]
        s["n"] += 1
        s["real_pnl"] += t["pnl"]
        if t["pnl"] > 0:
            s["real_wins"] += 1

        # TP eligibility
        if t["entry_price"] >= max_entry:
            # Can't TP — would need price > 100c which is impossible
            s["tp_pnl"] += t["pnl"]
            if t["pnl"] > 0:
                s["tp_wins"] += 1
            continue
        s["tp_eligible"] += 1

        # Find polls for this window after entry
        cache_key = f"{t['market_ticker']}_{t['window_id']}"
        if cache_key in poll_cache:
            polls = poll_cache[cache_key]
        else:
            polls = get_window_polls(t["asset"], t["market_ticker"], t["ts"], t["window_id"])
            poll_cache[cache_key] = polls

        # Look for TP trigger
        target = t["entry_price"] + tp_offset
        triggered = False
        for p in polls:
            if t["direction"] == "BULLISH":
                if p["yes_bid"] >= target:
                    triggered = True
                    sell_price = p["yes_bid"]
                    break
            else:  # BEARISH
                if p["no_bid"] >= target:
                    triggered = True
                    sell_price = p["no_bid"]
                    break

        if triggered:
            s["tp_triggered"] += 1
            tp_pnl = t["contracts"] * (sell_price - t["entry_price"]) / 100.0 - t["contracts"] * fee_per_contract / 100.0
            s["tp_pnl"] += tp_pnl
            s["tp_wins"] += 1  # TP always wins (locks in profit)
            if t["pnl"] > 0:
                # Original would have won; we may have given up some profit
                s["tp_lost_winnings"] += max(0, t["pnl"] - tp_pnl)
            else:
                # Original would have lost; TP saved us
                s["tp_avoided_loss"] += abs(t["pnl"]) + tp_pnl
        else:
            # No TP triggered — use actual outcome
            s["tp_pnl"] += t["pnl"]
            if t["pnl"] > 0:
                s["tp_wins"] += 1

    # Print summary
    print()
    print(f"{'Asset':<6} {'N':>4} {'Real WR':>7} {'Real PnL':>9} | {'TP-elig':>7} {'TP-fired':>8} {'TP WR':>6} {'TP PnL':>9} {'Delta':>9}")
    print("-" * 100)
    total_real = total_tp = 0.0
    total_n = total_real_wins = total_tp_wins = total_eligible = total_fired = 0
    total_avoided = total_given_up = 0.0
    for asset in sorted(stats):
        s = stats[asset]
        if s["n"] == 0:
            continue
        real_wr = 100 * s["real_wins"] / s["n"]
        tp_wr = 100 * s["tp_wins"] / s["n"]
        delta = s["tp_pnl"] - s["real_pnl"]
        total_real += s["real_pnl"]
        total_tp += s["tp_pnl"]
        total_n += s["n"]
        total_real_wins += s["real_wins"]
        total_tp_wins += s["tp_wins"]
        total_eligible += s["tp_eligible"]
        total_fired += s["tp_triggered"]
        total_avoided += s["tp_avoided_loss"]
        total_given_up += s["tp_lost_winnings"]
        print(f"{asset:<6} {s['n']:>4} {real_wr:>6.1f}% ${s['real_pnl']:>+8.2f} | {s['tp_eligible']:>7} {s['tp_triggered']:>8} {tp_wr:>5.1f}% ${s['tp_pnl']:>+8.2f} ${delta:>+8.2f}")
    print("-" * 100)
    real_wr = 100 * total_real_wins / total_n if total_n else 0
    tp_wr = 100 * total_tp_wins / total_n if total_n else 0
    delta = total_tp - total_real
    print(f"{'TOTAL':<6} {total_n:>4} {real_wr:>6.1f}% ${total_real:>+8.2f} | {total_eligible:>7} {total_fired:>8} {tp_wr:>5.1f}% ${total_tp:>+8.2f} ${delta:>+8.2f}")
    print()
    print(f"TP-eligible trades: {total_eligible} (entry < {max_entry}c)")
    print(f"TP triggered:       {total_fired} ({100*total_fired/total_eligible:.1f}% of eligible)" if total_eligible else "")
    print(f"Loss avoided by TP: ${total_avoided:+.2f}")
    print(f"Winnings given up:  ${total_given_up:+.2f}")
    print(f"Net TP impact:      ${delta:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=Path, default=PROJECT_ROOT / "output" / "trades.csv")
    ap.add_argument("--tp", type=int, default=20, help="TP offset in cents (default: 20)")
    ap.add_argument("--max-entry", type=int, default=80, help="Only TP trades with entry < this (default: 80)")
    ap.add_argument("--fee", type=float, default=1.0, help="Fee per contract on early sell, in cents (default: 1.0)")
    args = ap.parse_args()
    backtest(args.trades, args.tp, args.max_entry, args.fee)


if __name__ == "__main__":
    main()
