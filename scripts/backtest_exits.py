"""Backtest exit mechanisms: Pure SL, Bracket (TP+SL), Late-window protective sell.

Usage:
    python scripts/backtest_exits.py --trades output/trades.csv --mode sl --sl 20
    python scripts/backtest_exits.py --trades output/trades.csv --mode bracket --tp 30 --sl 20
    python scripts/backtest_exits.py --trades output/trades.csv --mode late --close-mins 3 --delta 10
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
    series = f"KX{asset}15M"
    asset_dir = POLLS_DIR / series
    if not asset_dir.exists():
        return []
    we = datetime.fromisoformat(window_end_iso.replace("Z", "+00:00"))
    ws = we - timedelta(minutes=15)
    block_start_hr = (ws.hour // 4) * 4
    files = []
    for try_hr in [block_start_hr, (block_start_hr + 4) % 24]:
        date_str = ws.strftime("%Y-%m-%d")
        if try_hr == 0 and block_start_hr == 20:
            date_str = (ws + timedelta(days=1)).strftime("%Y-%m-%d")
        path = asset_dir / f"{date_str}_{try_hr:02d}00_UTC.jsonl"
        if path.exists():
            files.append(path)
    return files


def get_window_polls(asset: str, market_ticker: str, entry_ts: datetime, window_end_iso: str) -> list[dict]:
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
                if p.get("type") != "poll" or p.get("market_ticker") != market_ticker:
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
                    "no_bid": int(p.get("no_bid", 0)),
                    "mins_to_close": float(p.get("mins_to_close", 0)),
                })
    polls.sort(key=lambda x: x["ts"])
    return polls


def evaluate_mode(t, polls, mode, params, fee):
    """Return (triggered, sell_price, exit_label) for this trade under the given mode."""
    entry_price = t["entry_price"]
    direction = t["direction"]

    for p in polls:
        my_bid = p["yes_bid"] if direction == "BULLISH" else p["no_bid"]

        if mode == "sl":
            if my_bid <= entry_price - params["sl"]:
                return True, my_bid, "SL"

        elif mode == "tp":
            if my_bid >= entry_price + params["tp"]:
                return True, my_bid, "TP"

        elif mode == "bracket":
            if my_bid >= entry_price + params["tp"]:
                return True, my_bid, "TP"
            if my_bid <= entry_price - params["sl"]:
                return True, my_bid, "SL"

        elif mode == "late":
            # Late protective sell: in last N mins, if losing by delta, sell
            if p["mins_to_close"] <= params["close_mins"] and my_bid <= entry_price - params["delta"]:
                return True, my_bid, "LATE"

    return False, None, None


def backtest_one(trades_path: Path, mode: str, params: dict, max_entry: int, fee: float) -> dict:
    rows = []
    with open(trades_path) as f:
        for row in csv.DictReader(f):
            if not row.get("outcome") or not row.get("pnl"):
                continue
            try:
                rows.append({
                    "ts": datetime.fromisoformat(row["timestamp"]),
                    "asset": row["asset"],
                    "window_id": row["window_id"],
                    "market_ticker": row["market_ticker"],
                    "direction": row["direction"],
                    "entry_price": int(row["price_cents"]),
                    "contracts": int(row["contracts"]),
                    "pnl": float(row["pnl"]),
                })
            except (ValueError, KeyError):
                continue
    rows.sort(key=lambda x: x["ts"])

    stats = defaultdict(lambda: {
        "n": 0, "real_pnl": 0.0, "real_wins": 0,
        "exit_eligible": 0, "exit_triggered": 0, "exit_pnl": 0.0,
        "by_label": defaultdict(int),
    })

    for t in rows:
        s = stats[t["asset"]]
        s["n"] += 1
        s["real_pnl"] += t["pnl"]
        if t["pnl"] > 0:
            s["real_wins"] += 1

        if t["entry_price"] >= max_entry:
            s["exit_pnl"] += t["pnl"]
            continue

        s["exit_eligible"] += 1
        polls = get_window_polls(t["asset"], t["market_ticker"], t["ts"], t["window_id"])
        triggered, sell_price, label = evaluate_mode(t, polls, mode, params, fee)

        if triggered:
            s["exit_triggered"] += 1
            s["by_label"][label] += 1
            exit_pnl = t["contracts"] * (sell_price - t["entry_price"]) / 100.0 - t["contracts"] * fee / 100.0
            s["exit_pnl"] += exit_pnl
        else:
            s["exit_pnl"] += t["pnl"]

    # Aggregate
    totals = {"n": 0, "real_pnl": 0.0, "exit_pnl": 0.0, "eligible": 0, "triggered": 0,
              "by_label": defaultdict(int), "per_asset": {}}
    for asset, s in stats.items():
        totals["n"] += s["n"]
        totals["real_pnl"] += s["real_pnl"]
        totals["exit_pnl"] += s["exit_pnl"]
        totals["eligible"] += s["exit_eligible"]
        totals["triggered"] += s["exit_triggered"]
        for l, c in s["by_label"].items():
            totals["by_label"][l] += c
        totals["per_asset"][asset] = {
            "n": s["n"], "real": s["real_pnl"], "exit": s["exit_pnl"],
            "delta": s["exit_pnl"] - s["real_pnl"],
            "eligible": s["exit_eligible"], "triggered": s["exit_triggered"],
        }
    totals["delta"] = totals["exit_pnl"] - totals["real_pnl"]
    return totals


def fmt_result(label: str, totals: dict) -> str:
    return (f"{label:<35} N={totals['n']:>4}  Real ${totals['real_pnl']:>+8.2f}  "
            f"Exit ${totals['exit_pnl']:>+8.2f}  D ${totals['delta']:>+8.2f}  "
            f"trig={totals['triggered']:>3}/{totals['eligible']:>3}")


def main():
    archives = [
        ("output/trades.csv", "Today"),
        ("output/v9_archive_2026-04-10/trades.csv", "v9"),
        ("output/v10_stacked_archive_2026-04-15/trades.csv", "v10s"),
    ]
    max_entry = 80
    fee = 1.0

    results = defaultdict(dict)  # rule_label -> archive -> totals

    print("=" * 100)
    print("OPTION 1: PURE STOP LOSS")
    print("=" * 100)
    for sl in [10, 15, 20, 25]:
        for path, name in archives:
            t = backtest_one(Path(path), "sl", {"sl": sl}, max_entry, fee)
            results[f"SL=-{sl}c"][name] = t
            print(fmt_result(f"SL=-{sl}c on {name}", t))
        print()

    print("=" * 100)
    print("OPTION 2: BRACKET (TP + SL)")
    print("=" * 100)
    for tp in [20, 30]:
        for sl in [15, 20]:
            for path, name in archives:
                t = backtest_one(Path(path), "bracket", {"tp": tp, "sl": sl}, max_entry, fee)
                results[f"Bracket TP=+{tp}/SL=-{sl}"][name] = t
                tp_count = t["by_label"].get("TP", 0)
                sl_count = t["by_label"].get("SL", 0)
                print(fmt_result(f"TP=+{tp}/SL=-{sl} on {name}", t) + f" (TP={tp_count}, SL={sl_count})")
            print()

    print("=" * 100)
    print("OPTION 3: LATE-WINDOW PROTECTIVE SELL")
    print("=" * 100)
    for close_mins in [2, 3, 5]:
        for delta in [5, 10]:
            for path, name in archives:
                t = backtest_one(Path(path), "late", {"close_mins": close_mins, "delta": delta}, max_entry, fee)
                results[f"Late mins<={close_mins}, delta>={delta}c"][name] = t
                print(fmt_result(f"close<={close_mins}m d>={delta}c on {name}", t))
            print()

    # Cross-archive summary table
    print("=" * 100)
    print("CROSS-ARCHIVE SUMMARY (best deltas)")
    print("=" * 100)
    print(f"{'Rule':<35} {'Today':>10} {'v9':>10} {'v10s':>10} {'SUM':>10}")
    print("-" * 80)
    for rule_label, by_arch in results.items():
        today_d = by_arch.get("Today", {}).get("delta", 0)
        v9_d = by_arch.get("v9", {}).get("delta", 0)
        v10_d = by_arch.get("v10s", {}).get("delta", 0)
        total = today_d + v9_d + v10_d
        flag = " <==" if total > 0 else ""
        print(f"{rule_label:<35} ${today_d:>+8.2f} ${v9_d:>+8.2f} ${v10_d:>+8.2f} ${total:>+8.2f}{flag}")


if __name__ == "__main__":
    main()
