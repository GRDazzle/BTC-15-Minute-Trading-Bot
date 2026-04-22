"""Conditional TP backtest: A (per-asset session PnL gate), G (signal-flip), A+G.

Rules:
  A: Enable TP only on assets with negative running session PnL (UTC-day reset)
  G: TP when model signal direction flips opposite to entry, anywhere mid-window
  A+G: both must hold

Usage:
    python scripts/backtest_tp_conditional.py --trades output/trades.csv --signals output/signal_log.csv --tp 30 --rule A
    python scripts/backtest_tp_conditional.py --trades output/v9_archive_2026-04-10/trades.csv --signals output/v9_archive_2026-04-10/signal_log.csv --tp 30 --rule G
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
        fname = f"{date_str}_{try_hr:02d}00_UTC.jsonl"
        path = asset_dir / fname
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
                    "yes_ask": int(p.get("yes_ask", 100)),
                    "no_bid": int(p.get("no_bid", 0)),
                    "no_ask": int(p.get("no_ask", 100)),
                })
    polls.sort(key=lambda x: x["ts"])
    return polls


def load_signals_index(signals_path: Path) -> dict:
    """Index signals by (asset, window_id) -> sorted list of (ts, direction, ensemble_p)."""
    if not signals_path.exists():
        return {}
    idx = defaultdict(list)
    with open(signals_path) as f:
        for row in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(row["timestamp"])
                ens_p = float(row.get("ensemble_p", 0.5))
                direction = row.get("direction", "").upper()
                if direction not in ("BULLISH", "BEARISH"):
                    continue
                idx[(row["asset"], row["window_id"])].append({
                    "ts": ts, "direction": direction, "ensemble_p": ens_p,
                })
            except (ValueError, KeyError):
                continue
    for k in idx:
        idx[k].sort(key=lambda x: x["ts"])
    return idx


def find_flip_time(signals: list[dict], entry_ts: datetime, entry_dir: str) -> datetime | None:
    """Return the timestamp of the first signal AFTER entry that flips direction."""
    for s in signals:
        if s["ts"] <= entry_ts:
            continue
        if s["direction"] != entry_dir:
            return s["ts"]
    return None


def backtest(trades_path: Path, signals_path: Path, tp_offset: int, max_entry: int,
             fee: float, rule: str) -> None:
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
                    "pnl": float(row["pnl"]),
                })
            except (ValueError, KeyError):
                continue
    rows.sort(key=lambda x: x["ts"])  # chronological
    print(f"Loaded {len(rows)} settled trades, rule={rule}, TP=+{tp_offset}c, max_entry={max_entry}c")

    signals_idx = load_signals_index(signals_path)
    print(f"Loaded signals for {len(signals_idx)} (asset, window) keys")
    print()

    # Per-asset session PnL state (resets per UTC day)
    asset_day_pnl: dict = defaultdict(float)  # (asset, date) -> pnl

    # Stats per asset
    stats = defaultdict(lambda: {
        "n": 0, "real_pnl": 0.0, "real_wins": 0,
        "tp_eligible": 0, "tp_triggered": 0, "tp_pnl": 0.0, "tp_wins": 0,
        "tp_avoided_loss": 0.0, "tp_lost_winnings": 0.0,
    })

    for t in rows:
        s = stats[t["asset"]]
        s["n"] += 1
        s["real_pnl"] += t["pnl"]
        if t["pnl"] > 0:
            s["real_wins"] += 1

        date_key = (t["asset"], t["ts"].date())
        asset_session_pnl_pre_trade = asset_day_pnl[date_key]

        # Update session PnL with this trade's outcome (for next trade's gate)
        # But the gate uses PRE-trade session PnL, since that's what we'd know
        # at the moment of decision.

        # Eligibility check
        if t["entry_price"] >= max_entry:
            s["tp_pnl"] += t["pnl"]
            if t["pnl"] > 0:
                s["tp_wins"] += 1
            asset_day_pnl[date_key] += t["pnl"]
            continue

        # Rule A: per-asset session PnL must be negative
        rule_A_ok = asset_session_pnl_pre_trade < 0
        # Rule G: signal must flip opposite to entry at some point in window
        flip_time = None
        if rule in ("G", "A+G"):
            # signal_log uses YYYYMMDD_HHMM (window START), trades.csv uses ISO (window END)
            we = datetime.fromisoformat(t["window_id"].replace("Z", "+00:00"))
            ws_key = (we - timedelta(minutes=15)).strftime("%Y%m%d_%H%M")
            sigs = signals_idx.get((t["asset"], ws_key), [])
            flip_time = find_flip_time(sigs, t["ts"], t["direction"])
            rule_G_ok = flip_time is not None
        else:
            rule_G_ok = True

        if rule == "A":
            eligible = rule_A_ok
        elif rule == "G":
            eligible = rule_G_ok
        elif rule == "A+G":
            eligible = rule_A_ok and rule_G_ok
        elif rule == "unconditional":
            eligible = True
        else:
            raise ValueError(f"Unknown rule: {rule}")

        if not eligible:
            s["tp_pnl"] += t["pnl"]
            if t["pnl"] > 0:
                s["tp_wins"] += 1
            asset_day_pnl[date_key] += t["pnl"]
            continue

        s["tp_eligible"] += 1

        polls = get_window_polls(t["asset"], t["market_ticker"], t["ts"], t["window_id"])
        # For G/A+G, only consider polls AFTER the flip time
        if flip_time is not None:
            polls = [p for p in polls if p["ts"] >= flip_time]

        target = t["entry_price"] + tp_offset
        triggered = False
        for p in polls:
            if t["direction"] == "BULLISH":
                if p["yes_bid"] >= target:
                    triggered = True
                    sell_price = p["yes_bid"]
                    break
            else:
                if p["no_bid"] >= target:
                    triggered = True
                    sell_price = p["no_bid"]
                    break

        if triggered:
            s["tp_triggered"] += 1
            tp_pnl = t["contracts"] * (sell_price - t["entry_price"]) / 100.0 - t["contracts"] * fee / 100.0
            s["tp_pnl"] += tp_pnl
            s["tp_wins"] += 1
            if t["pnl"] > 0:
                s["tp_lost_winnings"] += max(0, t["pnl"] - tp_pnl)
            else:
                s["tp_avoided_loss"] += abs(t["pnl"]) + tp_pnl
            asset_day_pnl[date_key] += tp_pnl
        else:
            s["tp_pnl"] += t["pnl"]
            if t["pnl"] > 0:
                s["tp_wins"] += 1
            asset_day_pnl[date_key] += t["pnl"]

    # Print summary
    print(f"{'Asset':<6} {'N':>4} {'Real PnL':>9} {'TP-elig':>7} {'TP-fired':>8} {'TP PnL':>9} {'Delta':>9}")
    print("-" * 70)
    total_real = total_tp = 0.0
    total_n = total_eligible = total_fired = 0
    for asset in sorted(stats):
        s = stats[asset]
        if s["n"] == 0:
            continue
        delta = s["tp_pnl"] - s["real_pnl"]
        total_real += s["real_pnl"]
        total_tp += s["tp_pnl"]
        total_n += s["n"]
        total_eligible += s["tp_eligible"]
        total_fired += s["tp_triggered"]
        print(f"{asset:<6} {s['n']:>4} ${s['real_pnl']:>+8.2f} {s['tp_eligible']:>7} {s['tp_triggered']:>8} ${s['tp_pnl']:>+8.2f} ${delta:>+8.2f}")
    print("-" * 70)
    delta = total_tp - total_real
    print(f"{'TOTAL':<6} {total_n:>4} ${total_real:>+8.2f} {total_eligible:>7} {total_fired:>8} ${total_tp:>+8.2f} ${delta:>+8.2f}")
    print()
    print(f"Net TP impact: ${delta:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=Path, default=PROJECT_ROOT / "output" / "trades.csv")
    ap.add_argument("--signals", type=Path, default=PROJECT_ROOT / "output" / "signal_log.csv")
    ap.add_argument("--tp", type=int, default=30)
    ap.add_argument("--max-entry", type=int, default=80)
    ap.add_argument("--fee", type=float, default=1.0)
    ap.add_argument("--rule", choices=["A", "G", "A+G", "unconditional"], default="A")
    args = ap.parse_args()
    backtest(args.trades, args.signals, args.tp, args.max_entry, args.fee, args.rule)


if __name__ == "__main__":
    main()
