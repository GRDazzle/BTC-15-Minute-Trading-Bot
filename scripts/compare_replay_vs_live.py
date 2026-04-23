"""Compare replay_live_inference output against actual live signal_log.

For a given date range and set of models (old or _align2s), this script:
  1. Loads output/replay_trades.csv (from a prior replay run)
  2. Loads output/signal_log.csv filtered to the same range
  3. Matches each live trade-action to its nearest replay entry (by window + dm)
  4. Reports gap distribution on ml_p, lstm_p, direction agreement, trade count
  5. Optionally writes a side-by-side CSV for deeper manual review

Intent:
  The replay harness calls the SAME MLProcessor + LSTMProcessor live uses.
  If replay ml_p matches live ml_p within tolerance, the model math is faithful.
  Residual gaps are state-timing (sub-second tick buffer drift), which we
  can't fully control but can bound.

Usage:
  # Replay first:
  python scripts/replay_live_inference.py \
    --from 2026-04-22 --to 2026-04-22 \
    --xgb-suffix _align2s --lstm-suffix _align2s \
    --out output/replay_align2s_apr22.csv

  # Then diff:
  python scripts/compare_replay_vs_live.py \
    --replay output/replay_align2s_apr22.csv \
    --after-ts 2026-04-22T00:00:00 --before-ts 2026-04-23T00:00:00 \
    --tolerance 0.03

Exit code 0 if pass-rate >= --min-pass-rate (default 0.80), else 1.
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_SIGNAL_LOG = PROJECT_ROOT / "output" / "signal_log.csv"


def _load_live_trades(path: Path, after_ts: str, before_ts: str, assets: set) -> list[dict]:
    """Load live signal_log entries with action='trade' within range."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            ts = r.get("timestamp", "")
            if ts < after_ts or ts >= before_ts:
                continue
            if r.get("action") != "trade":
                continue
            if assets and r.get("asset") not in assets:
                continue
            rows.append(r)
    return rows


def _load_replay_trades(path: Path, assets: set) -> list[dict]:
    rows = []
    if not path.exists():
        print(f"ERROR: replay CSV not found: {path}", file=sys.stderr)
        sys.exit(2)
    with open(path) as f:
        for r in csv.DictReader(f):
            if assets and r.get("asset") not in assets:
                continue
            rows.append(r)
    return rows


def _window_id_from_live(r: dict) -> str:
    """Live signal_log uses 'window_id' like '20260422_1845'.
    Replay uses 'window_id' like '2026-04-22T19:00:00Z'.
    Normalize to 'YYYY-MM-DD HH:MM' for matching."""
    w = r.get("window_id", "")
    if "T" in w:  # replay format
        return w[:16].replace("T", " ")
    # live format "20260422_1845" -> 2026-04-22 18:45
    if len(w) >= 13 and w[8] == "_":
        return f"{w[:4]}-{w[4:6]}-{w[6:8]} {w[9:11]}:{w[11:13]}"
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True, help="Path to replay_trades.csv")
    ap.add_argument("--live", default=str(LIVE_SIGNAL_LOG), help="Path to live signal_log.csv")
    ap.add_argument("--after-ts", required=True, help="Start ISO timestamp (inclusive)")
    ap.add_argument("--before-ts", required=True, help="End ISO timestamp (exclusive)")
    ap.add_argument("--asset", action="append",
                    help="Filter by asset; repeat for multiple; default all 4")
    ap.add_argument("--tolerance", type=float, default=0.03,
                    help="|replay_ml_p - live_ml_p| threshold for a match")
    ap.add_argument("--min-pass-rate", type=float, default=0.80,
                    help="Fraction of matched pairs within tolerance required for exit 0")
    ap.add_argument("--out", default=None,
                    help="Optional side-by-side CSV path for manual review")
    args = ap.parse_args()

    assets = set(a.upper() for a in (args.asset or ["BTC", "ETH", "SOL", "XRP"]))

    live_trades = _load_live_trades(Path(args.live), args.after_ts, args.before_ts, assets)
    replay_trades = _load_replay_trades(Path(args.replay), assets)
    print(f"Loaded {len(live_trades)} live trades, {len(replay_trades)} replay trades "
          f"in range {args.after_ts} .. {args.before_ts}")

    # Index replay by (asset, window_norm)
    replay_by_key = {}
    for r in replay_trades:
        k = (r.get("asset"), _window_id_from_live(r))
        replay_by_key[k] = r

    # Match each live trade to its replay counterpart (if any)
    matched = []
    live_only = []
    for lr in live_trades:
        k = (lr.get("asset"), _window_id_from_live(lr))
        rr = replay_by_key.pop(k, None)
        if rr is not None:
            matched.append((lr, rr))
        else:
            live_only.append(lr)
    replay_only = list(replay_by_key.values())

    # Analyze matched pairs
    within_tol = 0
    direction_agree = 0
    ml_p_gaps = []
    for lr, rr in matched:
        try:
            lp = float(lr.get("ml_p") or 0.0)
            rp = float(rr.get("ml_p") or 0.0)
        except ValueError:
            continue
        gap = abs(lp - rp)
        ml_p_gaps.append(gap)
        if gap <= args.tolerance:
            within_tol += 1
        if lr.get("direction") == rr.get("direction"):
            direction_agree += 1

    total = len(matched)
    pass_rate = within_tol / total if total else 0.0
    dir_rate = direction_agree / total if total else 0.0

    # Report
    print("\n=== MATCH STATS ===")
    print(f"Matched pairs (live & replay both traded):  {total}")
    print(f"Live-only trades (replay skipped):          {len(live_only)}")
    print(f"Replay-only trades (live skipped):          {len(replay_only)}")
    print()
    if matched:
        print(f"ml_p gap statistics:")
        print(f"  mean: {sum(ml_p_gaps)/len(ml_p_gaps):.4f}")
        print(f"  max:  {max(ml_p_gaps):.4f}")
        print(f"  pairs within ±{args.tolerance} tolerance: {within_tol}/{total} "
              f"({pass_rate*100:.1f}%)")
        print(f"  direction agreement: {direction_agree}/{total} ({dir_rate*100:.1f}%)")

    if live_only:
        print(f"\n=== LIVE-ONLY (first 10) — trades replay would not have taken ===")
        for lr in live_only[:10]:
            try:
                pnl = float(lr.get("pnl") or 0.0)
            except ValueError:
                pnl = 0.0
            print(f"  {lr.get('timestamp','')[:19]} {lr.get('asset')} "
                  f"{lr.get('direction')} ml_p={lr.get('ml_p')} "
                  f"entry={lr.get('entry_price')}c contracts={lr.get('contracts')} pnl={pnl:+.2f}")

    if replay_only:
        print(f"\n=== REPLAY-ONLY (first 10) — trades live did not take ===")
        for rr in replay_only[:10]:
            try:
                pnl = float(rr.get("pnl") or 0.0)
            except ValueError:
                pnl = 0.0
            print(f"  {rr.get('timestamp','')[:19]} {rr.get('asset')} "
                  f"{rr.get('direction')} ml_p={rr.get('ml_p')} "
                  f"entry={rr.get('price_cents')}c pnl={pnl:+.2f}")

    # Optional side-by-side CSV
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "type", "asset", "window", "dm_live", "dm_replay",
                "dir_live", "dir_replay", "ml_p_live", "ml_p_replay", "ml_p_gap",
                "ens_p_live", "ens_p_replay",
                "entry_live", "entry_replay",
                "pnl_live", "pnl_replay",
            ])
            for lr, rr in matched:
                try:
                    gap = abs(float(lr.get("ml_p") or 0.0) - float(rr.get("ml_p") or 0.0))
                except ValueError:
                    gap = 0.0
                w.writerow([
                    "match", lr.get("asset"), _window_id_from_live(lr),
                    lr.get("dm"), rr.get("dm"),
                    lr.get("direction"), rr.get("direction"),
                    lr.get("ml_p"), rr.get("ml_p"), f"{gap:.4f}",
                    lr.get("ensemble_p"), rr.get("ensemble_p"),
                    lr.get("entry_price"), rr.get("price_cents"),
                    lr.get("pnl", ""), rr.get("pnl", ""),
                ])
            for lr in live_only:
                w.writerow(["live_only", lr.get("asset"), _window_id_from_live(lr),
                            lr.get("dm"), "",
                            lr.get("direction"), "",
                            lr.get("ml_p"), "", "",
                            lr.get("ensemble_p"), "",
                            lr.get("entry_price"), "",
                            lr.get("pnl", ""), ""])
            for rr in replay_only:
                w.writerow(["replay_only", rr.get("asset"), _window_id_from_live(rr),
                            "", rr.get("dm"),
                            "", rr.get("direction"),
                            "", rr.get("ml_p"), "",
                            "", rr.get("ensemble_p"),
                            "", rr.get("price_cents"),
                            "", rr.get("pnl", "")])
        print(f"\nSide-by-side CSV written to {out_path}")

    if pass_rate < args.min_pass_rate:
        print(f"\nFAIL: pass_rate {pass_rate*100:.1f}% < {args.min_pass_rate*100:.0f}% required")
        sys.exit(1)
    print(f"\nPASS: pass_rate {pass_rate*100:.1f}% >= {args.min_pass_rate*100:.0f}%")


if __name__ == "__main__":
    main()
