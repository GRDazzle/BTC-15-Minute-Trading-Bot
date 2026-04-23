"""Batch feature parity audit across recent live trades.

Iterates over all live trade entries (action='trade') in output/signal_log.csv
within a time range, runs audit_feature_parity on each, and summarizes pass
rate against a tolerance.

Intent:
  Single-trade audits are suggestive. A batch audit over 20-50 trades tells us
  whether the training/live gap is genuinely closed, or whether we got lucky
  on a couple of samples. This is the authoritative sign-off test.

Usage:
  python scripts/audit_live_batch.py \
    --after-ts 2026-04-22T00:00:00 --before-ts 2026-04-23T00:00:00 \
    --model-suffix _align2s \
    --tolerance 0.03

Exit 0 if pass_rate >= --min-pass-rate (default 0.80), else 1.
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_SIGNAL_LOG = PROJECT_ROOT / "output" / "signal_log.csv"
AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "audit_feature_parity.py"


def _parse_live_entry(r: dict):
    """Extract (asset, window_iso, dm, offset_s, live_ml_p) from a signal_log row.

    window_id in signal_log is like '20260422_1845' (window START YYYYMMDD_HHMM).
    The event fired at r['timestamp']; offset_s = seconds past (window_start + 5min + dm min)."""
    asset = r.get("asset", "").upper()
    if asset not in {"BTC", "ETH", "SOL", "XRP"}:
        return None
    try:
        dm = int(r.get("dm", "0"))
        live_ml_p = float(r.get("ml_p", "0"))
    except (ValueError, TypeError):
        return None
    wid = r.get("window_id", "")
    if len(wid) < 13 or wid[8] != "_":
        return None
    w_iso = f"{wid[:4]}-{wid[4:6]}-{wid[6:8]}T{wid[9:11]}:{wid[11:13]}:00+00:00"

    # Compute offset within the dm minute from the fire timestamp
    ts = r.get("timestamp", "")
    try:
        from datetime import datetime, timezone, timedelta
        event_ts = datetime.fromisoformat(ts)
        window_dt = datetime.fromisoformat(w_iso)
        dm_boundary = window_dt + timedelta(minutes=5 + dm)
        offset_s = int((event_ts - dm_boundary).total_seconds())
        if offset_s < 0 or offset_s > 59:
            offset_s = max(0, min(59, offset_s))
    except Exception:
        offset_s = 0

    return {"asset": asset, "window_iso": w_iso, "dm": dm,
            "offset_s": offset_s, "live_ml_p": live_ml_p,
            "timestamp": ts, "direction": r.get("direction", "")}


def _run_audit(entry: dict, model_suffix: str) -> tuple[float | None, float | None]:
    """Run audit_feature_parity.py, return (training_csv_xgb, offline_recompute_xgb)."""
    cmd = [
        sys.executable, str(AUDIT_SCRIPT),
        "--asset", entry["asset"],
        "--window", entry["window_iso"],
        "--dm", str(entry["dm"]),
        "--offset-seconds", str(entry["offset_s"]),
    ]
    if model_suffix:
        cmd += ["--model-suffix", model_suffix]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None, None
    if res.returncode != 0:
        return None, None
    # Parse "from training_csv features: 0.8687" and "from offline recompute: 0.8167"
    training_p = offline_p = None
    for line in res.stdout.splitlines():
        m = re.search(r"from training_csv features:\s+([0-9.]+)", line)
        if m:
            try:
                training_p = float(m.group(1))
            except ValueError:
                pass
        m = re.search(r"from offline recompute:\s+([0-9.]+)", line)
        if m:
            try:
                offline_p = float(m.group(1))
            except ValueError:
                pass
    return training_p, offline_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default=str(LIVE_SIGNAL_LOG))
    ap.add_argument("--after-ts", required=True)
    ap.add_argument("--before-ts", required=True)
    ap.add_argument("--asset", action="append",
                    help="Filter by asset; repeat for multiple; default all 4")
    ap.add_argument("--model-suffix", default="",
                    help="e.g. _align2s to audit staging model")
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--min-pass-rate", type=float, default=0.80)
    ap.add_argument("--limit", type=int, default=100,
                    help="Max trades to audit (default 100)")
    ap.add_argument("--out", default=None,
                    help="Optional CSV output with per-trade audit results")
    args = ap.parse_args()

    assets = set(a.upper() for a in (args.asset or ["BTC", "ETH", "SOL", "XRP"]))
    entries = []
    with open(args.live) as f:
        for r in csv.DictReader(f):
            ts = r.get("timestamp", "")
            if ts < args.after_ts or ts >= args.before_ts:
                continue
            if r.get("action") != "trade":
                continue
            if r.get("asset") not in assets:
                continue
            e = _parse_live_entry(r)
            if e is not None:
                entries.append(e)
    entries = entries[:args.limit]
    print(f"Auditing {len(entries)} live trades in {args.after_ts} .. {args.before_ts}"
          f" with model_suffix='{args.model_suffix}'")

    results = []
    for i, e in enumerate(entries, 1):
        tp, op = _run_audit(e, args.model_suffix)
        gap_tp = abs(tp - e["live_ml_p"]) if tp is not None else None
        gap_op = abs(op - e["live_ml_p"]) if op is not None else None
        passed = gap_tp is not None and gap_tp <= args.tolerance
        results.append({**e, "training_p": tp, "offline_p": op,
                        "gap_tp": gap_tp, "gap_op": gap_op, "passed": passed})
        status = "PASS" if passed else ("FAIL" if tp is not None else "SKIP")
        gap_str = f"{gap_tp:.4f}" if gap_tp is not None else "  n/a "
        tp_str = f"{tp:.4f}" if tp is not None else "  n/a "
        print(f"  [{i:2d}/{len(entries)}] {e['asset']} "
              f"{e['window_iso'][11:16]} dm={e['dm']}+{e['offset_s']}s "
              f"live_ml_p={e['live_ml_p']:.4f} train_ml_p={tp_str} "
              f"gap={gap_str} {status}")

    evaluated = [r for r in results if r["gap_tp"] is not None]
    if not evaluated:
        print("\nNo audits produced a result. Check training CSV and model availability.")
        sys.exit(2)
    passed = sum(1 for r in evaluated if r["passed"])
    pass_rate = passed / len(evaluated)
    gaps = [r["gap_tp"] for r in evaluated]
    print()
    print("=== BATCH AUDIT SUMMARY ===")
    print(f"Audited:     {len(evaluated)} / {len(entries)} (skipped {len(entries)-len(evaluated)})")
    print(f"Pass rate:   {passed}/{len(evaluated)} = {pass_rate*100:.1f}% (tolerance ±{args.tolerance})")
    print(f"Gap mean:    {sum(gaps)/len(gaps):.4f}")
    print(f"Gap max:     {max(gaps):.4f}")
    print(f"Gap median:  {sorted(gaps)[len(gaps)//2]:.4f}")

    # Per-asset breakdown
    by_asset = {}
    for r in evaluated:
        a = r["asset"]
        by_asset.setdefault(a, []).append(r)
    print("\nPer-asset:")
    for a in sorted(by_asset):
        arr = by_asset[a]
        p = sum(1 for r in arr if r["passed"])
        g = sum(r["gap_tp"] for r in arr) / len(arr)
        print(f"  {a}: {p}/{len(arr)} ({p/len(arr)*100:.0f}%)  mean gap={g:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "timestamp", "asset", "window_iso", "dm", "offset_s", "direction",
                "live_ml_p", "training_p", "offline_p", "gap_tp", "gap_op", "passed",
            ])
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"\nPer-trade results written to {out_path}")

    if pass_rate < args.min_pass_rate:
        print(f"\nFAIL: {pass_rate*100:.1f}% < {args.min_pass_rate*100:.0f}% required")
        sys.exit(1)
    print(f"\nPASS")


if __name__ == "__main__":
    main()
