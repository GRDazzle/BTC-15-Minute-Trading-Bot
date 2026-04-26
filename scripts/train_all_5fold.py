"""Train 5 models per asset for walk-forward cross-validation.

For each asset, produces 5 XGB + 5 LSTM models under the common suffix (e.g.
_v14):

  {asset}_v14_fold1_{xgb.json,lstm.pt}  — trained EXCLUDING fold1 dates
  {asset}_v14_fold2_{xgb.json,lstm.pt}  — excluding fold2
  {asset}_v14_fold3_{xgb.json,lstm.pt}  — excluding fold3
  {asset}_v14_fold4_{xgb.json,lstm.pt}  — excluding fold4
  {asset}_v14_{xgb.json,lstm.pt}        — ALL data (deployment + OOS eval)

Fold sweep later uses each _foldN model to evaluate fold N — so sweep picks
combos on data the model has never seen. OOS uses _v14 (all data) — the
deploy-candidate model.

Usage:
    python scripts/train_all_5fold.py --assets BTC,ETH,SOL,XRP,HYPE,BNB,DOGE \\
        --days-total 45 --oos-days 7 --n-folds 4 --model-suffix _v14 \\
        --parallel 2

`--parallel N` controls concurrent XGB trainings. LSTM trainings run
sequentially (GPU contention).
"""
import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
LOG_DIR = PROJECT_ROOT / "output" / "retrain_20260424" / "training"


def compute_folds(anchor: date, days_total: int, oos_days: int,
                  n_folds: int) -> tuple[list[tuple[str, date, date]], tuple[date, date] | None]:
    """Return [(fold_name, from, to), ...] training folds + optional (oos_from, oos_to).

    New scheme (oos_days=0, default): all folds span days_total. The LAST
    fold serves as the OOS test (using its own exclusion model `_foldN`,
    which never saw fold N during training). 1 orphan day max if
    days_total % n_folds != 0; orphan is at the START of the window so the
    most recent fold extends to anchor-1.

    Legacy scheme (oos_days>0): training_days = days_total - oos_days,
    folds span training_days, OOS is the last oos_days held out separately.
    """
    if oos_days == 0:
        fold_len = days_total // n_folds  # e.g. 45 // 4 = 11
        # Anchor train_start so fold N ends at anchor-1 (most recent day).
        train_start = anchor - timedelta(days=fold_len * n_folds)
        folds = []
        for i in range(n_folds):
            lo = train_start + timedelta(days=i * fold_len)
            hi = train_start + timedelta(days=(i + 1) * fold_len) - timedelta(days=1)
            folds.append((f"fold{i+1}", lo, hi))
        return folds, None
    else:
        training_days = days_total - oos_days
        fold_len = training_days // n_folds
        train_start = anchor - timedelta(days=days_total)
        folds = []
        for i in range(n_folds):
            lo = train_start + timedelta(days=i * fold_len)
            hi = train_start + timedelta(days=(i + 1) * fold_len) - timedelta(days=1)
            folds.append((f"fold{i+1}", lo, hi))
        oos_lo = anchor - timedelta(days=oos_days)
        oos_hi = anchor - timedelta(days=1)
        return folds, (oos_lo, oos_hi)


def _run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        f.write(f"[cmd] {' '.join(cmd)}\n\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def train_xgb_one(asset: str, suffix: str, model_suffix: str,
                  exclude_from: date | None, exclude_to: date | None,
                  min_dm: int) -> tuple[str, int]:
    cmd = [
        PYTHON, "-u", "scripts/train_xgb.py",
        "--asset", asset,
        "--min-dm", str(min_dm),
        "--model-suffix", model_suffix,
    ]
    if exclude_from and exclude_to:
        cmd += ["--exclude-from-date", exclude_from.isoformat(),
                "--exclude-to-date", exclude_to.isoformat()]
    log = LOG_DIR / f"xgb_{asset.lower()}{model_suffix}.log"
    return asset + model_suffix, _run(cmd, log)


def train_lstm_one(asset: str, suffix: str, model_suffix: str,
                   exclude_from: date | None, exclude_to: date | None,
                   min_dm: int) -> tuple[str, int]:
    cmd = [
        PYTHON, "-u", "scripts/train_lstm.py",
        "--asset", asset,
        "--min-dm", str(min_dm),
        "--model-suffix", model_suffix,
    ]
    if exclude_from and exclude_to:
        cmd += ["--exclude-from-date", exclude_from.isoformat(),
                "--exclude-to-date", exclude_to.isoformat()]
    log = LOG_DIR / f"lstm_{asset.lower()}{model_suffix}.log"
    return asset + model_suffix, _run(cmd, log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True,
                    help="Comma-separated, e.g. BTC,ETH,SOL,XRP,HYPE,BNB,DOGE")
    ap.add_argument("--days-total", type=int, default=45,
                    help="Training window size in days (default 45 = matches regen)")
    ap.add_argument("--oos-days", type=int, default=0,
                    help="Held-out OOS days separate from folds. Default 0 = "
                         "new scheme (fold N is the OOS test, model `_foldN` "
                         "is genuinely OOS for both params and model). "
                         "Set >0 for legacy scheme with separate OOS window.")
    ap.add_argument("--n-folds", type=int, default=4,
                    help="Number of CV folds (default 4)")
    ap.add_argument("--model-suffix", type=str, default="_v14",
                    help="Base model suffix (default _v14). Fold models get _v14_fold1, etc.")
    ap.add_argument("--min-dm", type=int, default=2)
    ap.add_argument("--end-date", type=str, default=None,
                    help="Anchor end date YYYY-MM-DD (default today UTC)")
    ap.add_argument("--xgb-parallel", type=int, default=2,
                    help="Concurrent XGB trainings (default 2; be gentle on CPU)")
    ap.add_argument("--skip-xgb", action="store_true")
    ap.add_argument("--skip-lstm", action="store_true")
    args = ap.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")]
    anchor = date.fromisoformat(args.end_date) if args.end_date else date.today()
    folds, oos_window = compute_folds(
        anchor, args.days_total, args.oos_days, args.n_folds,
    )

    print(f"{'='*70}")
    print(f"Walk-forward 5-model training")
    print(f"  assets : {assets}")
    print(f"  anchor : {anchor}")
    print(f"  suffix : {args.model_suffix}")
    print(f"  xgb-parallel : {args.xgb_parallel}")
    print(f"{'='*70}")
    if oos_window is None:
        # New scheme: fold N IS the OOS test, no separate window
        for i, (name, lo, hi) in enumerate(folds):
            tag = " (OOS — live deploy model)" if i == len(folds) - 1 else ""
            print(f"  {name}: {lo} -> {hi}  ({(hi-lo).days+1} days){tag}")
    else:
        oos_lo, oos_hi = oos_window
        for name, lo, hi in folds:
            print(f"  {name}: {lo} -> {hi}  ({(hi-lo).days+1} days)")
        print(f"  oos  : {oos_lo} -> {oos_hi}")
    print(f"{'='*70}\n")

    # Build the (asset, model_suffix, exclude_from, exclude_to) list for each
    # of the 5 models per asset.
    jobs = []  # (asset, base_suffix, full_model_suffix, excl_from, excl_to)
    for asset in assets:
        for name, lo, hi in folds:
            jobs.append((asset, args.model_suffix,
                         f"{args.model_suffix}_{name}", lo, hi))
        # All-data model (deployed + OOS eval)
        jobs.append((asset, args.model_suffix,
                     args.model_suffix, None, None))

    print(f"Total training jobs: {len(jobs)}  "
          f"({len(assets)} assets x 5 models)")
    print()

    # --- XGB: concurrent with --xgb-parallel workers ---
    if not args.skip_xgb:
        print(f"\n[XGB] Training {len(jobs)} models with {args.xgb_parallel} parallel workers...")
        xgb_fails = []
        with ProcessPoolExecutor(max_workers=args.xgb_parallel) as executor:
            futures = {
                executor.submit(train_xgb_one, asset, base, ms, ef, et, args.min_dm): (asset, ms)
                for asset, base, ms, ef, et in jobs
            }
            done = 0
            for fut in as_completed(futures):
                asset, ms = futures[fut]
                done += 1
                try:
                    label, rc = fut.result()
                except Exception as e:
                    print(f"  [{done}/{len(jobs)}] {asset}{ms} CRASHED: {e}")
                    xgb_fails.append((asset, ms))
                    continue
                status = "ok" if rc == 0 else f"FAIL(exit={rc})"
                print(f"  [{done}/{len(jobs)}] {label}: {status}")
                if rc != 0:
                    xgb_fails.append((asset, ms))
        print(f"[XGB] done. failures={len(xgb_fails)}")
    else:
        print("[XGB] skipped (--skip-xgb)")

    # --- LSTM: sequential (GPU) ---
    if not args.skip_lstm:
        print(f"\n[LSTM] Training {len(jobs)} models sequentially (GPU)...")
        lstm_fails = []
        for i, (asset, base, ms, ef, et) in enumerate(jobs, 1):
            print(f"  [{i}/{len(jobs)}] {asset}{ms}...", end=" ", flush=True)
            label, rc = train_lstm_one(asset, base, ms, ef, et, args.min_dm)
            status = "ok" if rc == 0 else f"FAIL(exit={rc})"
            print(status)
            if rc != 0:
                lstm_fails.append((asset, ms))
        print(f"[LSTM] done. failures={len(lstm_fails)}")
    else:
        print("[LSTM] skipped (--skip-lstm)")

    print("\nTraining complete. Next step (use the SAME --end-date so fold boundaries match):")
    print(f"  python scripts/retrain_4fold.py --assets {','.join(assets)} "
          f"--end-date {anchor.isoformat()} "
          f"--days-total {args.days_total} --oos-days {args.oos_days} "
          f"--n-folds {args.n_folds} --model-suffix {args.model_suffix}")


if __name__ == "__main__":
    main()
