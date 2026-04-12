"""Full pipeline: regenerate training data + walk-forward model selection.

Chains:
  1. Generate training data (45 days, with Kalshi features + close_ref fix)
  2. Walk-forward model selection (K folds, PnL-scored hyperparams)

Does NOT deploy. Produces a report for human review.

Usage:
    python scripts/run_full_pipeline.py --assets BTC,ETH,SOL,XRP
    python scripts/run_full_pipeline.py --assets BTC,ETH,SOL,XRP,HYPE,BNB,DOGE --days 45
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_step(name, cmd, timeout=7200):
    print(f"\n{'='*70}")
    print(f"  STEP: {name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*70}")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=timeout)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}, {elapsed:.0f}s)")
        return False
    print(f"  OK ({elapsed:.0f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Full retrain pipeline")
    parser.add_argument("--assets", default="BTC,ETH,SOL,XRP", help="Comma-separated assets")
    parser.add_argument("--days", type=int, default=45, help="Days of training data")
    parser.add_argument("--n-folds", type=int, default=4, help="Walk-forward folds")
    parser.add_argument("--threshold", type=float, default=0.70, help="Signal threshold")
    parser.add_argument("--stacked", action="store_true", help="Enable LSTM stacking (lstm_p as XGB feature)")
    args = parser.parse_args()

    python = sys.executable
    pipeline_start = time.time()

    # Step 1: Generate XGB training data for each asset sequentially
    for asset in args.assets.split(","):
        asset = asset.strip()
        ok = run_step(
            f"Generate XGB training data: {asset}",
            [python, "scripts/generate_training_data.py",
             "--asset", asset, "--days", str(args.days)],
        )
        if not ok:
            print(f"ABORT: XGB data generation failed for {asset}")
            return 1

    if args.stacked:
        # Step 2: Generate LSTM training data
        for asset in args.assets.split(","):
            asset = asset.strip()
            ok = run_step(
                f"Generate LSTM training data: {asset}",
                [python, "scripts/generate_lstm_training_data.py",
                 "--asset", asset, "--days", str(args.days)],
            )
            if not ok:
                print(f"WARNING: LSTM data gen failed for {asset}, stacking will use defaults")

        # Step 3: Generate out-of-fold LSTM predictions -> stacked CSVs
        ok = run_step(
            "Generate stacked LSTM features (OOF predictions)",
            [python, "scripts/generate_stacked_features.py",
             "--asset", args.assets,
             "--n-folds", "5"],
            timeout=14400,
        )
        if not ok:
            print("ABORT: stacked feature generation failed")
            return 1

    # Step 4: Walk-forward model selection
    wf_cmd = [
        python, "scripts/walk_forward_select.py",
        "--asset", args.assets,
        "--days", str(args.days),
        "--n-folds", str(args.n_folds),
        "--threshold", str(args.threshold),
    ]
    if args.stacked:
        wf_cmd.append("--stacked")

    ok = run_step(
        f"Walk-forward model selection {'(stacked)' if args.stacked else ''}",
        wf_cmd,
        timeout=14400,
    )
    if not ok:
        print("ABORT: walk-forward selection failed")
        return 1

    total = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE ({total:.0f}s / {total/60:.0f}m)")
    print(f"{'='*70}")
    print(f"\nModels saved to models/ (NOT deployed)")
    print(f"Review the walk-forward results above before deploying.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
