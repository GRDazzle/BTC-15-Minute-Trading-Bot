"""
Weekly retrain wrapper with live progress display.

Shows which step is running, elapsed time, and status.
Resumes from where it left off if --skip-download is used.

Usage:
    python retrain.py                    # Full retrain
    python retrain.py --skip-download    # Skip download (already done)
    python retrain.py --assets BTC,ETH   # Specific assets
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STEPS = [
    ("1/8", "Download aggTrades",        ["scripts/download_binance_aggtrades.py", "--assets", "{assets}", "--days", "{days}"]),
    ("2/8", "Generate XGBoost data",     ["scripts/generate_training_data.py", "--asset", "{assets}", "--days", "{days}"]),
    ("3/8", "Train XGBoost models",      ["scripts/train_xgb.py", "--asset", "{assets}", "--min-dm", "{min_dm}"]),
    ("4/8", "Generate LSTM data",        ["scripts/generate_lstm_training_data.py", "--asset", "{assets}", "--days", "{days}"]),
    ("5/8", "Train LSTM models",         ["scripts/train_lstm.py", "--asset", "{assets}", "--min-dm", "{min_dm}"]),
    ("6/8", "Fetch recent aggTrades",    ["scripts/fetch_recent_aggtrades.py", "--assets", "{assets}", "--hours", "48"]),
    ("7/8", "Purge old data",            None),  # Handled inline
    ("8/8", "3-way ensemble sweep",      ["scripts/ensemble_combo_sweep.py", "--asset", "{assets}", "--min-dm", "{min_dm}", "--max-dm", "8"]),
]


def run_step(step_num: str, name: str, cmd: list[str] | None, params: dict) -> bool:
    """Run one pipeline step with progress display."""
    print(f"\n{'='*60}")
    print(f"  [{step_num}] {name}")
    print(f"{'='*60}")

    if cmd is None:
        # Inline step (purge)
        from scripts.weekly_retrain import purge_kalshi_polls, purge_old_aggtrades
        purge_kalshi_polls(max_age_days=14)
        purge_old_aggtrades(max_age_days=14)
        print("  Done.")
        return True

    # Substitute params
    full_cmd = [sys.executable]
    for arg in cmd:
        full_cmd.append(arg.format(**params))

    print(f"  cmd: {' '.join(full_cmd)}")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    print()

    t0 = time.time()
    result = subprocess.run(
        full_cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  OK ({elapsed:.0f}s)")
        return True
    else:
        print(f"\n  FAILED (exit {result.returncode}, {elapsed:.0f}s)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Weekly retrain with progress display")
    parser.add_argument("--assets", default="BTC,ETH,SOL,XRP")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--min-dm", type=int, default=2)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    params = {
        "assets": args.assets,
        "days": str(args.days),
        "min_dm": str(args.min_dm),
    }

    print(f"\n{'#'*60}")
    print(f"  WEEKLY RETRAIN - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Assets: {args.assets}  Days: {args.days}  Min DM: {args.min_dm}")
    print(f"{'#'*60}")

    pipeline_start = time.time()
    failed = []

    for step_num, name, cmd in STEPS:
        # Skip download if requested
        if step_num == "1/8" and args.skip_download:
            print(f"\n  [{step_num}] {name} -- SKIPPED (--skip-download)")
            continue

        ok = run_step(step_num, name, cmd, params)

        if not ok:
            failed.append(name)
            # Abort on critical failures
            if step_num in ("2/8", "3/8"):
                print(f"\n  ABORT: {name} is required for subsequent steps.")
                break
            print(f"  WARNING: {name} failed, continuing...")

    total = time.time() - pipeline_start
    print(f"\n{'#'*60}")
    if failed:
        print(f"  DONE with warnings ({total/60:.0f}min). Failed: {', '.join(failed)}")
    else:
        print(f"  DONE ({total/60:.0f}min). All steps succeeded.")
    print(f"  Bot will hot-reload new models at next window boundary.")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
