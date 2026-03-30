"""
Weekly retrain pipeline.

Chains the full ML pipeline: download -> generate features -> train -> sweep.
Designed to be run by Windows Task Scheduler, cron, or manually.
The live bot hot-reloads new models/config at the next window boundary.

Usage:
    python scripts/weekly_retrain.py
    python scripts/weekly_retrain.py --assets BTC,ETH --days 14
    python scripts/weekly_retrain.py --skip-download
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "weekly_retrain.log"


def log(msg: str) -> None:
    """Print and append to log file."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name: str, cmd: list[str], max_retries: int = 2) -> bool:
    """Run a subprocess step with retry. Returns True on success."""
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            log(f"  Retry {attempt}/{max_retries}...")
            time.sleep(10)  # Brief pause before retry

        log(f"--- {name} ---")
        log(f"  cmd: {' '.join(cmd)}")
        start = time.time()

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )

        elapsed = time.time() - start
        if result.returncode == 0:
            log(f"  OK ({elapsed:.0f}s)")
            lines = result.stdout.strip().splitlines()
            for line in lines[-5:]:
                log(f"  > {line}")
            return True
        else:
            log(f"  FAILED (exit {result.returncode}, {elapsed:.0f}s)")
            for line in result.stderr.strip().splitlines()[-10:]:
                log(f"  ! {line}")

    log(f"  GAVE UP after {max_retries} attempts")
    return False


KALSHI_POLLS_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
AGGTRADES_DIR = PROJECT_ROOT / "data" / "aggtrades"


def purge_old_aggtrades(max_age_days: int = 14) -> None:
    """Delete aggTrades CSV files older than max_age_days.

    File names encode the date: {SYMBOL}-aggTrades-YYYY-MM-DD.csv
    """
    if not AGGTRADES_DIR.exists():
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    cutoff_date = cutoff.date()
    removed = 0

    for asset_dir in AGGTRADES_DIR.iterdir():
        if not asset_dir.is_dir():
            continue
        for csv_file in asset_dir.glob("*-aggTrades-*.csv"):
            # Parse date from filename: "BTCUSDT-aggTrades-2026-03-13.csv"
            try:
                # Date is always the last 10 chars before ".csv"
                stem = csv_file.stem  # "BTCUSDT-aggTrades-2026-03-13"
                file_date_str = stem[-10:]  # "2026-03-13"
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            if file_date < cutoff_date:
                os.remove(csv_file)
                removed += 1

    if removed:
        log(f"Purged {removed} aggTrades files older than {max_age_days} days")
    else:
        log(f"No aggTrades files older than {max_age_days} days to purge")


def purge_kalshi_polls(max_age_days: int = 14) -> None:
    """Delete Kalshi polling JSONL files older than max_age_days.

    File names encode the date: YYYY-MM-DD_HHMM_UTC.jsonl
    """
    if not KALSHI_POLLS_DIR.exists():
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    cutoff_date = cutoff.date()
    removed = 0

    for series_dir in KALSHI_POLLS_DIR.iterdir():
        if not series_dir.is_dir():
            continue
        for jsonl_file in series_dir.glob("*.jsonl"):
            # Parse date from filename: "2026-03-13_0400_UTC.jsonl"
            try:
                file_date_str = jsonl_file.name[:10]  # "2026-03-13"
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            if file_date < cutoff_date:
                os.remove(jsonl_file)
                removed += 1

    if removed:
        log(f"Purged {removed} Kalshi polling files older than {max_age_days} days")
    else:
        log(f"No Kalshi polling files older than {max_age_days} days to purge")


def main():
    parser = argparse.ArgumentParser(description="Weekly ML retrain pipeline")
    parser.add_argument(
        "--assets",
        type=str,
        default="BTC,ETH,SOL,XRP",
        help="Comma-separated assets (default: BTC,ETH,SOL,XRP)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of data to use (default: 30)",
    )
    parser.add_argument(
        "--min-dm",
        type=int,
        default=2,
        help="Min decision minute for training/sweep (default: 2)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download (use existing data)",
    )
    args = parser.parse_args()

    python = sys.executable
    assets = args.assets
    days = str(args.days)
    min_dm = str(args.min_dm)

    log("=" * 60)
    log(f"Weekly retrain: assets={assets} days={days} min_dm={min_dm}")
    log("=" * 60)

    pipeline_start = time.time()
    failed = []

    # Step 1: Download fresh tick data
    if not args.skip_download:
        ok = run_step("Download aggTrades", [
            python, "scripts/download_binance_aggtrades.py",
            "--assets", assets,
            "--days", days,
        ])
        if not ok:
            failed.append("download")
            log("WARNING: Download failed, continuing with existing data")

    # Step 2: Generate training features
    ok = run_step("Generate training data", [
        python, "scripts/generate_training_data.py",
        "--asset", assets,
        "--days", days,
    ])
    if not ok:
        failed.append("generate")
        log("ABORT: Cannot train without training data")
        return 1

    # Step 3: Train XGBoost models (dm 2+)
    ok = run_step("Train XGBoost models (dm 2+)", [
        python, "scripts/train_xgb.py",
        "--asset", assets,
        "--min-dm", min_dm,
    ])
    if not ok:
        failed.append("train_xgb")
        log("ABORT: Cannot sweep without trained XGBoost models")
        return 1

    # Step 4: Generate LSTM training data
    ok = run_step("Generate LSTM training data", [
        python, "scripts/generate_lstm_training_data.py",
        "--asset", assets,
        "--days", days,
    ])
    if not ok:
        log("WARNING: LSTM data generation failed, LSTM models will not be updated")

    # Step 5: Train LSTM models (one at a time to avoid memory crashes)
    asset_list = [a.strip() for a in assets.split(",")]
    lstm_failed = False
    for asset in asset_list:
        ok = run_step(f"Train LSTM model ({asset})", [
            python, "scripts/train_lstm.py",
            "--asset", asset,
            "--min-dm", min_dm,
        ])
        if not ok:
            lstm_failed = True
    if lstm_failed:
        log("WARNING: Some LSTM models failed to train")

    # Step 6: Fetch recent aggTrades via REST (fills gap if bot was offline)
    ok = run_step("Fetch recent aggTrades (REST)", [
        python, "scripts/fetch_recent_aggtrades.py",
        "--assets", assets, "--hours", "48",
    ])
    if not ok:
        log("WARNING: REST aggTrades fetch failed, continuing with existing data")

    # Step 7: Purge old data (>14 days) — after fetch so new data isn't purged
    purge_kalshi_polls(max_age_days=45)
    purge_old_aggtrades(max_age_days=45)

    # Step 8: 3-way ensemble combo sweep (XGB + LSTM + Fusion with dynamic weighting)
    ok = run_step("Ensemble combo sweep (3-way)", [
        python, "scripts/ensemble_combo_sweep.py",
        "--asset", assets,
        "--min-dm", min_dm, "--max-dm", "8",
    ])
    if not ok:
        log("WARNING: Ensemble combo sweep failed")

    total = time.time() - pipeline_start
    log("=" * 60)
    if failed:
        log(f"Done with warnings ({total:.0f}s). Failed steps: {failed}")
    else:
        log(f"Done ({total:.0f}s). All steps succeeded.")
    log("Live bot will pick up new params at next window boundary.")
    log("=" * 60)

    return 1 if "generate" in failed or "train" in failed else 0


if __name__ == "__main__":
    sys.exit(main())
