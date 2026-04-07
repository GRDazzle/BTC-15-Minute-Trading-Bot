"""
Weekly retrain pipeline.

Chains the full ML pipeline: download -> generate features -> train -> sweep.
Designed to be run by Windows Task Scheduler, cron, or manually.
The live bot hot-reloads new models/config at the next window boundary.

Behavior:
    All assets get RETRAINED. New models are written to date-tagged paths
    (e.g. models/BTC_2026-04-07_xgb.json). The dated copies are kept as
    snapshots/rollback targets.

    Only assets with NEGATIVE PnL over the last --pnl-days (default 7)
    have their dated models PROMOTED to the live path (e.g. BTC_xgb.json),
    and only those assets get re-swept. Profitable assets keep their
    existing live model and config untouched.

    Use --promote-all to force promotion of every newly trained model
    regardless of PnL.

Usage:
    python scripts/weekly_retrain.py
    python scripts/weekly_retrain.py --assets BTC,ETH --days 14
    python scripts/weekly_retrain.py --skip-download
    python scripts/weekly_retrain.py --promote-all
    python scripts/weekly_retrain.py --day-type weekday   # Sunday: retrain weekday models only
    python scripts/weekly_retrain.py --day-type weekend   # Monday: retrain weekend models only
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import json as _json
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

        # Run with below-normal priority so live bot isn't starved
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            creationflags=creation_flags,
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


TRADES_CSV = PROJECT_ROOT / "output" / "trades.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def promote_dated_models(assets: list[str], variants: list[str], date_tag: str) -> int:
    """Copy date-tagged trained files into their live paths.

    For each (asset, variant) pair, copies:
      models/{ASSET}{variant}_{date_tag}_xgb.json -> models/{ASSET}{variant}_xgb.json
      models/{ASSET}{variant}_{date_tag}_lstm.pt  -> models/{ASSET}{variant}_lstm.pt

    `variant` is the live variant suffix: '' (standard), '_weekday', '_weekend'.

    Returns the number of files actually promoted.
    """
    promoted = 0
    for asset in assets:
        for variant in variants:
            for ext in ("xgb.json", "lstm.pt"):
                dated = MODELS_DIR / f"{asset}{variant}_{date_tag}_{ext}"
                live = MODELS_DIR / f"{asset}{variant}_{ext}"
                if not dated.exists():
                    log(f"  [promote] MISSING {dated.name} (skipping)")
                    continue
                try:
                    shutil.copy2(dated, live)
                    log(f"  [promote] {dated.name} -> {live.name}")
                    promoted += 1
                except Exception as e:
                    log(f"  [promote] FAILED {dated.name}: {e}")
    return promoted


def filter_assets_by_pnl(asset_list: list[str], days: int, min_trades: int = 20) -> tuple[list[str], list[str]]:
    """Return (assets_to_promote, assets_kept) based on recent PnL.

    Rules:
      - asset has < min_trades in window  -> promote (insufficient data, default to fresh)
      - asset has negative PnL in window  -> promote (replace losing model)
      - asset has positive PnL in window  -> keep (current model is winning, leave it)
    """
    if not TRADES_CSV.exists():
        log("No trades.csv found, retraining all assets")
        return asset_list, []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    pnl_by_asset = {a: 0.0 for a in asset_list}
    n_by_asset = {a: 0 for a in asset_list}

    try:
        with open(TRADES_CSV, "r") as f:
            for row in csv.DictReader(f):
                asset = row.get("asset")
                if asset not in pnl_by_asset:
                    continue
                try:
                    ts = datetime.fromisoformat(row["timestamp"])
                except (ValueError, KeyError):
                    continue
                if ts < cutoff:
                    continue
                pnl_str = (row.get("pnl") or "").strip()
                if not pnl_str or pnl_str == "pending":
                    continue
                try:
                    pnl_by_asset[asset] += float(pnl_str)
                    n_by_asset[asset] += 1
                except ValueError:
                    continue
    except Exception as e:
        log(f"Failed to read trades.csv: {e} -- retraining all assets")
        return asset_list, []

    to_retrain: list[str] = []
    skipped: list[str] = []
    log(f"Per-asset PnL over last {days} days (min_trades={min_trades}):")
    for asset in asset_list:
        pnl = pnl_by_asset[asset]
        n = n_by_asset[asset]
        if n < min_trades:
            log(f"  {asset}: ${pnl:+.2f} over {n} trades (insufficient) -> RETRAIN")
            to_retrain.append(asset)
        elif pnl < 0:
            log(f"  {asset}: ${pnl:+.2f} over {n} trades -> RETRAIN")
            to_retrain.append(asset)
        else:
            log(f"  {asset}: ${pnl:+.2f} over {n} trades -> KEEP (profitable)")
            skipped.append(asset)
    return to_retrain, skipped


def send_telegram(message: str) -> None:
    """Send a Telegram notification via the bot API."""
    env_path = Path.home() / ".claude" / "channels" / "telegram" / ".env"
    if not env_path.exists():
        return
    token = None
    chat_id = "8292385179"
    for line in env_path.read_text().splitlines():
        if line.startswith("TELEGRAM_BOT_TOKEN="):
            token = line.split("=", 1)[1].strip()
    if not token:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = _json.dumps({"chat_id": chat_id, "text": message}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log(f"Telegram notification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Weekly ML retrain pipeline")
    parser.add_argument(
        "--assets",
        type=str,
        default="BTC,ETH,SOL,XRP,HYPE,BNB,DOGE",
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
    parser.add_argument(
        "--day-type",
        choices=["all", "weekday", "weekend"],
        default="all",
        help="Which models to retrain (default: all)",
    )
    parser.add_argument(
        "--promote-all",
        action="store_true",
        help="Promote all newly trained models regardless of PnL (default: only promote assets with negative PnL)",
    )
    parser.add_argument(
        "--pnl-days",
        type=int,
        default=7,
        help="Window in days for per-asset PnL filter (default: 7)",
    )
    parser.add_argument(
        "--min-trades-for-keep",
        type=int,
        default=20,
        help="Minimum trades in window before an asset is allowed to be kept on positive PnL (default: 20)",
    )
    parser.add_argument(
        "--date-tag",
        type=str,
        default=None,
        help="Date tag for trained model files (default: today's UTC date YYYY-MM-DD)",
    )
    args = parser.parse_args()

    python = sys.executable
    assets = args.assets
    days = str(args.days)
    min_dm = str(args.min_dm)
    day_type = args.day_type

    asset_list = [a.strip() for a in assets.split(",")]

    # Date tag for staged models, e.g. "2026-04-07"
    date_tag = args.date_tag or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    log("=" * 60)
    log(
        f"Weekly retrain: assets={assets} days={days} min_dm={min_dm} "
        f"day_type={day_type} date_tag={date_tag}"
    )
    log("=" * 60)

    pipeline_start = time.time()
    failed = []
    promoted_assets: list[str] = []
    kept_assets: list[str] = []

    # Step 1: Download fresh tick data from Coinbase
    if not args.skip_download:
        ok = run_step("Download Coinbase trades", [
            python, "scripts/fetch_coinbase_trades.py",
            "--assets", assets,
            "--days", days,
        ])
        if not ok:
            failed.append("download")
            log("WARNING: Download failed, continuing with existing data")

    # Steps 2-8: Standard models (only when day_type is "all")
    if day_type == "all":
        pass  # fall through to standard pipeline below
    else:
        # Single day-type retrain (Sun: weekday models, Mon: weekend models)
        log(f"Skipping standard pipeline (day_type={day_type})")
        live_variant = f"_{day_type}"           # _weekday or _weekend
        dated_variant = f"_{day_type}_{date_tag}"  # _weekday_2026-04-07

        ok = run_step(f"Generate XGB training data ({day_type})", [
            python, "scripts/generate_training_data.py",
            "--asset", assets, "--days", days,
            "--day-filter", day_type,
        ])
        if not ok:
            log(f"ABORT: {day_type} XGB data gen failed")
            return 1

        ok = run_step(f"Train XGB models ({day_type}) [date-tagged]", [
            python, "scripts/train_xgb.py",
            "--asset", assets, "--min-dm", min_dm,
            "--data-suffix", live_variant, "--model-suffix", dated_variant,
        ])
        xgb_ok = ok
        if not ok:
            failed.append(f"train_xgb_{day_type}")

        ok = run_step(f"Generate LSTM data ({day_type})", [
            python, "scripts/generate_lstm_training_data.py",
            "--asset", assets, "--days", days,
            "--day-filter", day_type,
        ])
        if not ok:
            log(f"WARNING: {day_type} LSTM data gen failed")

        lstm_ok = True
        for asset in asset_list:
            ok = run_step(f"Train LSTM ({asset} {day_type}) [date-tagged]", [
                python, "scripts/train_lstm.py",
                "--asset", asset, "--min-dm", min_dm,
                "--data-suffix", live_variant, "--model-suffix", dated_variant,
            ])
            if not ok:
                lstm_ok = False
                log(f"WARNING: {day_type} LSTM training failed for {asset}")

        # Decide which assets to promote (overall PnL filter; same gate for variant)
        if args.promote_all:
            promoted_assets = list(asset_list)
            kept_assets = []
            log("--promote-all set: promoting every asset")
        else:
            promoted_assets, kept_assets = filter_assets_by_pnl(
                asset_list,
                days=args.pnl_days,
                min_trades=args.min_trades_for_keep,
            )

        if promoted_assets and xgb_ok and lstm_ok:
            log(f"Promoting {day_type} models for: {','.join(promoted_assets)}")
            promote_dated_models(promoted_assets, [live_variant], date_tag)
        else:
            log(
                f"SKIPPING {day_type} promotion -- "
                f"promoted={promoted_assets}, xgb_ok={xgb_ok}, lstm_ok={lstm_ok}"
            )

        # Run sweep ONLY for promoted assets (kept assets keep their existing config)
        if promoted_assets and xgb_ok and lstm_ok:
            ok = run_step(f"Ensemble sweep ({day_type}) [promoted only]", [
                python, "scripts/ensemble_combo_sweep.py",
                "--asset", ",".join(promoted_assets),
                "--min-dm", min_dm, "--max-dm", "8",
                "--model-suffix", live_variant, "--day-filter", day_type,
                "--kalshi-days", "14",
            ])
            if not ok:
                log(f"WARNING: {day_type} ensemble sweep failed")
        elif not promoted_assets:
            log(f"SKIPPING {day_type} ensemble sweep -- no assets to promote")

        total = time.time() - pipeline_start
        kept_msg = f" Kept: {','.join(kept_assets)}." if kept_assets else ""
        promoted_msg = f" Promoted: {','.join(promoted_assets)}." if promoted_assets else " Promoted: none."
        log("=" * 60)
        if failed:
            log(f"Done with warnings ({total:.0f}s). Failed steps: {failed}")
            send_telegram(
                f"Retrain ({day_type}) done with warnings ({total:.0f}s). Failed: {failed}.{promoted_msg}{kept_msg}"
            )
        else:
            log(f"Done ({total:.0f}s). All steps succeeded.")
            send_telegram(
                f"Retrain ({day_type}) complete ({total:.0f}s).{promoted_msg}{kept_msg} Bot will hot-reload."
            )
        log("=" * 60)
        return 1 if failed else 0

    # ---- Standard models (date-tagged) ----
    std_dated_suffix = f"_{date_tag}"   # e.g. _2026-04-07

    ok = run_step("Generate training data", [
        python, "scripts/generate_training_data.py",
        "--asset", assets,
        "--days", days,
    ])
    if not ok:
        failed.append("generate")
        log("ABORT: Cannot train without training data")
        return 1

    std_xgb_ok = run_step("Train XGBoost models [date-tagged]", [
        python, "scripts/train_xgb.py",
        "--asset", assets,
        "--min-dm", min_dm,
        "--model-suffix", std_dated_suffix,
    ])
    if not std_xgb_ok:
        failed.append("train_xgb")
        log("WARNING: Standard XGB training failed")

    ok = run_step("Generate LSTM training data", [
        python, "scripts/generate_lstm_training_data.py",
        "--asset", assets,
        "--days", days,
    ])
    if not ok:
        log("WARNING: LSTM data generation failed, LSTM models will not be updated")

    std_lstm_ok = True
    for asset in asset_list:
        ok = run_step(f"Train LSTM model ({asset}) [date-tagged]", [
            python, "scripts/train_lstm.py",
            "--asset", asset,
            "--min-dm", min_dm,
            "--model-suffix", std_dated_suffix,
        ])
        if not ok:
            std_lstm_ok = False
    if not std_lstm_ok:
        log("WARNING: Some standard LSTM models failed to train")

    # ---- Weekday models (date-tagged) ----
    wd_dated_suffix = f"_weekday_{date_tag}"
    ok = run_step("Generate XGB training data (weekday)", [
        python, "scripts/generate_training_data.py",
        "--asset", assets, "--days", days,
        "--day-filter", "weekday",
    ])
    wd_xgb_ok = False
    wd_lstm_ok = True
    if ok:
        wd_xgb_ok = run_step("Train XGB models (weekday) [date-tagged]", [
            python, "scripts/train_xgb.py",
            "--asset", assets, "--min-dm", min_dm,
            "--data-suffix", "_weekday", "--model-suffix", wd_dated_suffix,
        ])
        ok2 = run_step("Generate LSTM data (weekday)", [
            python, "scripts/generate_lstm_training_data.py",
            "--asset", assets, "--days", days,
            "--day-filter", "weekday",
        ])
        if not ok2:
            log("WARNING: weekday LSTM data gen failed")
        for asset in asset_list:
            ok2 = run_step(f"Train LSTM ({asset} weekday) [date-tagged]", [
                python, "scripts/train_lstm.py",
                "--asset", asset, "--min-dm", min_dm,
                "--data-suffix", "_weekday", "--model-suffix", wd_dated_suffix,
            ])
            if not ok2:
                wd_lstm_ok = False
    else:
        log("WARNING: weekday XGB data gen failed -- skipping weekday models")

    # ---- Weekend models (date-tagged) ----
    we_dated_suffix = f"_weekend_{date_tag}"
    ok = run_step("Generate XGB training data (weekend)", [
        python, "scripts/generate_training_data.py",
        "--asset", assets, "--days", days,
        "--day-filter", "weekend",
    ])
    we_xgb_ok = False
    we_lstm_ok = True
    if ok:
        we_xgb_ok = run_step("Train XGB models (weekend) [date-tagged]", [
            python, "scripts/train_xgb.py",
            "--asset", assets, "--min-dm", min_dm,
            "--data-suffix", "_weekend", "--model-suffix", we_dated_suffix,
        ])
        ok2 = run_step("Generate LSTM data (weekend)", [
            python, "scripts/generate_lstm_training_data.py",
            "--asset", assets, "--days", days,
            "--day-filter", "weekend",
        ])
        if not ok2:
            log("WARNING: weekend LSTM data gen failed")
        for asset in asset_list:
            ok2 = run_step(f"Train LSTM ({asset} weekend) [date-tagged]", [
                python, "scripts/train_lstm.py",
                "--asset", asset, "--min-dm", min_dm,
                "--data-suffix", "_weekend", "--model-suffix", we_dated_suffix,
            ])
            if not ok2:
                we_lstm_ok = False
    else:
        log("WARNING: weekend XGB data gen failed -- skipping weekend models")

    # ---- Purge old data ----
    purge_kalshi_polls(max_age_days=45)
    purge_old_aggtrades(max_age_days=45)

    # ---- Decide which assets to PROMOTE (based on recent live PnL) ----
    if args.promote_all:
        promoted_assets = list(asset_list)
        kept_assets = []
        log("--promote-all set: promoting every asset")
    else:
        promoted_assets, kept_assets = filter_assets_by_pnl(
            asset_list,
            days=args.pnl_days,
            min_trades=args.min_trades_for_keep,
        )

    if not promoted_assets:
        log("Nothing to promote -- all assets profitable. Dated models kept as snapshots.")
        total = time.time() - pipeline_start
        send_telegram(
            f"Retrain (all) complete ({total:.0f}s). Trained {len(asset_list)} assets, "
            f"promoted none. All profitable. Kept: {','.join(kept_assets)}"
        )
        log("=" * 60)
        return 0

    # ---- Promote standard models for negative-PnL assets ----
    log(f"Promoting standard models for: {','.join(promoted_assets)}")
    if std_xgb_ok and std_lstm_ok:
        promote_dated_models(promoted_assets, [""], date_tag)
    else:
        log(
            f"SKIPPING standard promotion -- "
            f"xgb_ok={std_xgb_ok}, lstm_ok={std_lstm_ok}"
        )
        failed.append("std_promote_skipped")

    # ---- Promote weekday/weekend variants for the same assets ----
    if wd_xgb_ok and wd_lstm_ok:
        log(f"Promoting weekday models for: {','.join(promoted_assets)}")
        promote_dated_models(promoted_assets, ["_weekday"], date_tag)
    else:
        log(f"SKIPPING weekday promotion -- xgb_ok={wd_xgb_ok}, lstm_ok={wd_lstm_ok}")

    if we_xgb_ok and we_lstm_ok:
        log(f"Promoting weekend models for: {','.join(promoted_assets)}")
        promote_dated_models(promoted_assets, ["_weekend"], date_tag)
    else:
        log(f"SKIPPING weekend promotion -- xgb_ok={we_xgb_ok}, lstm_ok={we_lstm_ok}")

    # ---- Sweep ONLY promoted assets (kept assets keep their existing config) ----
    promoted_csv = ",".join(promoted_assets)

    if std_xgb_ok and std_lstm_ok:
        ok = run_step("Ensemble combo sweep (3-way) [promoted only]", [
            python, "scripts/ensemble_combo_sweep.py",
            "--asset", promoted_csv,
            "--min-dm", min_dm, "--max-dm", "8",
            "--kalshi-days", "14",
        ])
        if not ok:
            log("WARNING: Standard ensemble sweep failed")

    if wd_xgb_ok and wd_lstm_ok:
        ok = run_step("Ensemble sweep (weekday) [promoted only]", [
            python, "scripts/ensemble_combo_sweep.py",
            "--asset", promoted_csv,
            "--min-dm", min_dm, "--max-dm", "8",
            "--model-suffix", "_weekday", "--day-filter", "weekday",
            "--kalshi-days", "14",
        ])
        if not ok:
            log("WARNING: weekday ensemble sweep failed")

    if we_xgb_ok and we_lstm_ok:
        ok = run_step("Ensemble sweep (weekend) [promoted only]", [
            python, "scripts/ensemble_combo_sweep.py",
            "--asset", promoted_csv,
            "--min-dm", min_dm, "--max-dm", "8",
            "--model-suffix", "_weekend", "--day-filter", "weekend",
            "--kalshi-days", "14",
        ])
        if not ok:
            log("WARNING: weekend ensemble sweep failed")

    total = time.time() - pipeline_start
    kept_msg = f" Kept: {','.join(kept_assets)}." if kept_assets else ""
    promoted_msg = f" Promoted: {','.join(promoted_assets)}."
    log("=" * 60)
    if failed:
        log(f"Done with warnings ({total:.0f}s). Failed steps: {failed}")
        send_telegram(
            f"Retrain (all) done with warnings ({total:.0f}s). Failed: {failed}.{promoted_msg}{kept_msg}"
        )
    else:
        log(f"Done ({total:.0f}s). All steps succeeded.")
        send_telegram(
            f"Retrain (all) complete ({total:.0f}s).{promoted_msg}{kept_msg} Bot will hot-reload."
        )
    log("Live bot will pick up new params at next window boundary.")
    log("=" * 60)

    return 1 if "generate" in failed or "train_xgb" in failed else 0


if __name__ == "__main__":
    sys.exit(main())
