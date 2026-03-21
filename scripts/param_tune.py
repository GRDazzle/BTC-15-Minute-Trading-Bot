"""
Rolling parameter tuner: find optimal (ml_weight, threshold) using recent
signal_log.csv data + Kalshi settlement outcomes.

Unlike pnl_sweep.py which needs BacktestSimulator + Binance aggTrades,
this script reads the live signal log directly -- every 10s checkpoint
already has ml_p, fusion_p, and Kalshi prices. We just need to match
outcomes and sweep combos.

Designed to run every 2 hours via the strategy's _param_tuning_loop().

Usage:
  python scripts/param_tune.py --hours 12
  python scripts/param_tune.py --hours 12 --min-windows 30
"""
import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader_kalshi import load_kalshi_windows

SIGNAL_LOG_PATH = PROJECT_ROOT / "output" / "signal_log.csv"
KALSHI_DATA_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "param_tune"

# Sweep grid (matches pnl_sweep.py)
ML_WEIGHTS = [round(w * 0.05, 2) for w in range(21)]  # 0.0, 0.05, ..., 1.0
THRESHOLDS = [round(0.55 + i * 0.01, 2) for i in range(16)]  # 0.55 to 0.70

KALSHI_FEE_CENTS = 2
MIN_WINDOWS_DEFAULT = 20


# -- Phase 1: Load signal checkpoints -----------------------------------------

def load_signal_checkpoints(
    hours: int,
) -> dict[str, dict[str, list[dict]]]:
    """Parse signal_log.csv and group by (asset, window_id).

    Returns:
        {asset: {window_id: [checkpoint_dicts]}}
        Each checkpoint: {dm, ml_p, fusion_p, yes_ask, no_ask}
    """
    if not SIGNAL_LOG_PATH.exists():
        print(f"Signal log not found: {SIGNAL_LOG_PATH}")
        return {}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    # asset -> window_id -> [checkpoints]
    result: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    total_rows = 0
    kept_rows = 0

    with open(SIGNAL_LOG_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1

            # Parse timestamp
            ts_str = row.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            if ts < cutoff:
                continue

            kept_rows += 1
            asset = row.get("asset", "").upper()
            window_id = row.get("window_id", "")
            if not asset or not window_id:
                continue

            try:
                checkpoint = {
                    "dm": int(row.get("dm", 0)),
                    "ml_p": float(row.get("ml_p", 0.5)),
                    "fusion_p": float(row.get("fusion_p", 0.5)),
                    "yes_ask": int(row.get("kalshi_yes_ask", 0)),
                    "no_ask": int(row.get("kalshi_no_ask", 0)),
                }
            except (ValueError, TypeError):
                continue

            result[asset][window_id].append(checkpoint)

    # Sort checkpoints by dm within each window
    for asset_windows in result.values():
        for window_id in asset_windows:
            asset_windows[window_id].sort(key=lambda c: c["dm"])

    print(f"Loaded {kept_rows}/{total_rows} signal rows (last {hours}h)")
    print(f"Assets: {sorted(result.keys())}")
    for asset in sorted(result.keys()):
        print(f"  {asset}: {len(result[asset])} windows")

    return dict(result)


# -- Phase 2: Match outcomes --------------------------------------------------

def match_outcomes(
    asset: str,
    signal_windows: dict[str, list[dict]],
) -> list[dict]:
    """Match signal_log windows to Kalshi settlement outcomes.

    Args:
        asset: e.g. "BTC"
        signal_windows: {window_id: [checkpoints]}

    Returns:
        List of enriched window dicts with outcome attached.
    """
    kalshi_windows = load_kalshi_windows(KALSHI_DATA_DIR, asset)
    if not kalshi_windows:
        print(f"  No Kalshi polling data for {asset}")
        return []

    enriched = []
    matched = 0
    unmatched = 0

    for window_id, checkpoints in signal_windows.items():
        # Convert window_id (start time) to close_time (start + 15 min)
        try:
            window_start = datetime.strptime(window_id, "%Y%m%d_%H%M")
            window_start = window_start.replace(tzinfo=timezone.utc)
            close_time = window_start + timedelta(minutes=15)
        except ValueError:
            unmatched += 1
            continue

        # Look up in Kalshi windows by close_time
        kw = kalshi_windows.get(close_time)
        if kw is None or kw.outcome is None:
            unmatched += 1
            continue

        matched += 1
        enriched.append({
            "window_id": window_id,
            "outcome": kw.outcome,
            "checkpoints": checkpoints,
        })

    print(f"  {asset}: {matched} matched, {unmatched} unmatched")
    return enriched


# -- Phase 3: Sweep combos ----------------------------------------------------

def load_config(asset: str) -> dict:
    """Load per-asset config from trading.json."""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        return {
            "initial_balance": 25.0,
            "max_contracts_per_trade": 10,
            "max_price_cents": 85,
            "min_price_cents": 15,
        }

    defaults = config.get("defaults", {})
    asset_cfg = config.get("assets", {}).get(asset.upper(), {})

    return {
        "initial_balance": asset_cfg.get("initial_balance", defaults.get("initial_balance", 25.0)),
        "max_contracts_per_trade": asset_cfg.get("max_contracts_per_trade", defaults.get("max_contracts_per_trade", 10)),
        "max_price_cents": asset_cfg.get("max_price_cents", defaults.get("max_price_cents", 85)),
        "min_price_cents": asset_cfg.get("min_price_cents", defaults.get("min_price_cents", 15)),
    }


def calculate_contracts(
    balance: float,
    price_cents: int,
    confidence: float,
    max_contracts: int,
) -> int:
    """Determine how many contracts to buy (matches pnl_sweep.py)."""
    cost_per = (price_cents + KALSHI_FEE_CENTS) / 100.0
    if cost_per <= 0 or balance <= 0:
        return 0
    max_by_balance = int(balance / cost_per)
    scale = min(1.0, confidence)
    desired = max(1, int(max_by_balance * scale))
    return min(desired, max_contracts)


def sweep_combo_pnl(
    ml_weight: float,
    threshold: float,
    enriched_windows: list[dict],
    config: dict,
) -> dict:
    """Evaluate one (ml_weight, threshold) combo by dollar PnL.

    For each window, find first checkpoint crossing the threshold,
    then simulate PnL using Kalshi prices from signal_log.
    """
    fusion_weight = 1.0 - ml_weight
    balance = config["initial_balance"]
    max_contracts = config["max_contracts_per_trade"]
    max_price = config["max_price_cents"]
    min_price = config["min_price_cents"]

    total_windows = len(enriched_windows)
    traded_count = 0
    win_count = 0
    total_pnl = 0.0
    skipped_kelly = 0
    skipped_no_ask = 0

    for win in enriched_windows:
        outcome = win["outcome"]
        if not outcome:
            continue

        predicted = None
        confidence = 0.0
        entry_price = 0
        side = ""

        for cp in win["checkpoints"]:
            ensemble_p = ml_weight * cp["ml_p"] + fusion_weight * cp["fusion_p"]

            if ensemble_p >= threshold:
                predicted = "BULLISH"
                confidence = ensemble_p
                side = "yes"
                entry_price = cp["yes_ask"]
                break
            elif ensemble_p <= 1.0 - threshold:
                predicted = "BEARISH"
                confidence = 1.0 - ensemble_p
                side = "no"
                entry_price = cp["no_ask"]
                break

        if predicted is None:
            continue

        # Validate entry price
        if entry_price <= 0 or entry_price >= 100:
            skipped_no_ask += 1
            continue

        # Kelly band filter
        if entry_price > max_price or entry_price < min_price:
            skipped_kelly += 1
            continue

        # Position sizing
        contracts = calculate_contracts(balance, entry_price, confidence, max_contracts)
        if contracts < 1:
            continue

        # Cost and fees
        cost = (entry_price / 100.0) * contracts
        fees = (KALSHI_FEE_CENTS / 100.0) * contracts
        balance -= (cost + fees)

        # Settlement
        won = (side == outcome)
        revenue = contracts * 1.00 if won else 0.0
        pnl = revenue - cost - fees
        balance += revenue

        traded_count += 1
        total_pnl += pnl
        if won:
            win_count += 1

    win_rate = win_count / traded_count * 100 if traded_count else 0.0
    avg_pnl = total_pnl / traded_count if traded_count else 0.0
    traded_pct = traded_count / total_windows * 100 if total_windows else 0.0

    return {
        "ml_weight": ml_weight,
        "threshold": threshold,
        "total_pnl": round(total_pnl, 4),
        "win_rate": round(win_rate, 1),
        "traded_count": traded_count,
        "traded_pct": round(traded_pct, 1),
        "win_count": win_count,
        "loss_count": traded_count - win_count,
        "avg_pnl_per_trade": round(avg_pnl, 4),
        "final_balance": round(balance, 2),
        "total_windows": total_windows,
        "skipped_kelly": skipped_kelly,
        "skipped_no_ask": skipped_no_ask,
    }


# -- Phase 4: Write config ----------------------------------------------------

def half_kelly_band(accuracy: float, fee_cents: int = KALSHI_FEE_CENTS) -> tuple[int, int]:
    """Compute entry price band using half-Kelly criterion.

    Returns (min_price_cents, max_price_cents).
    """
    p = accuracy / 100.0
    max_price = int(p * 100) - fee_cents
    min_price = int((1.0 - p) * 100) + fee_cents + 1

    max_price = max(50, min(95, max_price))
    min_price = max(5, min(50, min_price))

    return min_price, max_price


def write_tune_config(best_per_asset: dict[str, dict], min_dm: int) -> None:
    """Write tuned ensemble params to config/trading.json.

    Only updates ensemble section; preserves all other config.
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    else:
        config = {"defaults": {}, "assets": {}}

    for asset, best in best_per_asset.items():
        if asset not in config.get("assets", {}):
            config.setdefault("assets", {})[asset] = {}

        config["assets"][asset]["ensemble"] = {
            "ml_weight": best["ml_weight"],
            "threshold": best["threshold"],
            "min_dm": min_dm,
            "accuracy": round(best["win_rate"], 1),
            "net_correct": best["win_count"] - best["loss_count"],
            "traded_pct": best["traded_pct"],
            "pnl_sweep_total_pnl": best["total_pnl"],
            "pnl_sweep_source": "param_tune",
        }
        # Entry bands (min/max_price_cents) are set manually in trading.json
        # and validated against real Kalshi PnL data -- do NOT overwrite them here.

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"\n{'='*70}")
    print(f"  Params written to {CONFIG_PATH}")
    print(f"{'='*70}")
    print(f"  {'Asset':<5} | {'ML_W':>5} | {'Thresh':>6} | {'PnL':>9} | "
          f"{'WinRate':>7} | {'Trades':>6} | {'Kelly':>10}")
    print(f"  {'-'*65}")
    for asset, best in best_per_asset.items():
        pnl_sign = "+" if best["total_pnl"] >= 0 else ""
        min_p, max_p = half_kelly_band(best["win_rate"])
        print(f"  {asset:<5} | {best['ml_weight']:5.2f} | {best['threshold']:6.2f} | "
              f"{pnl_sign}${best['total_pnl']:7.2f} | {best['win_rate']:6.1f}% | "
              f"{best['traded_count']:6d} | {min_p:2d}c-{max_p:2d}c")
    print()


# -- Main ----------------------------------------------------------------------

def run_asset(
    asset: str,
    signal_windows: dict[str, list[dict]],
    min_windows: int,
    min_dm: int,
) -> dict | None:
    """Run param tuning for one asset. Returns best combo or None."""
    print(f"\n--- {asset} ---")

    # Phase 2: Match outcomes
    enriched_windows = match_outcomes(asset, signal_windows)
    if len(enriched_windows) < min_windows:
        print(f"  Only {len(enriched_windows)} settled windows "
              f"(need {min_windows}). Skipping.")
        return None

    # Load config for position sizing
    config = load_config(asset)

    total_combos = len(ML_WEIGHTS) * len(THRESHOLDS)
    print(f"  Sweeping {total_combos} combos over {len(enriched_windows)} windows...")

    # Phase 3: Sweep
    t0 = time.time()
    all_results = []
    for ml_w in ML_WEIGHTS:
        for thresh in THRESHOLDS:
            result = sweep_combo_pnl(ml_w, thresh, enriched_windows, config)
            all_results.append(result)
    t_sweep = time.time() - t0
    print(f"  Sweep completed in {t_sweep:.2f}s")

    # Filter: need at least some trades
    MIN_TRADE_RATE = 2.0
    qualifying = [r for r in all_results
                  if r["traded_pct"] >= MIN_TRADE_RATE and r["traded_count"] >= 5]

    if not qualifying:
        print(f"  No combos met the {MIN_TRADE_RATE:.0f}% trade rate filter!")
        qualifying = sorted(all_results, key=lambda r: r["total_pnl"], reverse=True)[:20]
    else:
        qualifying.sort(key=lambda r: r["total_pnl"], reverse=True)

    # Print top 10
    header = (
        f"  {'Rank':>4} | {'PnL':>9} | {'WinRate':>7} | "
        f"{'Traded':>6} | {'ML_W':>5} | {'Thresh':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, r in enumerate(qualifying[:10], 1):
        pnl_sign = "+" if r["total_pnl"] >= 0 else ""
        print(
            f"  {i:4d} | {pnl_sign}${r['total_pnl']:7.2f} | "
            f"{r['win_rate']:6.1f}% | {r['traded_count']:6d} | "
            f"{r['ml_weight']:5.2f} | {r['threshold']:6.2f}"
        )

    # Export CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{asset}_param_tune.csv"
    fieldnames = [
        "ml_weight", "threshold", "total_pnl", "win_rate",
        "traded_count", "traded_pct", "win_count", "loss_count",
        "avg_pnl_per_trade", "final_balance", "total_windows",
        "skipped_kelly", "skipped_no_ask",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: x["total_pnl"], reverse=True):
            writer.writerow(r)
    print(f"  Exported to {csv_path}")

    # Select best: highest total_pnl, tie-break on lower threshold
    if not qualifying:
        return None

    top_pnl = qualifying[0]["total_pnl"]
    tolerance = max(0.50, abs(top_pnl) * 0.05)
    near_top = [r for r in qualifying if r["total_pnl"] >= top_pnl - tolerance]
    near_top.sort(key=lambda r: (r["threshold"], -r["total_pnl"]))
    best = near_top[0]

    print(f"\n  Selected: ml_weight={best['ml_weight']:.2f} "
          f"threshold={best['threshold']:.2f} PnL=${best['total_pnl']:+.2f} "
          f"win_rate={best['win_rate']:.1f}% ({best['traded_count']} trades)")
    return best


def main():
    # Configure loguru
    logger.remove()
    log_path = PROJECT_ROOT / "logs" / "param_tune.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), mode="w", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(
        description="Rolling parameter tuner using signal_log.csv + Kalshi outcomes",
    )
    parser.add_argument(
        "--hours", type=int, default=12,
        help="Look back N hours in signal_log.csv (default: 12)",
    )
    parser.add_argument(
        "--min-windows", type=int, default=MIN_WINDOWS_DEFAULT,
        help=f"Minimum settled windows per asset (default: {MIN_WINDOWS_DEFAULT})",
    )
    parser.add_argument(
        "--min-dm", type=int, default=2,
        help="Minimum decision minute (default: 2)",
    )
    args = parser.parse_args()

    print(f"Rolling param tune: last {args.hours}h, min_windows={args.min_windows}")
    print(f"Signal log: {SIGNAL_LOG_PATH}")
    print(f"Kalshi data: {KALSHI_DATA_DIR}")

    # Phase 1: Load signal checkpoints
    t_start = time.time()
    all_checkpoints = load_signal_checkpoints(args.hours)
    if not all_checkpoints:
        print("\nNo signal data found. Run the bot to accumulate signal_log.csv.")
        sys.exit(1)

    # Run per-asset sweeps
    best_per_asset: dict[str, dict] = {}
    for asset, signal_windows in sorted(all_checkpoints.items()):
        best = run_asset(asset, signal_windows, args.min_windows, args.min_dm)
        if best is not None:
            best_per_asset[asset] = best

    # Phase 4: Write config
    if best_per_asset:
        write_tune_config(best_per_asset, args.min_dm)
    else:
        print("\nNo assets had sufficient data for param tuning.")
        print("Config NOT updated.")
        sys.exit(1)

    t_total = time.time() - t_start
    print(f"Total time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
