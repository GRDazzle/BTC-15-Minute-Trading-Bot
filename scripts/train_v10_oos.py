"""Train v10 XGBoost models and run OOS test against this week's Kalshi data.

Splits training data by date:
  - Train: everything before the cutoff date (default: 2026-04-07)
  - Test:  cutoff date and after

Reports:
  - Train/test accuracy
  - Feature importance (top 15)
  - OOS PnL simulation using Kalshi polling data for prices + outcomes

Usage:
    python scripts/train_v10_oos.py --asset SOL
    python scripts/train_v10_oos.py --asset BTC,ETH,SOL,XRP --cutoff 2026-04-07
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import FEATURE_NAMES
from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker

TRAINING_DIR = PROJECT_ROOT / "ml" / "training_data"
MODELS_DIR = PROJECT_ROOT / "models"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"


def load_features(asset: str, min_dm: int = 2):
    """Load feature CSV, return (X, y, window_starts, dms)."""
    path = TRAINING_DIR / f"{asset}_features.csv"
    if not path.exists():
        print(f"  ERROR: {path} not found")
        return None, None, None, None

    X_rows = []
    y_rows = []
    ws_rows = []
    dm_rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dm = int(float(row.get("minute_in_window", 0)))
            # Approximate dm from hour_utc and window_start isn't reliable
            # Use the feature value directly
            label = int(float(row["label"]))
            window_start = row.get("window_start", "")

            feats = [float(row.get(name, 0.0)) for name in FEATURE_NAMES]
            X_rows.append(feats)
            y_rows.append(label)
            ws_rows.append(window_start)
            dm_rows.append(dm)

    return np.array(X_rows), np.array(y_rows), ws_rows, dm_rows


def train_and_evaluate(asset: str, cutoff_date: str, min_dm: int = 2):
    """Train on pre-cutoff data, evaluate on post-cutoff data."""
    import xgboost as xgb

    print(f"\n{'='*70}")
    print(f"  {asset} — v10 OOS Test (cutoff: {cutoff_date})")
    print(f"{'='*70}")

    X, y, ws, dms = load_features(asset, min_dm)
    if X is None:
        return

    # Split by date
    train_mask = np.array([w[:10] < cutoff_date for w in ws])
    test_mask = np.array([w[:10] >= cutoff_date for w in ws])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    ws_test = [w for w, m in zip(ws, test_mask) if m]
    dms_test = [d for d, m in zip(dms, test_mask) if m]

    print(f"  Train: {len(X_train)} rows ({train_mask.sum()} windows pre-{cutoff_date})")
    print(f"  Test:  {len(X_test)} rows ({test_mask.sum()} windows {cutoff_date}+)")

    if len(X_train) < 100 or len(X_test) < 50:
        print("  SKIP: insufficient data")
        return

    # Train XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "seed": 42,
        "verbosity": 0,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"{asset}_v10_xgb.json"
    model.save_model(str(model_path))
    print(f"  Model saved: {model_path}")

    # Train accuracy
    train_pred = (model.predict(dtrain) > 0.5).astype(int)
    train_acc = (train_pred == y_train).mean() * 100

    # Test accuracy
    test_proba = model.predict(dtest)
    test_pred = (test_proba > 0.5).astype(int)
    test_acc = (test_pred == y_test).mean() * 100

    print(f"\n  Train accuracy: {train_acc:.1f}%")
    print(f"  Test accuracy:  {test_acc:.1f}%")

    # Feature importance (top 15)
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
    print(f"\n  Top 15 features:")
    for name, gain in sorted_imp[:15]:
        marker = " *" if name.startswith("kalshi_") or name.startswith("price_stability") or name.startswith("velocity_stability") or name.startswith("direction_changes") or name.startswith("range_vs_trend") else ""
        print(f"    {name:<30} {gain:>8.1f}{marker}")

    # ---- OOS PnL simulation ----
    print(f"\n  --- OOS PnL Simulation (Kalshi data, {cutoff_date}+) ---")

    # Load Kalshi outcomes for the test period
    kalshi_idx = KalshiPollIndex(KALSHI_DIR, asset, min_date=cutoff_date)
    print(f"  Kalshi events loaded: {len(kalshi_idx.event_tickers)}")

    # Group test predictions by window
    y_test_arr = y_test  # already sliced by train_and_evaluate
    window_preds = defaultdict(list)
    for i, (ws_val, dm_val) in enumerate(zip(ws_test, dms_test)):
        window_preds[ws_val].append({
            "p_bull": float(test_proba[i]),
            "actual": int(y_test_arr[i]),
            "dm": dm_val,
        })

    # Simulate trading at different thresholds
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        traded = 0
        wins = 0
        pnl = 0.0
        for ws_val, preds in window_preds.items():
            # Use first prediction that passes threshold (earliest dm)
            entry = None
            for pred in sorted(preds, key=lambda x: x["dm"]):
                if pred["p_bull"] >= threshold or pred["p_bull"] <= (1 - threshold):
                    entry = pred
                    break
            if entry is None:
                continue

            # Get Kalshi outcome
            try:
                window_dt = datetime.fromisoformat(ws_val)
                window_end = window_dt + timedelta(minutes=15)
            except Exception:
                continue

            event_ticker = window_start_to_event_ticker(asset, window_end)
            outcome = kalshi_idx.get_outcome(event_ticker)
            if outcome is None:
                continue

            # Get Kalshi price at entry time
            entry_time = window_dt + timedelta(minutes=5) + timedelta(minutes=entry["dm"])
            poll = kalshi_idx.find_poll(event_ticker, entry_time)
            if poll is None:
                continue

            direction = "BULLISH" if entry["p_bull"] > 0.5 else "BEARISH"
            side = "yes" if direction == "BULLISH" else "no"

            if side == "yes":
                price = poll["yes_ask"]
            else:
                price = poll["no_ask"]

            # Price band filter (same as live: 55-85c)
            if price < 55 or price > 85:
                continue

            won = side == outcome
            contracts = 10
            cost = (price / 100 + 0.02) * contracts
            trade_pnl = (contracts * 1.0 - cost) if won else -cost

            traded += 1
            if won:
                wins += 1
            pnl += trade_pnl

        wr = wins / traded * 100 if traded > 0 else 0
        print(f"    threshold={threshold:.2f}: {traded:>4} trades, {wins:>4}W, {wr:>5.1f}% WR, ${pnl:>+8.2f} PnL")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train v10 models + OOS test")
    parser.add_argument("--asset", required=True, help="Assets (comma-separated)")
    parser.add_argument("--cutoff", default="2026-04-07", help="OOS cutoff date (default: 2026-04-07)")
    parser.add_argument("--min-dm", type=int, default=2, help="Min decision minute")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    for asset in assets:
        train_and_evaluate(asset, args.cutoff, args.min_dm)


if __name__ == "__main__":
    main()
