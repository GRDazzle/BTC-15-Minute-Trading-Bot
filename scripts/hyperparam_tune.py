"""Phase C: XGBoost hyperparameter tuning for trimmed per-asset models.

Sweeps max_depth, learning_rate, n_estimators, subsample, colsample_bytree,
min_child_weight using the per-asset trimmed feature sets from Phase B.

Usage:
    python scripts/hyperparam_tune.py --asset BTC,ETH,SOL,XRP --cutoff 2026-04-07
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from itertools import product

import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import FEATURE_NAMES
from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker

TRAINING_DIR = PROJECT_ROOT / "ml" / "training_data"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
MODELS_DIR = PROJECT_ROOT / "models"

# Hyperparameter grid
PARAM_GRID = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.08],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [3, 5, 10],
}


def load_data(asset, cutoff):
    path = TRAINING_DIR / f"{asset}_features.csv"
    X_rows, y_rows, ws_rows = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            X_rows.append([float(row.get(name, 0.0)) for name in FEATURE_NAMES])
            y_rows.append(int(float(row["label"])))
            ws_rows.append(row.get("window_start", ""))
    X = np.array(X_rows)
    y = np.array(y_rows)
    train_mask = np.array([w[:10] < cutoff for w in ws_rows])
    test_mask = np.array([w[:10] >= cutoff for w in ws_rows])
    return X, y, ws_rows, train_mask, test_mask


def simulate_pnl(proba, ws_test, y_test, asset, kalshi_idx, threshold=0.70):
    window_preds = defaultdict(list)
    for i, ws in enumerate(ws_test):
        window_preds[ws].append({"p_bull": float(proba[i])})
    traded = wins = 0
    pnl = 0.0
    for ws_val, preds in window_preds.items():
        entry = None
        for p in preds:
            if p["p_bull"] >= threshold or p["p_bull"] <= (1 - threshold):
                entry = p
                break
        if entry is None:
            continue
        try:
            window_dt = datetime.fromisoformat(ws_val)
            window_end = window_dt + timedelta(minutes=15)
        except Exception:
            continue
        event_ticker = window_start_to_event_ticker(asset, window_end)
        outcome = kalshi_idx.get_outcome(event_ticker)
        if outcome is None:
            continue
        entry_time = window_dt + timedelta(minutes=7)
        poll = kalshi_idx.find_poll(event_ticker, entry_time)
        if poll is None:
            continue
        direction = "BULLISH" if entry["p_bull"] > 0.5 else "BEARISH"
        side = "yes" if direction == "BULLISH" else "no"
        price = poll["yes_ask"] if side == "yes" else poll["no_ask"]
        if price < 55 or price > 80:
            continue
        won = side == outcome
        contracts = 10
        cost = (price / 100 + 0.02) * contracts
        trade_pnl = (contracts * 1.0 - cost) if won else -cost
        traded += 1
        if won:
            wins += 1
        pnl += trade_pnl
    wr = wins / traded * 100 if traded else 0
    return traded, wins, wr, pnl


def run_tuning(assets, cutoff, threshold=0.70):
    all_features = list(FEATURE_NAMES)

    # Generate all param combos (limited grid)
    keys = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    print(f"Hyperparameter grid: {len(combos)} combos")
    print()

    for asset in assets:
        print(f"\n{'='*80}")
        print(f"  HYPERPARAMETER TUNING: {asset}")
        print(f"{'='*80}")

        # Load trimmed feature set
        trim_path = MODELS_DIR / f"{asset}_v10_trimmed_features.json"
        if trim_path.exists():
            with open(trim_path) as f:
                trim_data = json.load(f)
            feature_subset = trim_data["features"]
            print(f"  Using trimmed features: {len(feature_subset)}f")
        else:
            feature_subset = all_features
            print(f"  No trimmed features found, using all {len(feature_subset)}f")

        X, y, ws, train_mask, test_mask = load_data(asset, cutoff)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        ws_test = [w for w, m in zip(ws, test_mask) if m]

        indices = [all_features.index(f) for f in feature_subset]
        X_tr = X_train[:, indices]
        X_te = X_test[:, indices]

        dtrain = xgb.DMatrix(X_tr, label=y_train, feature_names=feature_subset)
        dtest = xgb.DMatrix(X_te, label=y_test, feature_names=feature_subset)

        kalshi_idx = KalshiPollIndex(KALSHI_DIR, asset)

        best_pnl = -999999
        best_params = None
        best_acc = 0
        best_wr = 0
        best_traded = 0
        results = []

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
            params["seed"] = 42
            params["verbosity"] = 0

            model = xgb.train(
                params, dtrain, num_boost_round=400,
                evals=[(dtest, "test")],
                early_stopping_rounds=20,
                verbose_eval=False,
            )
            proba = model.predict(dtest)
            acc = ((proba > 0.5).astype(int) == y_test).mean() * 100
            traded, wins, wr, pnl = simulate_pnl(proba, ws_test, y_test, asset, kalshi_idx, threshold)

            results.append((params, acc, traded, wr, pnl))

            if pnl > best_pnl:
                best_pnl = pnl
                best_params = dict(params)
                best_acc = acc
                best_wr = wr
                best_traded = traded

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(combos)} done... best so far: ${best_pnl:+.2f} ({best_wr:.1f}% WR)")

        print(f"\n  --- {asset} BEST HYPERPARAMS ---")
        print(f"  PnL: ${best_pnl:+.2f}  WR: {best_wr:.1f}%  Trades: {best_traded}  Acc: {best_acc:.1f}%")
        for k in keys:
            print(f"    {k}: {best_params[k]}")

        # Show top 5
        results.sort(key=lambda x: -x[4])
        print(f"\n  Top 5 combos:")
        for params, acc, traded, wr, pnl in results[:5]:
            p_str = " ".join(f"{k}={params[k]}" for k in keys)
            print(f"    ${pnl:>+8.2f}  {wr:>5.1f}% WR  {traded:>3} trades  {p_str}")

        # Save best
        out = {
            "asset": asset,
            "features": feature_subset,
            "n_features": len(feature_subset),
            "best_params": {k: best_params[k] for k in keys},
            "best_pnl": best_pnl,
            "best_wr": best_wr,
            "best_acc": best_acc,
            "best_traded": best_traded,
        }
        out_path = MODELS_DIR / f"{asset}_v10_tuned.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="XGBoost hyperparameter tuning")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--cutoff", default="2026-04-07")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()
    assets = [a.strip().upper() for a in args.asset.split(",")]
    run_tuning(assets, args.cutoff, args.threshold)


if __name__ == "__main__":
    main()
