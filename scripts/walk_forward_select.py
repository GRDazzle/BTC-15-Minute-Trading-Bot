"""Walk-forward model selection: pick hyperparams by average OOS PnL across time folds.

For each hyperparameter combo:
  1. Walk-forward train/test across K time folds
  2. Compute OOS PnL on each fold (using Kalshi prices + outcomes)
  3. Average OOS PnL across folds = the combo's score

Then:
  4. Pick the combo with best average OOS PnL
  5. Train final model on ALL data with those hyperparams
  6. Save model + feature list

This replaces both:
  - train_xgb.py's logloss-based model selection
  - The hyperparameter tuning step (Phase C)

Usage:
    python scripts/walk_forward_select.py --asset BTC,ETH,SOL,XRP --days 45
    python scripts/walk_forward_select.py --asset SOL --days 45 --n-folds 4
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import FEATURE_NAMES
from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker

TRAINING_DIR = PROJECT_ROOT / "ml" / "training_data"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
MODELS_DIR = PROJECT_ROOT / "models"

# Hyperparameter grid (same as Phase C but focused on the patterns that worked)
PARAM_GRID = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.08],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [3, 5, 10],
}


def load_feature_list(asset: str, stacked: bool = False) -> list[str]:
    """Load per-asset trimmed feature list if available.

    If stacked=True, appends 'lstm_p' to the feature list.
    """
    trim_path = MODELS_DIR / f"{asset.upper()}_v10_trimmed_features.json"
    if trim_path.exists():
        try:
            with open(trim_path) as f:
                data = json.load(f)
            features = data.get("features", [])
            if features:
                if stacked and "lstm_p" not in features:
                    features = features + ["lstm_p"]
                return features
        except Exception:
            pass
    features = list(FEATURE_NAMES)
    if stacked and "lstm_p" not in features:
        features = features + ["lstm_p"]
    return features


def load_data(asset: str, feature_list: list[str], stacked: bool = False):
    """Load training data CSV, return X, y, window_starts as arrays.

    If stacked=True, loads the _stacked.csv variant which has lstm_p column.
    """
    if stacked:
        path = TRAINING_DIR / f"{asset}_features_stacked.csv"
        if not path.exists():
            print(f"  WARNING: stacked CSV not found, falling back to non-stacked")
            path = TRAINING_DIR / f"{asset}_features.csv"
    else:
        path = TRAINING_DIR / f"{asset}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    X_rows, y_rows, ws_rows = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            X_rows.append([float(row.get(name, 0.0)) for name in feature_list])
            y_rows.append(int(float(row["label"])))
            ws_rows.append(row.get("window_start", ""))

    return np.array(X_rows), np.array(y_rows), ws_rows


def get_fold_boundaries(window_starts: list[str], n_folds: int, min_train_days: int = 14):
    """Create time-based fold boundaries for walk-forward validation.

    Returns list of (train_end_date, test_end_date) pairs.
    The first fold uses min_train_days of training data.
    Each subsequent fold extends the training window by one fold width.

    Example with 45 days, 4 folds, min_train_days=14:
      Fold 1: train days 1-14,  test days 15-22  (7 days test)
      Fold 2: train days 1-22,  test days 23-30
      Fold 3: train days 1-30,  test days 31-38
      Fold 4: train days 1-38,  test days 39-45
    """
    unique_dates = sorted(set(ws[:10] for ws in window_starts if ws))
    if not unique_dates:
        return []

    n_dates = len(unique_dates)
    # Reserve min_train_days for first training window
    test_pool = n_dates - min_train_days
    if test_pool < n_folds:
        # Not enough data for requested folds, use fewer
        n_folds = max(1, test_pool)

    fold_size = test_pool // n_folds

    folds = []
    for i in range(n_folds):
        train_end_idx = min_train_days + i * fold_size - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = min(test_start_idx + fold_size - 1, n_dates - 1)

        if test_start_idx >= n_dates:
            break

        train_end_date = unique_dates[train_end_idx]
        test_end_date = unique_dates[test_end_idx]
        folds.append((train_end_date, test_end_date))

    return folds


def simulate_pnl(proba, ws_test, asset, kalshi_idx, threshold=0.70):
    """Simulate PnL using Kalshi prices + outcomes. Returns (traded, wins, pnl)."""
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
    return traded, wins, pnl


def walk_forward_evaluate(X, y, ws, feature_list, folds, params, asset, kalshi_idx, threshold=0.70):
    """Train/test across all folds with given params, return average OOS PnL."""
    fold_pnls = []
    fold_wrs = []

    for train_end, test_end in folds:
        # Split by date
        train_mask = np.array([w[:10] <= train_end for w in ws])
        test_mask = np.array([w[:10] > train_end and w[:10] <= test_end for w in ws])

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        ws_test = [w for w, m in zip(ws, test_mask) if m]

        if len(X_train) < 100 or len(X_test) < 50:
            continue

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_list)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_list)

        xgb_params = {
            **params,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "seed": 42,
            "verbosity": 0,
        }

        model = xgb.train(
            xgb_params, dtrain, num_boost_round=400,
            evals=[(dtest, "test")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )
        proba = model.predict(dtest)
        traded, wins, pnl = simulate_pnl(proba, ws_test, asset, kalshi_idx, threshold)

        if traded > 0:
            fold_pnls.append(pnl)
            fold_wrs.append(wins / traded * 100)

    if not fold_pnls:
        return 0.0, 0.0, 0

    avg_pnl = sum(fold_pnls) / len(fold_pnls)
    avg_wr = sum(fold_wrs) / len(fold_wrs)
    return avg_pnl, avg_wr, len(fold_pnls)


def run_selection(assets, days, n_folds, threshold, stacked=False):
    keys = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    mode = "STACKED (XGB + lstm_p)" if stacked else "XGB only"
    print(f"Walk-forward model selection ({mode})")
    print(f"  Grid: {len(combos)} hyperparameter combos")
    print(f"  Folds: {n_folds}")
    print(f"  Threshold: {threshold}")
    print()

    for asset in assets:
        print(f"\n{'='*80}")
        print(f"  {asset}: Walk-Forward Model Selection ({mode})")
        print(f"{'='*80}")

        feature_list = load_feature_list(asset, stacked=stacked)
        print(f"  Features: {len(feature_list)}{' (includes lstm_p)' if stacked else ''}")

        X, y, ws = load_data(asset, feature_list, stacked=stacked)
        print(f"  Total rows: {len(X)}")

        # Get fold boundaries
        folds = get_fold_boundaries(ws, n_folds)
        print(f"  Folds: {len(folds)}")
        for i, (train_end, test_end) in enumerate(folds):
            n_train = sum(1 for w in ws if w[:10] <= train_end)
            n_test = sum(1 for w in ws if train_end < w[:10] <= test_end)
            print(f"    Fold {i+1}: train <= {train_end} ({n_train} rows), "
                  f"test {train_end} -> {test_end} ({n_test} rows)")

        # Load Kalshi data
        kalshi_idx = KalshiPollIndex(KALSHI_DIR, asset)
        print(f"  Kalshi events: {len(kalshi_idx.event_tickers)}")
        print()

        # Sweep all combos
        best_avg_pnl = -999999
        best_params = None
        best_avg_wr = 0
        results = []

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            avg_pnl, avg_wr, n_valid_folds = walk_forward_evaluate(
                X, y, ws, feature_list, folds, params, asset, kalshi_idx, threshold,
            )
            results.append((params, avg_pnl, avg_wr, n_valid_folds))

            if avg_pnl > best_avg_pnl and n_valid_folds >= len(folds) // 2:
                best_avg_pnl = avg_pnl
                best_params = dict(params)
                best_avg_wr = avg_wr

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(combos)} done... best avg PnL: ${best_avg_pnl:+.2f} ({best_avg_wr:.1f}% WR)")

        # Show top 10
        results.sort(key=lambda x: -x[1])
        print(f"\n  Top 10 combos (by average OOS PnL across {len(folds)} folds):")
        for params, avg_pnl, avg_wr, n_folds_valid in results[:10]:
            p_str = " ".join(f"{k}={params[k]}" for k in keys)
            print(f"    ${avg_pnl:>+8.2f} avg PnL  {avg_wr:>5.1f}% avg WR  folds={n_folds_valid}  {p_str}")

        if best_params is None:
            print(f"\n  ERROR: No valid combo found for {asset}")
            continue

        # Train final model on ALL data with best params
        print(f"\n  --- Training final model with best params ---")
        print(f"  Best avg OOS PnL: ${best_avg_pnl:+.2f} ({best_avg_wr:.1f}% WR)")
        for k in keys:
            print(f"    {k}: {best_params[k]}")

        dtrain_all = xgb.DMatrix(X, label=y, feature_names=feature_list)
        final_params = {
            **best_params,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "seed": 42,
            "verbosity": 0,
        }

        # Use 80/20 split for early stopping on the final model
        split_idx = int(len(X) * 0.85)
        X_tr, y_tr = X[:split_idx], y[:split_idx]
        X_es, y_es = X[split_idx:], y[split_idx:]
        dtrain_final = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_list)
        des = xgb.DMatrix(X_es, label=y_es, feature_names=feature_list)

        final_model = xgb.train(
            final_params, dtrain_final, num_boost_round=500,
            evals=[(des, "val")],
            early_stopping_rounds=25,
            verbose_eval=False,
        )

        # Evaluate on held-out portion
        proba_es = final_model.predict(des)
        acc = ((proba_es > 0.5).astype(int) == y_es).mean() * 100
        print(f"  Final model val accuracy: {acc:.1f}% (on last 15% of data)")

        # Save model
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"{asset}_xgb.json"
        final_model.save_model(str(model_path))
        print(f"  Model saved: {model_path}")

        # Copy to weekday/weekend paths (same model, no day-type split)
        import shutil
        for variant in ["_weekday", "_weekend"]:
            dst = MODELS_DIR / f"{asset}{variant}_xgb.json"
            shutil.copy2(model_path, dst)
        print(f"  Copied to weekday + weekend paths")

        # Save feature list + best params metadata
        meta = {
            "asset": asset,
            "features": feature_list,
            "n_features": len(feature_list),
            "best_params": {k: best_params[k] for k in keys},
            "best_avg_oos_pnl": round(best_avg_pnl, 2),
            "best_avg_oos_wr": round(best_avg_wr, 1),
            "n_folds": len(folds),
            "training_method": "walk_forward_selection",
        }
        meta_path = MODELS_DIR / f"{asset}_xgb_features.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Also save for weekday/weekend
        for variant in ["_weekday", "_weekend"]:
            dst = MODELS_DIR / f"{asset}{variant}_xgb_features.json"
            shutil.copy2(meta_path, dst)

        print(f"  Metadata saved: {meta_path}")

        # Feature importance
        importance = final_model.get_score(importance_type="gain")
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
        print(f"\n  Top 10 features:")
        for name, gain in sorted_imp[:10]:
            print(f"    {name:<30} {gain:>8.1f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Walk-forward model selection")
    parser.add_argument("--asset", required=True, help="Assets (comma-separated)")
    parser.add_argument("--days", type=int, default=45, help="Days of training data")
    parser.add_argument("--n-folds", type=int, default=4, help="Number of walk-forward folds")
    parser.add_argument("--threshold", type=float, default=0.70, help="Signal threshold for PnL sim")
    parser.add_argument("--stacked", action="store_true", help="Use stacked features (lstm_p from OOF)")
    args = parser.parse_args()
    assets = [a.strip().upper() for a in args.asset.split(",")]
    run_selection(assets, args.days, args.n_folds, args.threshold, stacked=args.stacked)


if __name__ == "__main__":
    main()
