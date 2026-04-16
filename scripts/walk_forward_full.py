"""Full combo walk-forward: sweep XGB hyperparams + min_dm + threshold + max_price.

All parameters are evaluated together through 4-fold walk-forward,
scored by average OOS PnL. This replaces the separate hyperparam tuning
and ensemble sweep steps with a single unified optimization.

Grid:
  XGB: max_depth(3) x lr(3) x subsample(3) x colsample(3) x min_child(3) = 243
  DM:  min_dm(3) = [2, 3, 4]
  Total model trains: 729 combos x 4 folds x 4 assets = 11,664

  Threshold and max_price don't require retraining — they're swept on
  the predictions from each trained model, so they're "free":
  threshold(3) x max_price(3) = 9 PnL sims per model = no extra cost

Usage:
    python scripts/walk_forward_full.py --asset BTC,ETH,SOL,XRP
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

# XGB hyperparameter grid
XGB_GRID = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.08],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [3, 5, 10],
}

# Execution parameter grid (swept on predictions, no retraining needed)
MIN_DM_VALUES = [2, 3, 4]
THRESHOLD_VALUES = [0.62, 0.65, 0.70, 0.75]
MAX_PRICE_VALUES = [70, 75, 80]


def load_feature_list(asset: str) -> list[str]:
    trim_path = MODELS_DIR / f"{asset.upper()}_v10_trimmed_features.json"
    if trim_path.exists():
        try:
            with open(trim_path) as f:
                features = json.load(f).get("features", [])
            if features:
                if "lstm_p" not in features:
                    features = features + ["lstm_p"]
                return features
        except Exception:
            pass
    features = list(FEATURE_NAMES) + ["lstm_p"]
    return features


def load_data(asset: str, feature_list: list[str]):
    path = TRAINING_DIR / f"{asset}_features_stacked.csv"
    if not path.exists():
        path = TRAINING_DIR / f"{asset}_features.csv"
    X_rows, y_rows, ws_rows, dm_rows = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            X_rows.append([float(row.get(name, 0.0)) for name in feature_list])
            y_rows.append(int(float(row["label"])))
            ws_rows.append(row.get("window_start", ""))
            dm_rows.append(int(float(row.get("minute_in_window", 0))))
    return np.array(X_rows), np.array(y_rows), np.array(ws_rows), np.array(dm_rows)


def get_fold_boundaries(ws, n_folds, min_train_days=14):
    unique_dates = sorted(set(w[:10] for w in ws if w))
    n_dates = len(unique_dates)
    test_pool = n_dates - min_train_days
    if test_pool < n_folds:
        n_folds = max(1, test_pool)
    fold_size = test_pool // n_folds
    folds = []
    for i in range(n_folds):
        train_end_idx = min_train_days + i * fold_size - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = min(test_start_idx + fold_size - 1, n_dates - 1)
        if test_start_idx >= n_dates:
            break
        folds.append((unique_dates[train_end_idx], unique_dates[test_end_idx]))
    return folds


def simulate_pnl(proba, ws_test, dm_test, asset, kalshi_idx, threshold, max_price, min_dm):
    """Simulate PnL with given threshold + max_price + min_dm filter."""
    window_preds = defaultdict(list)
    for i in range(len(proba)):
        if dm_test[i] < min_dm:
            continue
        ws = ws_test[i]
        window_preds[ws].append({"p_bull": float(proba[i]), "dm": int(dm_test[i])})

    traded = wins = 0
    pnl = 0.0
    for ws_val, preds in window_preds.items():
        entry = None
        for p in sorted(preds, key=lambda x: x["dm"]):
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
        entry_time = window_dt + timedelta(minutes=5 + entry["dm"])
        poll = kalshi_idx.find_poll(event_ticker, entry_time)
        if poll is None:
            continue
        direction = "BULLISH" if entry["p_bull"] > 0.5 else "BEARISH"
        side = "yes" if direction == "BULLISH" else "no"
        price = poll["yes_ask"] if side == "yes" else poll["no_ask"]
        if price < 55 or price > max_price:
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


def run_full_sweep(assets, n_folds=4):
    xgb_keys = list(XGB_GRID.keys())
    xgb_combos = list(product(*[XGB_GRID[k] for k in xgb_keys]))
    total_xgb = len(xgb_combos)
    total_dm = len(MIN_DM_VALUES)
    exec_combos = len(THRESHOLD_VALUES) * len(MAX_PRICE_VALUES)

    print(f"Full combo walk-forward sweep")
    print(f"  XGB grid: {total_xgb} combos")
    print(f"  min_dm: {MIN_DM_VALUES}")
    print(f"  threshold: {THRESHOLD_VALUES}")
    print(f"  max_price: {MAX_PRICE_VALUES}")
    print(f"  Per-asset: {total_xgb} XGB x {total_dm} dm x {n_folds} folds = {total_xgb * total_dm * n_folds} model trains")
    print(f"  Per model: {exec_combos} threshold x max_price sims (free)")
    print()

    for asset in assets:
        print(f"\n{'='*80}")
        print(f"  {asset}: Full Combo Walk-Forward")
        print(f"{'='*80}")

        feature_list = load_feature_list(asset)
        print(f"  Features: {len(feature_list)}")

        X, y, ws, dms = load_data(asset, feature_list)
        print(f"  Total rows: {len(X)}")

        folds = get_fold_boundaries(ws, n_folds)
        print(f"  Folds: {len(folds)}")
        for i, (te, ts) in enumerate(folds):
            print(f"    Fold {i+1}: train <= {te}, test -> {ts}")

        kalshi_idx = KalshiPollIndex(KALSHI_DIR, asset)
        print(f"  Kalshi events: {len(kalshi_idx.event_tickers)}")
        print()

        best_score = -999999
        best_config = None
        n_done = 0
        total = total_xgb * total_dm

        for min_dm in MIN_DM_VALUES:
            for xgb_combo in xgb_combos:
                xgb_params = dict(zip(xgb_keys, xgb_combo))
                xgb_params["objective"] = "binary:logistic"
                xgb_params["eval_metric"] = "logloss"
                xgb_params["seed"] = 42
                xgb_params["verbosity"] = 0

                # Walk-forward: train/predict across folds
                fold_predictions = []  # list of (proba, ws_test, dm_test) per fold

                for train_end, test_end in folds:
                    # Filter by min_dm for training
                    train_mask = (np.array([w[:10] <= train_end for w in ws])) & (dms >= min_dm)
                    test_mask = (np.array([w[:10] > train_end for w in ws])) & (np.array([w[:10] <= test_end for w in ws]))

                    X_train, y_train = X[train_mask], y[train_mask]
                    X_test, y_test = X[test_mask], y[test_mask]
                    ws_test = ws[test_mask]
                    dm_test = dms[test_mask]

                    if len(X_train) < 100 or len(X_test) < 50:
                        continue

                    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_list)
                    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_list)

                    model = xgb.train(
                        xgb_params, dtrain, num_boost_round=400,
                        evals=[(dtest, "test")],
                        early_stopping_rounds=20,
                        verbose_eval=False,
                    )
                    proba = model.predict(dtest)
                    fold_predictions.append((proba, ws_test, dm_test))

                if not fold_predictions:
                    n_done += 1
                    continue

                # Sweep threshold x max_price on the predictions (free — no retraining)
                for threshold in THRESHOLD_VALUES:
                    for max_price in MAX_PRICE_VALUES:
                        fold_pnls = []
                        fold_wrs = []
                        fold_trades = []

                        for proba, ws_test, dm_test in fold_predictions:
                            traded, wins, wr, pnl = simulate_pnl(
                                proba, ws_test, dm_test, asset, kalshi_idx,
                                threshold, max_price, min_dm,
                            )
                            if traded > 0:
                                fold_pnls.append(pnl)
                                fold_wrs.append(wr)
                                fold_trades.append(traded)

                        if not fold_pnls:
                            continue

                        avg_pnl = sum(fold_pnls) / len(fold_pnls)
                        avg_wr = sum(fold_wrs) / len(fold_wrs)
                        avg_trades = sum(fold_trades) / len(fold_trades)

                        # Require profitable in ALL folds — not just high average
                        all_positive = all(p > 0 for p in fold_pnls)
                        if (all_positive
                                and avg_pnl > best_score
                                and len(fold_pnls) >= len(folds)):
                            best_score = avg_pnl
                            best_config = {
                                **{k: xgb_params[k] for k in xgb_keys},
                                "min_dm": min_dm,
                                "threshold": threshold,
                                "max_price": max_price,
                                "avg_pnl": round(avg_pnl, 2),
                                "min_fold_pnl": round(min(fold_pnls), 2),
                                "avg_wr": round(avg_wr, 1),
                                "avg_trades": round(avg_trades, 1),
                                "n_valid_folds": len(fold_pnls),
                                "fold_pnls": [round(p, 2) for p in fold_pnls],
                            }

                n_done += 1
                if n_done % 50 == 0:
                    print(f"  {n_done}/{total} done... best: ${best_score:+.2f} ({best_config['avg_wr']:.1f}% WR, dm>={best_config['min_dm']}, thresh={best_config['threshold']}, maxP={best_config['max_price']})")

        if best_config is None:
            print(f"  ERROR: No valid combo found for {asset}")
            continue

        # Report top result
        print(f"\n  {'='*60}")
        print(f"  {asset} BEST CONFIG (avg across {best_config['n_valid_folds']} folds):")
        print(f"  {'='*60}")
        print(f"  Avg OOS PnL:  ${best_config['avg_pnl']:+.2f}")
        print(f"  Avg OOS WR:   {best_config['avg_wr']:.1f}%")
        print(f"  Avg trades:   {best_config['avg_trades']:.0f}/fold")
        print(f"  XGB params:")
        for k in xgb_keys:
            print(f"    {k}: {best_config[k]}")
        print(f"  Execution:")
        print(f"    min_dm:     {best_config['min_dm']}")
        print(f"    threshold:  {best_config['threshold']}")
        print(f"    max_price:  {best_config['max_price']}c")

        # Train final model on ALL data with best params
        print(f"\n  Training final model on all data...")
        final_min_dm = best_config["min_dm"]
        dm_mask = dms >= final_min_dm

        X_all = X[dm_mask]
        y_all = y[dm_mask]

        split = int(len(X_all) * 0.85)
        dtrain = xgb.DMatrix(X_all[:split], label=y_all[:split], feature_names=feature_list)
        dval = xgb.DMatrix(X_all[split:], label=y_all[split:], feature_names=feature_list)

        final_params = {
            **{k: best_config[k] for k in xgb_keys},
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "seed": 42,
            "verbosity": 0,
        }
        final_model = xgb.train(
            final_params, dtrain, num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=25,
            verbose_eval=False,
        )

        proba_val = final_model.predict(dval)
        acc = ((proba_val > 0.5).astype(int) == y_all[split:]).mean() * 100
        print(f"  Final val accuracy: {acc:.1f}%")

        # Save model
        import shutil
        model_path = MODELS_DIR / f"{asset}_xgb.json"
        final_model.save_model(str(model_path))
        for variant in ["_weekday", "_weekend"]:
            shutil.copy2(model_path, MODELS_DIR / f"{asset}{variant}_xgb.json")

        # Save metadata
        meta = {
            "asset": asset,
            "features": feature_list,
            "n_features": len(feature_list),
            "best_params": {k: best_config[k] for k in xgb_keys},
            "min_dm": best_config["min_dm"],
            "threshold": best_config["threshold"],
            "max_price": best_config["max_price"],
            "best_avg_oos_pnl": best_config["avg_pnl"],
            "best_avg_oos_wr": best_config["avg_wr"],
            "avg_trades_per_fold": best_config["avg_trades"],
            "n_folds": best_config["n_valid_folds"],
            "training_method": "walk_forward_full_combo",
        }
        meta_path = MODELS_DIR / f"{asset}_xgb_features.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        for variant in ["_weekday", "_weekend"]:
            shutil.copy2(meta_path, MODELS_DIR / f"{asset}{variant}_xgb_features.json")

        # Update config/trading.json with best execution params
        config_path = PROJECT_ROOT / "config" / "trading.json"
        with open(config_path) as f:
            cfg = json.load(f)
        asset_cfg = cfg.setdefault("assets", {}).setdefault(asset, {})
        ens = asset_cfg.setdefault("ensemble", {})
        ens["threshold"] = best_config["threshold"]
        ens["max_price_cents"] = best_config["max_price"]
        ens["min_dm"] = best_config["min_dm"]
        ens["walk_forward_pnl"] = best_config["avg_pnl"]
        ens["walk_forward_wr"] = best_config["avg_wr"]
        asset_cfg["max_price_cents"] = best_config["max_price"]
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")

        # Feature importance
        importance = final_model.get_score(importance_type="gain")
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
        print(f"\n  Top 10 features:")
        for name, gain in sorted_imp[:10]:
            marker = " *" if name == "lstm_p" else ""
            print(f"    {name:<30} {gain:>8.1f}{marker}")

        print(f"\n  Saved: {model_path}, {meta_path}, config updated")
        print()


def main():
    parser = argparse.ArgumentParser(description="Full combo walk-forward sweep")
    parser.add_argument("--asset", required=True, help="Assets (comma-separated)")
    parser.add_argument("--n-folds", type=int, default=4, help="Walk-forward folds")
    args = parser.parse_args()
    assets = [a.strip().upper() for a in args.asset.split(",")]
    run_full_sweep(assets, args.n_folds)


if __name__ == "__main__":
    main()
