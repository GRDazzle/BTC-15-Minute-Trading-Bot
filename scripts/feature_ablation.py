"""Feature ablation study: train models with each feature group removed.

For each of 13 feature groups, trains a model WITHOUT that group and
measures OOS PnL. Groups that hurt performance (removing them improves
PnL) should be dropped from the final model.

Usage:
    python scripts/feature_ablation.py --asset BTC,ETH,SOL,XRP --cutoff 2026-04-07
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import FEATURE_NAMES
from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker

TRAINING_DIR = PROJECT_ROOT / "ml" / "training_data"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"

# Feature groups for ablation
FEATURE_GROUPS = {
    "tick_velocity": ["velocity_30s", "velocity_60s", "velocity_300s", "velocity_900s"],
    "volatility": ["volatility_30s", "volatility_60s", "volatility_300s"],
    "volume": ["volume_30s", "volume_60s", "volume_180s", "buy_volume_ratio_60s", "buy_volume_ratio_300s"],
    "price_structure": ["vwap_deviation", "price_range_20", "aggressor_ratio_60s", "tick_intensity_30s", "large_trade_count"],
    "time": ["hour_sin", "hour_cos", "minute_in_window"],
    "momentum": ["return_skew_60s", "price_vs_open", "momentum_trend"],
    "sma": ["price_vs_sma5", "price_vs_sma15", "price_vs_sma_1h"],
    "market_condition": ["choppiness_60s", "range_pct_180s", "vol_acceleration"],
    "crash_detection": ["flips_per_tick_180s", "momentum_strength_180s"],
    "z_score": ["z_score_300s", "z_score_900s"],
    "obv_cvd": ["obv_slope_60s", "obv_slope_300s", "cvd_60s", "cvd_300s"],
    "stability_v10": ["price_stability_60s", "velocity_stability_60s", "direction_changes_60s", "range_vs_trend_60s"],
    "kalshi_v10": ["kalshi_yes_ask", "kalshi_spread", "kalshi_mid", "kalshi_mins_to_close"],
}


def load_data(asset, cutoff):
    """Load features, split by cutoff date."""
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


def train_and_eval(X_train, y_train, X_test, y_test, feature_names, feature_subset):
    """Train XGB on a feature subset, return test predictions."""
    # Get column indices for the subset
    indices = [feature_names.index(f) for f in feature_subset]
    X_tr = X_train[:, indices]
    X_te = X_test[:, indices]

    dtrain = xgb.DMatrix(X_tr, label=y_train, feature_names=feature_subset)
    dtest = xgb.DMatrix(X_te, label=y_test, feature_names=feature_subset)

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
        params, dtrain, num_boost_round=400,
        evals=[(dtest, "test")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    proba = model.predict(dtest)
    acc = ((proba > 0.5).astype(int) == y_test).mean() * 100
    return proba, acc, model


def simulate_pnl(proba, ws_test, y_test, asset, kalshi_idx, threshold=0.70):
    """Simulate PnL using Kalshi prices + outcomes."""
    window_preds = defaultdict(list)
    for i, ws in enumerate(ws_test):
        window_preds[ws].append({
            "p_bull": float(proba[i]),
            "actual": int(y_test[i]),
        })

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


def run_ablation(assets, cutoff, threshold=0.70):
    """Run the full ablation study."""
    all_feature_names = list(FEATURE_NAMES)

    for asset in assets:
        print(f"\n{'='*80}")
        print(f"  FEATURE ABLATION: {asset} (cutoff={cutoff}, threshold={threshold})")
        print(f"{'='*80}")

        X, y, ws, train_mask, test_mask = load_data(asset, cutoff)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        ws_test = [w for w, m in zip(ws, test_mask) if m]

        kalshi_idx = KalshiPollIndex(KALSHI_DIR, asset)

        print(f"  Train: {len(X_train)} rows, Test: {len(X_test)} rows")
        print()

        # Baseline: all 45 features
        proba_base, acc_base, _ = train_and_eval(
            X_train, y_train, X_test, y_test, all_feature_names, all_feature_names,
        )
        traded_b, wins_b, wr_b, pnl_b = simulate_pnl(
            proba_base, ws_test, y_test, asset, kalshi_idx, threshold,
        )
        print(f"  {'BASELINE (all 45)':<35} acc={acc_base:>5.1f}%  {traded_b:>4} trades  "
              f"{wr_b:>5.1f}% WR  ${pnl_b:>+8.2f}")
        print(f"  {'-'*75}")

        # Ablation: remove each group
        results = []
        for group_name, group_features in FEATURE_GROUPS.items():
            subset = [f for f in all_feature_names if f not in group_features]
            n_removed = len(all_feature_names) - len(subset)

            proba, acc, _ = train_and_eval(
                X_train, y_train, X_test, y_test, all_feature_names, subset,
            )
            traded, wins, wr, pnl = simulate_pnl(
                proba, ws_test, y_test, asset, kalshi_idx, threshold,
            )
            delta = pnl - pnl_b
            delta_wr = wr - wr_b

            marker = ""
            if delta > 5:
                marker = " << REMOVE (helps)"
            elif delta < -5:
                marker = " << KEEP (hurts to remove)"

            results.append({
                "group": group_name,
                "n_removed": n_removed,
                "acc": acc,
                "traded": traded,
                "wr": wr,
                "pnl": pnl,
                "delta_pnl": delta,
                "delta_wr": delta_wr,
            })

            print(f"  -{group_name:<33} acc={acc:>5.1f}%  {traded:>4} trades  "
                  f"{wr:>5.1f}% WR  ${pnl:>+8.2f}  (delta ${delta:>+7.2f}){marker}")

        # Summary
        print(f"\n  --- {asset} Summary ---")
        helps = [r for r in results if r["delta_pnl"] > 5]
        hurts = [r for r in results if r["delta_pnl"] < -5]
        neutral = [r for r in results if -5 <= r["delta_pnl"] <= 5]

        if helps:
            print(f"  Groups to REMOVE (removing improves PnL):")
            for r in sorted(helps, key=lambda x: -x["delta_pnl"]):
                print(f"    {r['group']:<30} delta=${r['delta_pnl']:>+7.2f}")
        if hurts:
            print(f"  Groups to KEEP (removing hurts PnL):")
            for r in sorted(hurts, key=lambda x: x["delta_pnl"]):
                print(f"    {r['group']:<30} delta=${r['delta_pnl']:>+7.2f}")
        if neutral:
            print(f"  Neutral groups (< $5 impact):")
            for r in neutral:
                print(f"    {r['group']:<30} delta=${r['delta_pnl']:>+7.2f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Feature ablation study")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--cutoff", default="2026-04-07")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()
    assets = [a.strip().upper() for a in args.asset.split(",")]
    run_ablation(assets, args.cutoff, args.threshold)


if __name__ == "__main__":
    main()
