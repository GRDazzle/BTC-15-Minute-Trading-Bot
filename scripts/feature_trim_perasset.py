"""Per-asset feature trimming: remove groups that hurt each asset individually.

For each asset, removes all groups where delta_pnl > +$5 (removing the group
improved PnL), trains a trimmed model, and compares to the 45-feature baseline.

Then tries cumulative removal: greedily removes the most helpful group, retrains,
then removes the next most helpful, etc., stopping when removal starts hurting.

Usage:
    python scripts/feature_trim_perasset.py --asset BTC,ETH,SOL,XRP --cutoff 2026-04-07
"""
import argparse
import csv
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
MODELS_DIR = PROJECT_ROOT / "models"

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

# Phase A results: delta PnL when removing each group
ABLATION_DELTAS = {
    "BTC": {
        "tick_velocity": -20.2, "volatility": -11.7, "volume": -16.4,
        "price_structure": -21.1, "time": -20.9, "momentum": +20.2,
        "sma": -22.5, "market_condition": -20.7, "crash_detection": -14.0,
        "z_score": -20.2, "obv_cvd": -13.6, "stability_v10": -10.1,
        "kalshi_v10": -30.1,
    },
    "ETH": {
        "tick_velocity": +10.1, "volatility": -2.4, "volume": -1.1,
        "price_structure": +6.5, "time": +15.7, "momentum": +21.5,
        "sma": +35.5, "market_condition": +11.7, "crash_detection": -9.9,
        "z_score": +6.4, "obv_cvd": -3.3, "stability_v10": +0.4,
        "kalshi_v10": +8.4,
    },
    "SOL": {
        "tick_velocity": +6.4, "volatility": -11.0, "volume": +3.0,
        "price_structure": -5.2, "time": -0.3, "momentum": -19.1,
        "sma": -24.0, "market_condition": -1.7, "crash_detection": -3.1,
        "z_score": +10.0, "obv_cvd": -0.6, "stability_v10": -0.3,
        "kalshi_v10": +3.4,
    },
    "XRP": {
        "tick_velocity": +10.1, "volatility": +5.1, "volume": +8.7,
        "price_structure": +6.1, "time": +8.2, "momentum": +11.0,
        "sma": +2.0, "market_condition": +0.4, "crash_detection": +13.5,
        "z_score": +8.5, "obv_cvd": +14.1, "stability_v10": +13.5,
        "kalshi_v10": -5.0,
    },
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


def train_and_eval(X_train, y_train, X_test, y_test, all_features, subset):
    indices = [all_features.index(f) for f in subset]
    X_tr = X_train[:, indices]
    X_te = X_test[:, indices]
    dtrain = xgb.DMatrix(X_tr, label=y_train, feature_names=subset)
    dtest = xgb.DMatrix(X_te, label=y_test, feature_names=subset)
    params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 5, "seed": 42,
        "verbosity": 0,
    }
    model = xgb.train(
        params, dtrain, num_boost_round=400,
        evals=[(dtest, "test")], early_stopping_rounds=20,
        verbose_eval=False,
    )
    proba = model.predict(dtest)
    acc = ((proba > 0.5).astype(int) == y_test).mean() * 100
    return proba, acc, model


def simulate_pnl(proba, ws_test, y_test, asset, kalshi_idx, threshold=0.70):
    window_preds = defaultdict(list)
    for i, ws in enumerate(ws_test):
        window_preds[ws].append({"p_bull": float(proba[i]), "actual": int(y_test[i])})

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


def run_greedy_trim(assets, cutoff, threshold=0.70):
    all_features = list(FEATURE_NAMES)

    for asset in assets:
        print(f"\n{'='*80}")
        print(f"  GREEDY FEATURE TRIM: {asset}")
        print(f"{'='*80}")

        X, y, ws, train_mask, test_mask = load_data(asset, cutoff)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        ws_test = [w for w, m in zip(ws, test_mask) if m]
        kalshi_idx = KalshiPollIndex(KALSHI_DIR, asset)

        # Baseline
        proba_b, acc_b, _ = train_and_eval(X_train, y_train, X_test, y_test, all_features, all_features)
        t_b, w_b, wr_b, pnl_b = simulate_pnl(proba_b, ws_test, y_test, asset, kalshi_idx, threshold)
        print(f"\n  Baseline (45 features): acc={acc_b:.1f}%  {t_b} trades  {wr_b:.1f}% WR  ${pnl_b:+.2f}")

        # Get groups sorted by delta (highest first = most helpful to remove)
        deltas = ABLATION_DELTAS[asset]
        candidates = sorted(
            [(g, d) for g, d in deltas.items() if d > 0],
            key=lambda x: -x[1],
        )

        if not candidates:
            print(f"  No groups to remove (all groups help {asset})")
            print(f"  Best model = baseline (45 features)")
            continue

        print(f"\n  Candidate groups to remove (delta > $0):")
        for g, d in candidates:
            print(f"    {g:<25} delta=${d:+.2f}")

        # Greedy removal
        print(f"\n  --- Greedy removal ---")
        current_features = list(all_features)
        removed_groups = []
        best_pnl = pnl_b
        best_features = list(all_features)
        best_removed = []

        for group_name, delta in candidates:
            group_feats = FEATURE_GROUPS[group_name]
            trial_features = [f for f in current_features if f not in group_feats]

            proba, acc, model = train_and_eval(
                X_train, y_train, X_test, y_test, all_features, trial_features,
            )
            traded, wins, wr, pnl = simulate_pnl(
                proba, ws_test, y_test, asset, kalshi_idx, threshold,
            )
            cum_delta = pnl - pnl_b

            marker = ""
            if pnl > best_pnl:
                marker = " << NEW BEST"
                best_pnl = pnl
                best_features = list(trial_features)
                best_removed = list(removed_groups) + [group_name]

            print(f"  Remove {group_name:<25} -> {len(trial_features):>2}f  acc={acc:.1f}%  "
                  f"{traded:>3} trades  {wr:.1f}% WR  ${pnl:>+8.2f}  (vs base ${cum_delta:>+7.2f}){marker}")

            if pnl >= best_pnl - 5:  # Allow small regression to explore further
                current_features = trial_features
                removed_groups.append(group_name)
            else:
                print(f"    (stopped: removing {group_name} degraded too much)")
                break

        # Final summary
        print(f"\n  --- {asset} BEST MODEL ---")
        print(f"  Features: {len(best_features)} (removed: {', '.join(best_removed) if best_removed else 'none'})")
        print(f"  PnL: ${best_pnl:+.2f} (baseline: ${pnl_b:+.2f}, improvement: ${best_pnl - pnl_b:+.2f})")

        # Save best feature list
        out_path = MODELS_DIR / f"{asset}_v10_trimmed_features.json"
        import json
        with open(out_path, "w") as f:
            json.dump({
                "asset": asset,
                "features": best_features,
                "removed_groups": best_removed,
                "n_features": len(best_features),
                "baseline_pnl": pnl_b,
                "trimmed_pnl": best_pnl,
            }, f, indent=2)
        print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-asset greedy feature trimming")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--cutoff", default="2026-04-07")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()
    assets = [a.strip().upper() for a in args.asset.split(",")]
    run_greedy_trim(assets, args.cutoff, args.threshold)


if __name__ == "__main__":
    main()
