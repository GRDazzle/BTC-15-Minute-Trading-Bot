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

TRADE_LOG_SUFFIX = ""  # set by --trade-log-suffix CLI flag


def load_feature_list(asset: str) -> list[str]:
    trim_path = MODELS_DIR / f"{asset.upper()}_v10_trimmed_features.json"
    if trim_path.exists():
        try:
            with open(trim_path) as f:
                features = json.load(f).get("features", [])
            if features:
                # Un-stacked: strip lstm_p if present from trimmed list
                features = [f for f in features if f != "lstm_p"]
                return features
        except Exception:
            pass
    return list(FEATURE_NAMES)


def load_data(asset: str, feature_list: list[str]):
    # Un-stacked training: use raw feature CSV (without lstm_p OOF predictions)
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


def simulate_pnl(proba, ws_test, dm_test, asset, kalshi_idx, threshold, max_price, min_dm,
                 trade_log: list | None = None, fold_idx: int | None = None):
    """Simulate PnL with given threshold + max_price + min_dm filter.

    Returns (traded, wins, wr, pnl, per_hour_stats) where per_hour_stats is
    {hour: {"n": int, "wins": int, "pnl": float}}.

    If trade_log is provided, per-trade dicts are appended for offline analysis.
    """
    # Preserve CSV row order within each window (multiple sub-minute rows per
    # integer dm exist; their order in the CSV is chronological). Tracks the
    # original row index so we can infer sub-minute timing for entry_time.
    window_preds = defaultdict(list)
    for i in range(len(proba)):
        if dm_test[i] < min_dm:
            continue
        ws = ws_test[i]
        window_preds[ws].append({
            "p_bull": float(proba[i]),
            "dm": int(dm_test[i]),
            "_row_i": i,  # global row index (chronological within window since CSV is sorted)
        })

    traded = wins = 0
    pnl = 0.0
    per_hour = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for ws_val, preds in window_preds.items():
        # Sort by (dm, _row_i) — stable but makes chronological ordering explicit.
        # Walk in order; first row with p crossing threshold wins (matches live's
        # "first qualifying 2s check" cadence at the granularity available in CSV).
        preds_sorted = sorted(preds, key=lambda x: (x["dm"], x["_row_i"]))
        entry = None
        entry_idx_in_dm = 0  # which check-within-minute (0 = first)
        dm_group_size = 1
        for p in preds_sorted:
            if p["p_bull"] >= threshold or p["p_bull"] <= (1 - threshold):
                entry = p
                same_dm = [q for q in preds_sorted if q["dm"] == p["dm"]]
                dm_group_size = max(1, len(same_dm))
                entry_idx_in_dm = same_dm.index(p)
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
        # Sub-minute entry_time: if there are N checks within this dm and we
        # fired on the k-th, approximate the within-minute offset as k/N * 60s.
        sub_minute_offset = int(60 * entry_idx_in_dm / dm_group_size)
        entry_time = (window_dt + timedelta(minutes=5 + entry["dm"])
                      + timedelta(seconds=sub_minute_offset))
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
        # Per-hour bucket — use entry_time hour (UTC)
        hr = entry_time.hour
        per_hour[hr]["n"] += 1
        per_hour[hr]["pnl"] += trade_pnl
        if won:
            per_hour[hr]["wins"] += 1
        if trade_log is not None:
            confidence = max(entry["p_bull"], 1 - entry["p_bull"])
            trade_log.append({
                "asset": asset,
                "fold": fold_idx if fold_idx is not None else "",
                "window_start": ws_val,
                "entry_time": entry_time.isoformat(),
                "hour_utc": hr,
                "dm": entry["dm"],
                "p_bull": round(entry["p_bull"], 4),
                "confidence": round(confidence, 4),
                "direction": direction,
                "side": side,
                "price_cents": price,
                "max_price_cap": max_price,
                "threshold": threshold,
                "min_dm": min_dm,
                "outcome": outcome,
                "won": int(won),
                "pnl": round(trade_pnl, 4),
            })
    wr = wins / traded * 100 if traded else 0
    return traded, wins, wr, pnl, dict(per_hour)


def write_trade_log(trade_log: list, asset: str):
    if not trade_log:
        return
    out_dir = PROJECT_ROOT / "output" / "walk_forward_trades"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{asset}{TRADE_LOG_SUFFIX}.csv"
    fieldnames = list(trade_log[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(trade_log)
    print(f"  Per-trade log: {out_path} ({len(trade_log)} trades)")


def run_full_sweep(assets, n_folds=4, staging=False, holdout_days=0):
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

        # Holdout partition: reserve the last N days as untouched for honest out-of-sample test
        holdout_cutoff_date = None
        sweep_mask = np.ones(len(ws), dtype=bool)
        if holdout_days > 0:
            all_dates = sorted(set(w[:10] for w in ws if w))
            if len(all_dates) > holdout_days:
                holdout_cutoff_date = all_dates[-holdout_days]  # first date IN holdout
                sweep_mask = np.array([w[:10] < holdout_cutoff_date for w in ws])
                holdout_rows = (~sweep_mask).sum()
                print(f"  Holdout: last {holdout_days} days ({holdout_cutoff_date} onwards) — {holdout_rows} rows withheld from sweep")
            else:
                print(f"  WARNING: only {len(all_dates)} date buckets available, skipping holdout")

        X_sweep, y_sweep = X[sweep_mask], y[sweep_mask]
        ws_sweep, dms_sweep = ws[sweep_mask], dms[sweep_mask]

        folds = get_fold_boundaries(ws_sweep, n_folds)
        print(f"  Folds (on sweep data only): {len(folds)}")
        for i, (te, ts) in enumerate(folds):
            print(f"    Fold {i+1}: train <= {te}, test -> {ts}")

        kalshi_idx = KalshiPollIndex(KALSHI_DIR, asset)
        print(f"  Kalshi events: {len(kalshi_idx.event_tickers)}")
        print()

        best_score = (-999999, -999999, -999999)  # (min_pnl, min_wr, avg_pnl)
        best_config = None
        best_fold_predictions = None  # snapshotted when best_config updates
        n_done = 0
        total = total_xgb * total_dm

        for min_dm in MIN_DM_VALUES:
            for xgb_combo in xgb_combos:
                xgb_params = dict(zip(xgb_keys, xgb_combo))
                xgb_params["objective"] = "binary:logistic"
                xgb_params["eval_metric"] = "logloss"
                xgb_params["seed"] = 42
                xgb_params["verbosity"] = 0
                xgb_params["device"] = "cuda"
                xgb_params["tree_method"] = "hist"

                # Walk-forward: train/predict across folds
                fold_predictions = []  # list of (proba, ws_test, dm_test) per fold

                for train_end, test_end in folds:
                    # Filter by min_dm for training (SWEEP data only, holdout never seen here)
                    train_mask = (np.array([w[:10] <= train_end for w in ws_sweep])) & (dms_sweep >= min_dm)
                    test_mask = (np.array([w[:10] > train_end for w in ws_sweep])) & (np.array([w[:10] <= test_end for w in ws_sweep]))

                    X_train, y_train = X_sweep[train_mask], y_sweep[train_mask]
                    X_test, y_test = X_sweep[test_mask], y_sweep[test_mask]
                    ws_test = ws_sweep[test_mask]
                    dm_test = dms_sweep[test_mask]

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
                        fold_per_hour = []  # list of {hour: {n, wins, pnl}} per fold

                        for proba, ws_test, dm_test in fold_predictions:
                            traded, wins, wr, pnl, ph = simulate_pnl(
                                proba, ws_test, dm_test, asset, kalshi_idx,
                                threshold, max_price, min_dm,
                            )
                            if traded > 0:
                                fold_pnls.append(pnl)
                                fold_wrs.append(wr)
                                fold_trades.append(traded)
                                fold_per_hour.append(ph)

                        if not fold_pnls:
                            continue

                        avg_pnl = sum(fold_pnls) / len(fold_pnls)
                        avg_wr = sum(fold_wrs) / len(fold_wrs)
                        avg_trades = sum(fold_trades) / len(fold_trades)

                        # Require profitable in ALL folds with consistent WR.
                        # Score = min_fold_pnl (best worst-case), with tie-break on avg_pnl.
                        # This picks the most CONSISTENT combo, not the flashiest.
                        all_positive = all(p > 0 for p in fold_pnls)
                        if not all_positive or len(fold_pnls) < len(folds):
                            continue

                        min_pnl = min(fold_pnls)
                        min_wr = min(fold_wrs)
                        # Primary score: minimum fold PnL (best worst-case)
                        # Secondary: minimum fold WR (most consistent WR)
                        score = (min_pnl, min_wr, avg_pnl)

                        if score > best_score:
                            best_score = score
                            best_config = {
                                **{k: xgb_params[k] for k in xgb_keys},
                                "min_dm": min_dm,
                                "threshold": threshold,
                                "max_price": max_price,
                                "avg_pnl": round(avg_pnl, 2),
                                "min_fold_pnl": round(min_pnl, 2),
                                "avg_wr": round(avg_wr, 1),
                                "min_fold_wr": round(min_wr, 1),
                                "avg_trades": round(avg_trades, 1),
                                "n_valid_folds": len(fold_pnls),
                                "fold_pnls": [round(p, 2) for p in fold_pnls],
                                "fold_wrs": [round(w, 1) for w in fold_wrs],
                                "fold_per_hour": fold_per_hour,
                            }
                            best_fold_predictions = fold_predictions

                n_done += 1
                if n_done % 50 == 0:
                    if best_config:
                        print(f"  {n_done}/{total} done... best: min_pnl=${best_config['min_fold_pnl']:+.0f} min_wr={best_config['min_fold_wr']:.0f}% avg=${best_config['avg_pnl']:+.0f} dm>={best_config['min_dm']} thresh={best_config['threshold']} maxP={best_config['max_price']}")
                    else:
                        print(f"  {n_done}/{total} done... no all-folds-positive combo found yet")

        if best_config is None:
            print(f"  ERROR: No valid combo found for {asset}")
            continue

        # Per-trade log for the winning config (replay on saved fold predictions)
        if best_fold_predictions is not None:
            trade_log = []
            for fi, (proba, ws_test, dm_test) in enumerate(best_fold_predictions):
                simulate_pnl(
                    proba, ws_test, dm_test, asset, kalshi_idx,
                    best_config["threshold"], best_config["max_price"], best_config["min_dm"],
                    trade_log=trade_log, fold_idx=fi + 1,
                )
            write_trade_log(trade_log, asset)

        # Report top result
        print(f"\n  {'='*60}")
        print(f"  {asset} BEST CONFIG (avg across {best_config['n_valid_folds']} folds):")
        print(f"  {'='*60}")
        print(f"  Avg OOS PnL:  ${best_config['avg_pnl']:+.2f}")
        print(f"  Avg OOS WR:   {best_config['avg_wr']:.1f}%")
        print(f"  Min fold PnL: ${best_config['min_fold_pnl']:+.2f}  (worst-case fold)")
        print(f"  Min fold WR:  {best_config['min_fold_wr']:.1f}%   (least consistent fold)")
        print(f"  Fold PnLs:    {best_config['fold_pnls']}")
        print(f"  Fold WRs:     {best_config['fold_wrs']}")
        print(f"  Avg trades:   {best_config['avg_trades']:.0f}/fold")

        # ---- Hour-of-day consistency analysis (across folds) ----
        # An hour is "consistent good/bad" if positive/negative in EVERY fold
        # that traded that hour (with at least 3 trades).
        agg_per_hour = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0,
                                             "fold_pnls": [], "fold_n": []})
        for fold_ph in best_config["fold_per_hour"]:
            for hr, s in fold_ph.items():
                if s["n"] < 3:  # require enough trades in fold to count
                    continue
                agg_per_hour[hr]["n"] += s["n"]
                agg_per_hour[hr]["wins"] += s["wins"]
                agg_per_hour[hr]["pnl"] += s["pnl"]
                agg_per_hour[hr]["fold_pnls"].append(round(s["pnl"], 2))
                agg_per_hour[hr]["fold_n"].append(s["n"])

        consistent_good = []  # hours positive in all qualifying folds
        consistent_bad = []   # hours negative in all qualifying folds
        for hr, s in agg_per_hour.items():
            if len(s["fold_pnls"]) < 2:
                continue  # need >=2 folds to call it consistent
            if all(p > 0 for p in s["fold_pnls"]):
                consistent_good.append((hr, s))
            elif all(p < 0 for p in s["fold_pnls"]):
                consistent_bad.append((hr, s))

        consistent_good.sort(key=lambda x: -x[1]["pnl"])
        consistent_bad.sort(key=lambda x: x[1]["pnl"])

        print(f"\n  Consistent GOOD hours (positive PnL in every qualifying fold):")
        if consistent_good:
            for hr, s in consistent_good:
                wr = 100 * s["wins"] / s["n"] if s["n"] else 0
                print(f"    Hour {hr:02d}: ${s['pnl']:+.2f} ({wr:.1f}% WR, {s['n']} trades, fold PnLs {s['fold_pnls']})")
        else:
            print(f"    (none)")

        print(f"\n  Consistent BAD hours (negative PnL in every qualifying fold):")
        if consistent_bad:
            for hr, s in consistent_bad:
                wr = 100 * s["wins"] / s["n"] if s["n"] else 0
                print(f"    Hour {hr:02d}: ${s['pnl']:+.2f} ({wr:.1f}% WR, {s['n']} trades, fold PnLs {s['fold_pnls']})")
        else:
            print(f"    (none)")
        # Stash for metadata write below
        best_config["consistent_good_hours"] = [hr for hr, _ in consistent_good]
        best_config["consistent_bad_hours"] = [hr for hr, _ in consistent_bad]
        best_config["per_hour_summary"] = {
            int(hr): {"n": s["n"], "wins": s["wins"], "pnl": round(s["pnl"], 2),
                      "fold_pnls": s["fold_pnls"]}
            for hr, s in agg_per_hour.items()
        }
        print(f"  XGB params:")
        for k in xgb_keys:
            print(f"    {k}: {best_config[k]}")
        print(f"  Execution:")
        print(f"    min_dm:     {best_config['min_dm']}")
        print(f"    threshold:  {best_config['threshold']}")
        print(f"    max_price:  {best_config['max_price']}c")

        # ---- Holdout evaluation: train winner on sweep data only, test on holdout ----
        holdout_result = None
        if holdout_cutoff_date is not None:
            print(f"\n  ==== HOLDOUT TEST (last {holdout_days} days, never seen by sweep) ====")
            final_min_dm = best_config["min_dm"]
            # Train on sweep portion only, with winner's XGB params
            sweep_dm_mask = dms_sweep >= final_min_dm
            X_tr = X_sweep[sweep_dm_mask]
            y_tr = y_sweep[sweep_dm_mask]
            dtrain_h = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_list)
            # Use holdout rows (post-cutoff) as both eval set for early stop AND for simulation
            holdout_idx = ~sweep_mask
            X_ho = X[holdout_idx]
            y_ho = y[holdout_idx]
            ws_ho = ws[holdout_idx]
            dms_ho = dms[holdout_idx]
            dtest_h = xgb.DMatrix(X_ho, label=y_ho, feature_names=feature_list)

            holdout_params = {
                **{k: best_config[k] for k in xgb_keys},
                "objective": "binary:logistic", "eval_metric": "logloss",
                "seed": 42, "verbosity": 0,
                "device": "cuda", "tree_method": "hist",
            }
            model_h = xgb.train(
                holdout_params, dtrain_h, num_boost_round=400,
                evals=[(dtest_h, "holdout")],
                early_stopping_rounds=20, verbose_eval=False,
            )
            proba_h = model_h.predict(dtest_h)

            # Simulate PnL on holdout with winner's threshold + max_price + min_dm
            h_traded, h_wins, h_wr, h_pnl, h_per_hour = simulate_pnl(
                proba_h, ws_ho, dms_ho, asset, kalshi_idx,
                best_config["threshold"], best_config["max_price"], best_config["min_dm"],
            )
            print(f"  Holdout PnL:     ${h_pnl:+.2f}")
            print(f"  Holdout WR:      {h_wr:.1f}%  ({h_wins}/{h_traded} trades)")
            print(f"  Holdout trades:  {h_traded}")
            sweep_avg = best_config["avg_pnl"]
            delta = h_pnl - sweep_avg
            tag = "[COHERENT]" if h_pnl > 0 and abs(delta) / max(sweep_avg, 1) < 0.5 else "[CHECK]"
            print(f"  vs sweep avg ${sweep_avg:+.2f}/fold: delta=${delta:+.2f} {tag}")
            holdout_result = {
                "days": holdout_days,
                "cutoff_date": holdout_cutoff_date,
                "pnl": round(h_pnl, 2),
                "wr": round(h_wr, 1),
                "trades": h_traded,
                "wins": h_wins,
                "sweep_avg_pnl": round(sweep_avg, 2),
                "delta_vs_sweep_avg": round(delta, 2),
            }

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
            "device": "cuda",
            "tree_method": "hist",
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

        # Save metadata (with per-fold breakdown so we can audit selection rules)
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
            "min_fold_pnl": best_config["min_fold_pnl"],
            "min_fold_wr": best_config["min_fold_wr"],
            "fold_pnls": best_config["fold_pnls"],
            "fold_wrs": best_config["fold_wrs"],
            "avg_trades_per_fold": best_config["avg_trades"],
            "n_folds": best_config["n_valid_folds"],
            "consistent_good_hours": best_config.get("consistent_good_hours", []),
            "consistent_bad_hours": best_config.get("consistent_bad_hours", []),
            "per_hour_summary": best_config.get("per_hour_summary", {}),
            "training_method": "walk_forward_full_combo",
            "holdout": holdout_result,
        }
        meta_path = MODELS_DIR / f"{asset}_xgb_features.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        for variant in ["_weekday", "_weekend"]:
            shutil.copy2(meta_path, MODELS_DIR / f"{asset}{variant}_xgb_features.json")

        # Update config/trading.json with best execution params (skip in staging mode)
        if not staging:
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
            ens["walk_forward_min_fold_pnl"] = best_config["min_fold_pnl"]
            ens["walk_forward_min_fold_wr"] = best_config["min_fold_wr"]
            ens["walk_forward_fold_pnls"] = best_config["fold_pnls"]
            ens["walk_forward_fold_wrs"] = best_config["fold_wrs"]
            ens["walk_forward_consistent_good_hours"] = best_config.get("consistent_good_hours", [])
            ens["walk_forward_consistent_bad_hours"] = best_config.get("consistent_bad_hours", [])
            if holdout_result is not None:
                ens["holdout_pnl"] = holdout_result["pnl"]
                ens["holdout_wr"] = holdout_result["wr"]
                ens["holdout_trades"] = holdout_result["trades"]
                ens["holdout_days"] = holdout_result["days"]
                ens["holdout_cutoff_date"] = holdout_result["cutoff_date"]
            asset_cfg["max_price_cents"] = best_config["max_price"]
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
                f.write("\n")
        else:
            print(f"  [STAGING] Skipped config update for {asset}")

        # Feature importance
        importance = final_model.get_score(importance_type="gain")
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
        print(f"\n  Top 10 features:")
        for name, gain in sorted_imp[:10]:
            print(f"    {name:<30} {gain:>8.1f}")

        print(f"\n  Saved: {model_path}, {meta_path}, config updated")
        print()


def main():
    parser = argparse.ArgumentParser(description="Full combo walk-forward sweep")
    parser.add_argument("--asset", required=True, help="Assets (comma-separated)")
    parser.add_argument("--n-folds", type=int, default=4, help="Walk-forward folds")
    parser.add_argument("--staging", type=str, default="",
                        help="Save models to staging dir (e.g. 'v12') instead of live models/. "
                             "Skips config update. Models go to models/staging_{name}/")
    parser.add_argument("--min-dm", type=str, default="",
                        help="Override MIN_DM_VALUES (comma-separated, e.g. '7' or '5,7'). "
                             "Filters both training rows and entry checkpoints.")
    parser.add_argument("--trade-log-suffix", type=str, default="",
                        help="Suffix for per-trade CSV output, e.g. '_dm7' -> BTC_dm7.csv")
    parser.add_argument("--holdout-days", type=int, default=0,
                        help="Reserve last N days as untouched holdout. Sweep only sees "
                             "earlier data; winner is retested on the holdout as an honest "
                             "out-of-sample check. 0 = disabled (default, old behavior).")
    args = parser.parse_args()

    if args.staging:
        global MODELS_DIR
        MODELS_DIR = PROJECT_ROOT / "models" / f"staging_{args.staging}"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"STAGING MODE: models -> {MODELS_DIR} (config NOT updated)")

    if args.min_dm:
        global MIN_DM_VALUES
        MIN_DM_VALUES = [int(x) for x in args.min_dm.split(",")]
        print(f"DM OVERRIDE: MIN_DM_VALUES -> {MIN_DM_VALUES}")

    global TRADE_LOG_SUFFIX
    TRADE_LOG_SUFFIX = args.trade_log_suffix

    assets = [a.strip().upper() for a in args.asset.split(",")]
    run_full_sweep(assets, args.n_folds, staging=bool(args.staging),
                   holdout_days=args.holdout_days)


if __name__ == "__main__":
    main()
