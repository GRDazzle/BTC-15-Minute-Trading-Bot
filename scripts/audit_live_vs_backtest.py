"""Diff live signal_log against backtest-predicted values on identical windows.

For each row in output/signal_log.csv since a cutoff timestamp:
  1. Find the matching training-CSV row by (asset, window_start, dm)
  2. Run the LIVE XGB booster on the training-CSV features -> xgb_p_bt
  3. Run the LIVE LSTM on the corresponding npz sequence -> lstm_p_bt
  4. Compute blend with the live config's k_xgb / k_lstm -> ensemble_p_bt
  5. Diff against the values live logged

This isolates WHERE the gap lives:
  - ml_p differs: features are being computed differently live vs training
  - lstm_p differs: LSTM sequence construction differs
  - ensemble_p differs despite components matching: blend formula / config bug

Usage:
  python scripts/audit_live_vs_backtest.py [--since 2026-04-22T02:26:00]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import FEATURE_NAMES

MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "ml" / "training_data"
SIGNAL_LOG = PROJECT_ROOT / "output" / "signal_log.csv"
CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"


def load_asset_config(asset):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    ens = cfg.get("assets", {}).get(asset, {}).get("ensemble", {})
    shared_k = float(ens.get("dynamic_k", 4.5))
    return {
        "xgb_min_w": float(ens.get("ml_weight", 0.1)),
        "xgb_max_w": float(ens.get("xgb_max_w", 0.60)),
        "lstm_min_w": float(ens.get("lstm_min_w", 0.10)),
        "lstm_max_w": float(ens.get("lstm_max_w", 0.40)),
        "k_xgb": float(ens.get("dynamic_k_xgb", shared_k)),
        "k_lstm": float(ens.get("dynamic_k_lstm", shared_k)),
    }


def load_xgb(asset):
    model_path = MODELS_DIR / f"{asset}_xgb.json"
    fn_path = MODELS_DIR / f"{asset}_xgb_features.json"
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    if fn_path.exists():
        with open(fn_path) as f:
            features = json.load(f).get("features", list(FEATURE_NAMES))
        features = [x for x in features if x != "lstm_p"]
    else:
        features = list(FEATURE_NAMES)
    return booster, features


def build_lstm_lookup(asset):
    """Pre-run live LSTM on the sequences npz, index by (ws, dm)."""
    import torch
    from ml.lstm_model import load_model

    seq_path = TRAINING_DIR / f"{asset}_lstm_sequences.npz"
    model_path = MODELS_DIR / f"{asset}_lstm.pt"
    if not seq_path.exists() or not model_path.exists():
        return {}
    d = np.load(seq_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    wss = d["window_starts"]
    dms = d["dms"]
    m, meta = load_model(str(model_path))
    m.eval()
    sm = meta.get("scaler_mean")
    ss = meta.get("scaler_std")
    if sm is not None and ss is not None:
        sm = np.array(sm, dtype=np.float32)
        ss = np.array(ss, dtype=np.float32)
        ss[ss == 0] = 1.0
        X = (X - sm) / ss
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    preds = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 512):
            x = torch.from_numpy(X[i:i+512])
            preds[i:i+512] = m(x).squeeze(-1).numpy()
    return {(str(w), int(dm)): float(p) for w, dm, p in zip(wss, dms, preds)}


def load_training_features(asset):
    """Returns dict[(window_start, dm)] -> feature row."""
    path = TRAINING_DIR / f"{asset}_features.csv"
    lookup = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ws = row.get("window_start", "")
            if not ws:
                continue
            try:
                dm = int(float(row.get("minute_in_window", 0)))
            except ValueError:
                continue
            lookup[(ws, dm)] = row
    return lookup


def blend(xgb_p, lstm_p, fusion_p, cfg):
    xgb_conf = abs(xgb_p - 0.5) * 2.0
    dyn_xgb_w = cfg["xgb_min_w"] + (cfg["xgb_max_w"] - cfg["xgb_min_w"]) * (xgb_conf ** cfg["k_xgb"])
    if lstm_p is not None:
        lstm_conf = abs(lstm_p - 0.5) * 2.0
        dyn_lstm_w = cfg["lstm_min_w"] + (cfg["lstm_max_w"] - cfg["lstm_min_w"]) * (lstm_conf ** cfg["k_lstm"])
        dyn_fus_w = max(0.0, 1.0 - dyn_xgb_w - dyn_lstm_w)
        ens = dyn_xgb_w*xgb_p + dyn_lstm_w*lstm_p + dyn_fus_w*fusion_p
    else:
        dyn_lstm_w = 0.0
        dyn_fus_w = max(0.0, 1.0 - dyn_xgb_w)
        ens = dyn_xgb_w*xgb_p + dyn_fus_w*fusion_p
    return ens, dyn_xgb_w, dyn_lstm_w, dyn_fus_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-04-22T00:00:00",
                    help="Only audit signal rows with timestamp >= this")
    ap.add_argument("--asset", default="BTC,ETH,SOL,XRP",
                    help="Assets to audit")
    ap.add_argument("--out", default="output/live_vs_backtest_audit.csv")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N rows (0 = all)")
    args = ap.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    if not SIGNAL_LOG.exists():
        print(f"No signal log at {SIGNAL_LOG}")
        return

    # Load signal log rows matching filter
    signal_rows = []
    with open(SIGNAL_LOG) as f:
        for r in csv.DictReader(f):
            if r["timestamp"] < args.since:
                continue
            if r["asset"] not in assets:
                continue
            signal_rows.append(r)
    print(f"Loaded {len(signal_rows)} signal log rows since {args.since}")
    if not signal_rows:
        return
    if args.limit:
        signal_rows = signal_rows[:args.limit]

    # Cache per-asset resources
    xgb_boosters = {}
    xgb_features = {}
    lstm_lookups = {}
    feat_lookups = {}
    cfgs = {}
    for asset in assets:
        print(f"Loading {asset}...")
        try:
            b, fs = load_xgb(asset)
            xgb_boosters[asset] = b
            xgb_features[asset] = fs
        except Exception as e:
            print(f"  XGB missing for {asset}: {e}")
            continue
        try:
            lstm_lookups[asset] = build_lstm_lookup(asset)
            print(f"  LSTM lookup: {len(lstm_lookups[asset])} entries")
        except Exception as e:
            print(f"  LSTM load failed: {e}")
            lstm_lookups[asset] = {}
        feat_lookups[asset] = load_training_features(asset)
        print(f"  Training features: {len(feat_lookups[asset])} entries")
        cfgs[asset] = load_asset_config(asset)

    # Audit each row
    out_rows = []
    mismatches_ml = []
    mismatches_lstm = []
    mismatches_ens = []
    missing_feat = 0
    for r in signal_rows:
        asset = r["asset"]
        if asset not in xgb_boosters:
            continue
        # Parse window_start from window_id (format like "20260422_0230")
        # We need actual ISO format. Fall back to reading training feat by dm-aware search.
        wid = r["window_id"]  # "YYYYMMDD_HHMM"
        if len(wid) != 13 or wid[8] != "_":
            continue
        ws_iso = f"{wid[:4]}-{wid[4:6]}-{wid[6:8]}T{wid[9:11]}:{wid[11:13]}:00+00:00"
        dm = int(r["dm"])
        key = (ws_iso, dm)

        feat_row = feat_lookups[asset].get(key)
        if feat_row is None:
            # Try variants of ws_iso format
            alt_keys = [k for k in feat_lookups[asset] if k[0].startswith(ws_iso[:16]) and k[1] == dm]
            if alt_keys:
                feat_row = feat_lookups[asset][alt_keys[0]]
                key = alt_keys[0]
            else:
                missing_feat += 1
                continue

        # XGB inference on training features
        features = xgb_features[asset]
        xvec = np.array([[float(feat_row.get(fn, 0.0)) for fn in features]], dtype=np.float32)
        dmat = xgb.DMatrix(xvec, feature_names=features)
        ml_p_bt = float(xgb_boosters[asset].predict(dmat)[0])

        # LSTM (live inference on npz sequence at this timestamp+dm)
        lstm_p_bt = lstm_lookups[asset].get(key)

        # Blend
        fusion_p = 0.5  # live will differ if fusion fires
        cfg = cfgs[asset]
        ens_p_bt, dxw, dlw, dfw = blend(ml_p_bt, lstm_p_bt, fusion_p, cfg)

        # Live values
        try:
            ml_p_live = float(r["ml_p"])
        except (ValueError, KeyError):
            continue
        lstm_p_live_raw = r.get("lstm_p", "")
        lstm_p_live = float(lstm_p_live_raw) if lstm_p_live_raw else None
        fusion_p_live = float(r.get("fusion_p", 0.5) or 0.5)
        ens_p_live = float(r.get("ensemble_p", ml_p_live))

        d_ml = ml_p_live - ml_p_bt
        d_lstm = (lstm_p_live - lstm_p_bt) if (lstm_p_live is not None and lstm_p_bt is not None) else None
        d_ens = ens_p_live - ens_p_bt

        if abs(d_ml) > 0.01:
            mismatches_ml.append((asset, wid, dm, ml_p_live, ml_p_bt, d_ml))
        if d_lstm is not None and abs(d_lstm) > 0.01:
            mismatches_lstm.append((asset, wid, dm, lstm_p_live, lstm_p_bt, d_lstm))
        if abs(d_ens) > 0.01:
            mismatches_ens.append((asset, wid, dm, ens_p_live, ens_p_bt, d_ens))

        out_rows.append({
            "timestamp": r["timestamp"],
            "asset": asset, "window_id": wid, "dm": dm,
            "ml_p_live": round(ml_p_live, 4), "ml_p_bt": round(ml_p_bt, 4),
            "d_ml_p": round(d_ml, 4),
            "lstm_p_live": "" if lstm_p_live is None else round(lstm_p_live, 4),
            "lstm_p_bt": "" if lstm_p_bt is None else round(lstm_p_bt, 4),
            "d_lstm_p": "" if d_lstm is None else round(d_lstm, 4),
            "fusion_p_live": round(fusion_p_live, 4),
            "ens_p_live": round(ens_p_live, 4), "ens_p_bt": round(ens_p_bt, 4),
            "d_ens_p": round(d_ens, 4),
        })

    # Write report
    out_path = PROJECT_ROOT / args.out
    if out_rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nSaved: {out_path} ({len(out_rows)} rows)")

    # Summary
    def stat(arr, name):
        if not arr:
            print(f"  {name}: 0 mismatches")
            return
        deltas = [x[-1] for x in arr]
        print(f"  {name}: {len(arr)} / {len(out_rows)} mismatched (>1%)")
        print(f"    mean delta: {np.mean(deltas):+.4f}")
        print(f"    max abs: {max(abs(d) for d in deltas):.4f}")
        # Show worst 3
        worst = sorted(arr, key=lambda x: -abs(x[-1]))[:3]
        for asset, wid, dm, live, bt, d in worst:
            print(f"    worst: {asset} {wid} dm={dm}  live={live:.3f} bt={bt:.3f} d={d:+.3f}")

    print(f"\n=== Audit summary ({len(out_rows)} rows audited, {missing_feat} missing features) ===")
    stat(mismatches_ml, "XGB (ml_p) mismatches")
    stat(mismatches_lstm, "LSTM (lstm_p) mismatches")
    stat(mismatches_ens, "Ensemble_p mismatches")

    # Per-asset summary of mean ml_p + lstm_p across live & backtest
    print(f"\n=== Per-asset averages ===")
    per_a = defaultdict(lambda: {"ml_live": [], "ml_bt": [], "lstm_live": [], "lstm_bt": []})
    for r in out_rows:
        per_a[r["asset"]]["ml_live"].append(r["ml_p_live"])
        per_a[r["asset"]]["ml_bt"].append(r["ml_p_bt"])
        if r["lstm_p_live"] != "":
            per_a[r["asset"]]["lstm_live"].append(r["lstm_p_live"])
        if r["lstm_p_bt"] != "":
            per_a[r["asset"]]["lstm_bt"].append(r["lstm_p_bt"])
    print(f"{'Asset':<6} {'N':>5} {'ml_live':>8} {'ml_bt':>8} {'d_ml':>7} {'lstm_live':>10} {'lstm_bt':>9} {'d_lstm':>8}")
    for asset, s in sorted(per_a.items()):
        if not s["ml_live"]: continue
        ml_l = np.mean(s["ml_live"]); ml_b = np.mean(s["ml_bt"])
        line = f"{asset:<6} {len(s['ml_live']):>5} {ml_l:>8.4f} {ml_b:>8.4f} {ml_l-ml_b:>+7.4f} "
        if s["lstm_live"]:
            ll = np.mean(s["lstm_live"]); lb = np.mean(s["lstm_bt"])
            line += f"{ll:>10.4f} {lb:>9.4f} {ll-lb:>+8.4f}"
        else:
            line += f"{'n/a':>10}"
        print(line)


if __name__ == "__main__":
    main()
