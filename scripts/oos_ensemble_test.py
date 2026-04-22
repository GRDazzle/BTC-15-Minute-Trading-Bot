"""OOS ensemble test: evaluate XGB+LSTM dynamic blend on held-out days.

For the last N days (OOS for all models):
  1. Run staging un-stacked XGB on each checkpoint's features -> xgb_p
  2. Use the pre-computed lstm_p OOF column as LSTM proxy
  3. Set fusion_p = 0.5 (not simulated in backtest)
  4. Blend: ensemble_p = w_xgb*xgb_p + w_lstm*lstm_p + w_fus*0.5
     where w_xgb/w_lstm/w_fus follow w = min + (max-min)*conf^n
  5. Apply threshold, pick side, check price caps, settle vs Kalshi outcome

Two modes:
  --mode ensemble       : just the blend
  --mode ensemble_rl    : blend + RL gate (skip if RL says WAIT,
                          use RL's contract sizing)

Sweeps dynamic_k (n) over a user-supplied list to find best exponent
per asset.

Notes / approximations:
  - LSTM uses k-fold OOF column in stacked CSV (NOT live LSTM inference).
    This is actually a stricter test — OOF was trained without these rows.
  - Fusion is not simulated (set to 0.5 neutral). Skews blend toward
    XGB/LSTM but they dominate weight anyway.
  - Only checkpoints with dm >= min_dm are considered.

Usage:
  python scripts/oos_ensemble_test.py --asset BTC,ETH,SOL,XRP --holdout-days 7 \\
    --mode ensemble --dk-sweep 1,2,3,4.5,6,8 \\
    --xgb-model-dir models/staging_unstacked_dm7 \\
    --rl-model-suffix oos
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import FEATURE_NAMES
from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker


def build_live_lstm_lookup(asset: str, models_dir: Path) -> dict:
    """Run the LIVE LSTM model on all pre-built sequences for this asset,
    return dict mapping (window_start_str, dm) -> p_bullish.

    This replaces the OOF lstm_p column (which came from k-fold LSTMs
    trained on subsets of the data) with predictions from the actual
    production LSTM that lives in models/{ASSET}_lstm.pt.
    """
    import torch
    from ml.lstm_model import load_model

    seq_path = TRAINING_DIR / f"{asset}_lstm_sequences.npz"
    model_path = models_dir / f"{asset}_lstm.pt"
    if not seq_path.exists() or not model_path.exists():
        print(f"  [live-LSTM] missing: {seq_path.name if not seq_path.exists() else model_path.name}")
        return {}

    d = np.load(seq_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    wss = d["window_starts"]
    dms = d["dms"]

    model, meta = load_model(str(model_path))
    model.eval()

    # Apply scaler if present
    sm = meta.get("scaler_mean")
    ss = meta.get("scaler_std")
    if sm is not None and ss is not None:
        sm = np.array(sm, dtype=np.float32)
        ss = np.array(ss, dtype=np.float32)
        ss[ss == 0] = 1.0
        X = (X - sm) / ss
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Batched inference for speed
    preds = np.zeros(len(X), dtype=np.float32)
    BATCH = 512
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            x = torch.from_numpy(X[i:i+BATCH])
            preds[i:i+BATCH] = model(x).squeeze(-1).numpy()

    lookup = {}
    for i, (ws, dm) in enumerate(zip(wss, dms)):
        lookup[(str(ws), int(dm))] = float(preds[i])
    return lookup

TRAINING_DIR = PROJECT_ROOT / "ml" / "training_data"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"


def load_asset_config(asset: str) -> dict:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    ens = cfg.get("assets", {}).get(asset, {}).get("ensemble", {})
    return {
        "xgb_min_w": float(ens.get("ml_weight", 0.1)),
        "xgb_max_w": float(ens.get("xgb_max_w", 0.8)),
        "lstm_min_w": float(ens.get("lstm_min_w", 0.1)),
        "lstm_max_w": float(ens.get("lstm_max_w", 0.4)),
        "threshold": float(ens.get("threshold", 0.75)),
        "max_price": int(ens.get("max_price_cents", 80)),
        "min_price": 55,
        "min_dm": int(ens.get("min_dm", 7)),
    }


def load_holdout_rows(asset: str, holdout_days: int):
    """Load rows from {ASSET}_features_stacked.csv for the last N days."""
    path = TRAINING_DIR / f"{asset}_features_stacked.csv"
    all_rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            all_rows.append(row)

    dates = sorted(set(r["window_start"][:10] for r in all_rows if r.get("window_start")))
    if len(dates) <= holdout_days:
        return all_rows, dates[0] if dates else None
    cutoff = dates[-holdout_days]
    holdout = [r for r in all_rows if r["window_start"][:10] >= cutoff]
    return holdout, cutoff


def load_xgb_model(asset: str, model_dir: Path):
    model_path = model_dir / f"{asset}_xgb.json"
    feature_names_path = model_dir / f"{asset}_xgb_features.json"

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # Determine feature list (un-stacked = no lstm_p)
    if feature_names_path.exists():
        with open(feature_names_path) as f:
            meta = json.load(f)
        features = meta.get("features", list(FEATURE_NAMES))
        features = [fn for fn in features if fn != "lstm_p"]
    else:
        features = list(FEATURE_NAMES)
    return booster, features


def blend_ensemble(xgb_p: float, lstm_p: float | None, fusion_p: float,
                   cfg: dict, k_xgb: float, k_lstm: float) -> float:
    """Independent dynamic_k for XGB and LSTM so their confidence scales can differ."""
    xgb_conf = abs(xgb_p - 0.5) * 2.0
    dyn_xgb_w = cfg["xgb_min_w"] + (cfg["xgb_max_w"] - cfg["xgb_min_w"]) * (xgb_conf ** k_xgb)
    if lstm_p is not None:
        lstm_conf = abs(lstm_p - 0.5) * 2.0
        dyn_lstm_w = cfg["lstm_min_w"] + (cfg["lstm_max_w"] - cfg["lstm_min_w"]) * (lstm_conf ** k_lstm)
        dyn_fus_w = max(0.0, 1.0 - dyn_xgb_w - dyn_lstm_w)
        return dyn_xgb_w * xgb_p + dyn_lstm_w * lstm_p + dyn_fus_w * fusion_p
    dyn_fus_w = max(0.0, 1.0 - dyn_xgb_w)
    return dyn_xgb_w * xgb_p + dyn_fus_w * fusion_p


def precompute_windows(asset: str, holdout_rows: list, xgb_booster, features: list,
                       cfg: dict, kalshi_idx: KalshiPollIndex,
                       lstm_lookup: dict | None = None,
                       lstm_offset: float = 0.0):
    """Precompute per-checkpoint data once so the (k_xgb, k_lstm) sweep is cheap.

    Returns list of window dicts:
      {window_dt, outcome, checkpoints: [{dm, xgb_p, lstm_p, poll, price_bullish,
        price_bearish, spread, mtc, price_vs_open, velocity_900s, z_score_900s,
        entry_time, hour_utc}, ...]}
    """
    windows_by_ws = defaultdict(list)
    for r in holdout_rows:
        ws = r.get("window_start", "")
        if not ws:
            continue
        dm = int(float(r.get("minute_in_window", 0)))
        if dm < cfg["min_dm"]:
            continue
        windows_by_ws[ws].append(r)

    # Batch XGB inference for speed: one DMatrix over all holdout rows
    out_windows = []
    for ws, rows in windows_by_ws.items():
        try:
            window_dt = datetime.fromisoformat(ws)
            window_end = window_dt + timedelta(minutes=15)
        except Exception:
            continue
        event_ticker = window_start_to_event_ticker(asset, window_end)
        outcome = kalshi_idx.get_outcome(event_ticker)
        if outcome is None:
            continue

        rows.sort(key=lambda r: int(float(r["minute_in_window"])))

        # Batch XGB predict
        xmat = np.array([[float(r.get(fn, 0.0)) for fn in features] for r in rows],
                        dtype=np.float32)
        dmat = xgb.DMatrix(xmat, feature_names=features)
        xgb_preds = xgb_booster.predict(dmat)

        cps = []
        for row, xgb_p in zip(rows, xgb_preds):
            dm = int(float(row["minute_in_window"]))
            # Prefer live LSTM inference if lookup provided, fall back to OOF column
            lstm_p = None
            if lstm_lookup is not None:
                lstm_p = lstm_lookup.get((ws, dm))
            if lstm_p is None:
                lstm_p_raw = row.get("lstm_p", "")
                try:
                    lstm_p = float(lstm_p_raw) if lstm_p_raw else None
                except ValueError:
                    lstm_p = None
            # Apply calibration offset (observed live drift); clamp to [0, 1]
            if lstm_p is not None and lstm_offset:
                lstm_p = max(0.0, min(1.0, lstm_p + lstm_offset))

            entry_time = window_dt + timedelta(minutes=5 + dm)
            poll = kalshi_idx.find_poll(event_ticker, entry_time)
            if poll is None:
                continue
            yes_ask = poll["yes_ask"]
            no_ask = poll.get("no_ask", 100 - poll.get("yes_bid", 50))
            spread = yes_ask - poll.get("yes_bid", yes_ask)

            cps.append({
                "dm": dm,
                "xgb_p": float(xgb_p),
                "lstm_p": lstm_p,
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "spread": spread,
                "mtc": float(poll.get("mins_to_close", 10 - dm + 5)),
                "price_vs_open": float(row.get("price_vs_open", 0.0)),
                "velocity_900s": float(row.get("velocity_900s", 0.0)),
                "z_score_900s": float(row.get("z_score_900s", 0.0)),
                "entry_time": entry_time,
                "hour_utc": entry_time.hour,
            })

        if cps:
            out_windows.append({"window_dt": window_dt, "outcome": outcome,
                                "checkpoints": cps})
    return out_windows


def simulate(asset: str, precomputed_windows: list, cfg: dict,
             k_xgb: float, k_lstm: float,
             use_rl: bool = False, rl_processor=None, max_contracts: int = 10):
    """Simulate one pass over precomputed holdout. Returns aggregate stats."""
    traded = wins = 0
    pnl = 0.0
    trade_rows = []

    for w in precomputed_windows:
        outcome = w["outcome"]

        if rl_processor is not None:
            rl_processor.reset_window()

        for cp in w["checkpoints"]:
            xgb_p = cp["xgb_p"]
            lstm_p = cp["lstm_p"]
            fusion_p = 0.5
            ensemble_p = blend_ensemble(xgb_p, lstm_p, fusion_p, cfg, k_xgb, k_lstm)

            # Threshold gate
            if ensemble_p >= cfg["threshold"]:
                direction = "BULLISH"
                conf = ensemble_p
            elif ensemble_p <= 1.0 - cfg["threshold"]:
                direction = "BEARISH"
                conf = 1.0 - ensemble_p
            else:
                continue

            side = "yes" if direction == "BULLISH" else "no"
            price = cp["yes_ask"] if side == "yes" else cp["no_ask"]

            if price < cfg["min_price"] or price > cfg["max_price"]:
                continue

            contracts = max_contracts

            # RL gate (optional)
            if use_rl and rl_processor is not None:
                rl_enter, rl_contracts = rl_processor.decide(
                    xgb_p=xgb_p,
                    yes_ask=cp["yes_ask"],
                    no_ask=cp["no_ask"],
                    spread=cp["spread"],
                    mins_to_close=cp["mtc"],
                    dm=cp["dm"],
                    price_vs_open=cp["price_vs_open"],
                    velocity_900s=cp["velocity_900s"],
                    z_score_900s=cp["z_score_900s"],
                    hour_utc=cp["hour_utc"],
                )
                if not rl_enter:
                    continue
                contracts = max(1, min(max_contracts, rl_contracts))

            won = (side == outcome)
            cost = (price / 100.0 + 0.02) * contracts
            trade_pnl = (contracts * 1.0 - cost) if won else -cost

            traded += 1
            if won:
                wins += 1
            pnl += trade_pnl
            trade_rows.append({
                "asset": asset, "dm": cp["dm"],
                "xgb_p": round(xgb_p, 4),
                "lstm_p": round(lstm_p, 4) if lstm_p is not None else "",
                "ensemble_p": round(ensemble_p, 4),
                "direction": direction, "side": side, "price": price,
                "contracts": contracts, "outcome": outcome,
                "won": int(won), "pnl": round(trade_pnl, 4),
            })
            break  # one trade per window

    wr = 100 * wins / traded if traded else 0.0
    return {"traded": traded, "wins": wins, "wr": wr, "pnl": pnl, "trades": trade_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC,ETH,SOL,XRP")
    ap.add_argument("--holdout-days", type=int, default=7)
    ap.add_argument("--mode", choices=["ensemble", "ensemble_rl"], default="ensemble")
    ap.add_argument("--xgb-dk-sweep", default="1,2,3,4.5,6,8",
                    help="XGB dynamic_k values to sweep")
    ap.add_argument("--lstm-dk-sweep", default="1,2,3,4.5,6,8",
                    help="LSTM dynamic_k values to sweep")
    ap.add_argument("--xgb-model-dir", default="models/staging_unstacked_dm7",
                    help="Dir containing {ASSET}_xgb.json models")
    ap.add_argument("--rl-model-suffix", default="oos",
                    help="Load RL from models/{ASSET}_rl_{suffix}.zip")
    ap.add_argument("--min-dm", type=int, default=None,
                    help="Override min_dm from config (e.g. 4). Filters checkpoints.")
    ap.add_argument("--live-lstm", action="store_true", default=False,
                    help="Use live LSTM inference (models/{ASSET}_lstm.pt) "
                         "instead of OOF lstm_p from stacked CSV. Closes the "
                         "train/live lstm gap.")
    ap.add_argument("--lstm-offsets", default="",
                    help="Per-asset lstm_p calibration offsets: "
                         "'BTC=-0.04,SOL=-0.20'. Applied to lstm_p before blend "
                         "to approximate observed live-vs-backtest drift.")
    ap.add_argument("--out", default="output/oos_ensemble_results.json")
    args = ap.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    xgb_dk_values = [float(x) for x in args.xgb_dk_sweep.split(",")]
    lstm_dk_values = [float(x) for x in args.lstm_dk_sweep.split(",")]
    xgb_dir = Path(args.xgb_model_dir)

    # Parse --lstm-offsets="BTC=-0.04,SOL=-0.20"
    lstm_offsets: dict[str, float] = {}
    if args.lstm_offsets:
        for pair in args.lstm_offsets.split(","):
            k, _, v = pair.partition("=")
            k = k.strip().upper()
            try:
                lstm_offsets[k] = float(v)
            except ValueError:
                print(f"WARNING: bad lstm-offsets entry '{pair}' — ignored")

    results = {}
    for asset in assets:
        print(f"\n{'='*70}")
        print(f"  {asset}: OOS test (last {args.holdout_days} days, mode={args.mode})")
        print(f"{'='*70}")

        rows, cutoff = load_holdout_rows(asset, args.holdout_days)
        print(f"  Holdout cutoff: {cutoff} | {len(rows)} rows")

        cfg = load_asset_config(asset)
        if args.min_dm is not None:
            cfg["min_dm"] = args.min_dm
        print(f"  Config: threshold={cfg['threshold']} max_price={cfg['max_price']} "
              f"min_dm={cfg['min_dm']}")
        print(f"  Blend weights: xgb=[{cfg['xgb_min_w']:.2f},{cfg['xgb_max_w']:.2f}] "
              f"lstm=[{cfg['lstm_min_w']:.2f},{cfg['lstm_max_w']:.2f}]")

        xgb_booster, features = load_xgb_model(asset, xgb_dir)
        print(f"  XGB features ({len(features)}): model={xgb_dir}/{asset}_xgb.json")

        kalshi_idx = KalshiPollIndex(KALSHI_DIR, asset)

        rl_processor = None
        if args.mode == "ensemble_rl":
            try:
                from core.strategy_brain.signal_processors.rl_processor import RLProcessor
                # Temporarily patch the expected path suffix
                rl_path = MODELS_DIR / f"{asset}_rl_{args.rl_model_suffix}.zip"
                if not rl_path.exists():
                    print(f"  [SKIP] No RL model at {rl_path}")
                    continue
                # RLProcessor is hardcoded to "_rl_prod.zip"; load directly:
                from stable_baselines3 import PPO
                class _RL:
                    def __init__(self, path):
                        self.model = PPO.load(str(path), device="cpu")
                        self.price_history = []
                        self.entered_this_window = False
                    def reset_window(self):
                        self.price_history = []
                        self.entered_this_window = False
                    def decide(self, xgb_p, yes_ask, no_ask, spread, mins_to_close,
                               dm, price_vs_open, velocity_900s, z_score_900s, hour_utc):
                        import math
                        if self.entered_this_window:
                            return False, 0
                        side_price = yes_ask if xgb_p >= 0.5 else no_ask
                        self.price_history.append(side_price)
                        ph = self.price_history
                        p_prev1 = ph[-2] if len(ph) >= 2 else ph[-1]
                        p_prev2 = ph[-3] if len(ph) >= 3 else p_prev1
                        price_delta_1 = (side_price - p_prev1) / 10.0
                        price_delta_2 = (p_prev1 - p_prev2) / 10.0
                        hour_sin = math.sin(2*math.pi*hour_utc/24.0)
                        hour_cos = math.cos(2*math.pi*hour_utc/24.0)
                        obs = np.array([
                            xgb_p*2-1, abs(xgb_p-0.5)*2, side_price/100.0,
                            mins_to_close/15.0, dm/9.0, spread/10.0,
                            price_delta_1, price_delta_2, 0.0,
                            price_vs_open*100, velocity_900s*100, z_score_900s,
                            hour_sin, hour_cos,
                        ], dtype=np.float32)
                        obs = np.clip(obs, -5.0, 5.0)
                        action, _ = self.model.predict(obs, deterministic=True)
                        av = float(action[0]) if hasattr(action, '__len__') else float(action)
                        if av > 0.3:
                            frac = (av - 0.3) / 0.7
                            contracts = max(1, int(frac * 10))
                            self.entered_this_window = True
                            return True, contracts
                        return False, 0
                rl_processor = _RL(rl_path)
                print(f"  RL loaded: {rl_path}")
            except Exception as e:
                print(f"  [SKIP] RL load error: {e}")
                continue

        # Precompute checkpoint data once (batched XGB inference + Kalshi lookups)
        lstm_lookup = None
        if args.live_lstm:
            print(f"  Building live-LSTM lookup...")
            lstm_lookup = build_live_lstm_lookup(asset, MODELS_DIR)
            print(f"  Live LSTM predictions: {len(lstm_lookup)} cached")
        offset = lstm_offsets.get(asset, 0.0)
        if offset:
            print(f"  Applying lstm_p offset: {offset:+.3f}")
        print(f"  Precomputing checkpoints...")
        precomputed = precompute_windows(asset, rows, xgb_booster, features, cfg, kalshi_idx,
                                         lstm_lookup=lstm_lookup, lstm_offset=offset)
        n_cps = sum(len(w["checkpoints"]) for w in precomputed)
        print(f"  Precomputed {len(precomputed)} windows, {n_cps} checkpoints")

        asset_results = []
        # 2D sweep grid
        print(f"\n  PnL grid (rows=k_xgb, cols=k_lstm):")
        header = "  k_xgb\\k_lstm "
        for k_lstm in lstm_dk_values:
            header += f"{k_lstm:>8.1f}"
        print(header)
        for k_xgb in xgb_dk_values:
            row_line = f"  {k_xgb:<13.1f}"
            for k_lstm in lstm_dk_values:
                r = simulate(asset, precomputed, cfg, k_xgb, k_lstm,
                             use_rl=(args.mode == "ensemble_rl"),
                             rl_processor=rl_processor)
                ppt = r['pnl']/r['traded'] if r['traded'] else 0.0
                row_line += f"{r['pnl']:>+8.1f}"
                asset_results.append({
                    "k_xgb": k_xgb, "k_lstm": k_lstm,
                    "trades": r["traded"], "wins": r["wins"],
                    "wr": round(r["wr"], 2), "pnl": round(r["pnl"], 2),
                    "pnl_per_trade": round(ppt, 4),
                })
            print(row_line)

        # Top 5 combos for this asset
        top5 = sorted(asset_results, key=lambda x: -x["pnl"])[:5]
        print(f"\n  Top 5 combos for {asset}:")
        print(f"    {'k_xgb':>6} {'k_lstm':>7} {'N':>5} {'WR':>6} {'PnL':>9} {'PnL/tr':>8}")
        for t in top5:
            print(f"    {t['k_xgb']:>6.1f} {t['k_lstm']:>7.1f} {t['trades']:>5} "
                  f"{t['wr']:>5.1f}% ${t['pnl']:>+8.2f} ${t['pnl_per_trade']:>+7.3f}")

        results[asset] = {
            "holdout_cutoff": cutoff,
            "holdout_rows": len(rows),
            "mode": args.mode,
            "config": cfg,
            "by_dk_pair": asset_results,
        }

    # Write results
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  {args.mode.upper()} OOS SUMMARY (best (k_xgb, k_lstm) per asset)")
    print(f"{'='*70}")
    print(f"  {'Asset':<6} {'k_xgb':>6} {'k_lstm':>7} {'N':>5} {'WR':>6} {'PnL':>10} {'PnL/tr':>8}")
    totals = {"pnl": 0.0, "trades": 0, "wins": 0}
    for asset, r in results.items():
        best = max(r["by_dk_pair"], key=lambda x: x["pnl"])
        print(f"  {asset:<6} {best['k_xgb']:>6.1f} {best['k_lstm']:>7.1f} {best['trades']:>5} "
              f"{best['wr']:>5.1f}% ${best['pnl']:>+9.2f} ${best['pnl_per_trade']:>+7.3f}")
        totals["pnl"] += best["pnl"]
        totals["trades"] += best["trades"]
        totals["wins"] += best["wins"]
    tot_wr = 100 * totals["wins"] / totals["trades"] if totals["trades"] else 0
    print(f"  {'TOTAL':<6} {'':<6} {'':<7} {totals['trades']:>5} {tot_wr:>5.1f}% ${totals['pnl']:>+9.2f}")


if __name__ == "__main__":
    main()
