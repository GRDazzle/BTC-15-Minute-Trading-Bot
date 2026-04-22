"""Per-asset RL hyperparameter sweep with all-folds-positive gate.

For each asset:
  - Sweep (ent_coef, skip_penalty) combinations
  - Train and evaluate each combo across 4 walk-forward folds
  - Stop when a combo has positive RL delta in EVERY fold
  - Report the passing combo (or best if none pass)

Usage:
    python scripts/sweep_rl_per_asset.py --asset BTC,ETH,SOL,XRP --timesteps 500000
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


ENT_COEF_VALUES = [0.01, 0.02, 0.05]
SKIP_PENALTY_VALUES = [-0.2, -0.5]


def get_fold_boundaries(windows, n_folds=4, min_train_days=14):
    dates = sorted(set(w["window_start"][:10] for w in windows))
    n_dates = len(dates)
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
        folds.append((dates[train_end_idx], dates[test_end_idx]))
    return folds


def _hour_of(window_start: str) -> int:
    try:
        return datetime.fromisoformat(window_start).hour
    except Exception:
        return -1


def _log_trade(rows, asset, fold_idx, strategy, ent_coef, skip_penalty, w, cp,
               side, price, contracts, pnl):
    xgb_p = cp.get("xgb_p", 0.5)
    rows.append({
        "asset": asset,
        "fold": fold_idx,
        "strategy": strategy,
        "ent_coef": ent_coef,
        "skip_penalty": skip_penalty,
        "window_start": w.get("window_start", ""),
        "hour_utc": _hour_of(w.get("window_start", "")),
        "dm": cp.get("dm", -1),
        "mins_to_close": cp.get("mins_to_close", -1),
        "p_bull": round(float(xgb_p), 4),
        "confidence": round(abs(xgb_p - 0.5) * 2, 4),
        "direction": "BULLISH" if xgb_p >= 0.5 else "BEARISH",
        "side": side,
        "price_cents": price,
        "contracts": contracts,
        "outcome": w.get("outcome", ""),
        "won": int(side == w.get("outcome")),
        "pnl": round(float(pnl), 4),
    })


def eval_fold(asset, train_w, test_w, ent_coef, skip_penalty, timesteps,
              fold_idx: int | None = None, trade_rows: list | None = None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.meta_controller_v3 import MetaControllerV3

    train_env = DummyVecEnv([lambda: MetaControllerV3(
        windows=train_w, augment=True, skip_penalty=skip_penalty,
    )])

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=1024, batch_size=256,
        n_epochs=10, gamma=0.99, ent_coef=ent_coef, clip_range=0.2,
        device="cpu", verbose=0,
    )

    model.learn(total_timesteps=timesteps)

    # Evaluate — replay env step-by-step and track which checkpoint RL entered on
    eval_env = MetaControllerV3(windows=test_w, augment=False, skip_penalty=skip_penalty)

    rl_pnl = 0.0
    rl_trades = 0
    rl_wins = 0
    bl_pnl = 0.0
    bl_trades = 0
    bl_wins = 0

    for w in test_w:
        eval_env._current_window = w
        eval_env._step_idx = 0
        eval_env._entered = False
        eval_env._entry_price = 0
        eval_env._entry_side = None
        eval_env._entry_contracts = 0
        eval_env._price_history = []
        obs = eval_env._get_obs()
        done = False
        rl_entry_step = None
        while not done:
            # Snapshot pre-step state so we can capture the checkpoint where RL entered
            pre_entered = eval_env._entered
            pre_step = eval_env._step_idx
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = eval_env.step(action)
            if not pre_entered and eval_env._entered:
                rl_entry_step = pre_step

        if eval_env._entered:
            rl_trades += 1
            settle = eval_env._settle()
            rl_pnl += settle
            if settle > 0:
                rl_wins += 1
            if trade_rows is not None and rl_entry_step is not None:
                cp = w["checkpoints"][rl_entry_step]
                _log_trade(
                    trade_rows, asset, fold_idx, "rl", ent_coef, skip_penalty,
                    w, cp, eval_env._entry_side, eval_env._entry_price,
                    eval_env._entry_contracts, settle,
                )

        for cp in w["checkpoints"]:
            p = cp.get("xgb_p", 0.5)
            if p >= 0.75 or p <= 0.25:
                side = "yes" if p >= 0.75 else "no"
                price = cp["yes_ask"] if side == "yes" else cp["no_ask"]
                if 55 <= price <= 80:
                    won = side == w["outcome"]
                    cost = (price / 100.0 + 0.02) * 10
                    bpnl = (10.0 - cost) if won else -cost
                    bl_pnl += bpnl
                    bl_trades += 1
                    if bpnl > 0:
                        bl_wins += 1
                    if trade_rows is not None:
                        _log_trade(
                            trade_rows, asset, fold_idx, "xgb_baseline",
                            ent_coef, skip_penalty, w, cp, side, price, 10, bpnl,
                        )
                break

    rl_wr = 100 * rl_wins / rl_trades if rl_trades else 0
    bl_wr = 100 * bl_wins / bl_trades if bl_trades else 0

    return {
        "rl_pnl": rl_pnl, "bl_pnl": bl_pnl, "delta": rl_pnl - bl_pnl,
        "rl_trades": rl_trades, "bl_trades": bl_trades,
        "rl_wr": rl_wr, "bl_wr": bl_wr,
    }


def sweep_asset(asset, timesteps=500000):
    from rl.trading_env import load_training_windows

    print(f"\n{'='*70}")
    print(f"  Per-asset RL sweep: {asset}")
    print(f"{'='*70}")

    all_windows = load_training_windows(asset)
    folds = get_fold_boundaries(all_windows, n_folds=4)
    print(f"  Total windows: {len(all_windows)}, folds: {len(folds)}")

    combos = list(product(ENT_COEF_VALUES, SKIP_PENALTY_VALUES))
    print(f"  Sweeping {len(combos)} combos × {len(folds)} folds = {len(combos) * len(folds)} trains")
    print()

    all_results = []
    passing_combos = []
    trade_rows = []  # per-trade log across all combos/folds for this asset

    for combo_idx, (ent_coef, skip_penalty) in enumerate(combos):
        print(f"  [{combo_idx+1}/{len(combos)}] ent_coef={ent_coef}, skip_penalty={skip_penalty}")

        fold_deltas = []
        fold_rl_pnls = []
        fold_bl_pnls = []

        for fold_idx, (train_end, test_end) in enumerate(folds):
            train_w = [w for w in all_windows if w["window_start"][:10] <= train_end]
            test_w = [w for w in all_windows
                      if w["window_start"][:10] > train_end and w["window_start"][:10] <= test_end]

            if len(train_w) < 100 or len(test_w) < 20:
                continue

            t0 = time.time()
            r = eval_fold(
                asset, train_w, test_w, ent_coef, skip_penalty, timesteps,
                fold_idx=fold_idx + 1, trade_rows=trade_rows,
            )
            elapsed = time.time() - t0
            fold_deltas.append(r["delta"])
            fold_rl_pnls.append(r["rl_pnl"])
            fold_bl_pnls.append(r["bl_pnl"])
            sign = "+" if r["delta"] > 0 else ""
            print(f"    Fold {fold_idx+1}: RL ${r['rl_pnl']:>+7.1f} vs XGB ${r['bl_pnl']:>+7.1f}  delta {sign}${r['delta']:>+7.1f} ({elapsed:.0f}s)")

        if len(fold_deltas) < len(folds):
            continue

        all_positive = all(d > 0 for d in fold_deltas)
        min_delta = min(fold_deltas)
        avg_delta = sum(fold_deltas) / len(fold_deltas)
        total_delta = sum(fold_deltas)

        result = {
            "ent_coef": ent_coef,
            "skip_penalty": skip_penalty,
            "fold_deltas": [round(d, 2) for d in fold_deltas],
            "fold_rl_pnls": [round(p, 2) for p in fold_rl_pnls],
            "fold_bl_pnls": [round(p, 2) for p in fold_bl_pnls],
            "all_positive": all_positive,
            "min_delta": round(min_delta, 2),
            "avg_delta": round(avg_delta, 2),
            "total_delta": round(total_delta, 2),
        }
        all_results.append(result)
        if all_positive:
            passing_combos.append(result)

        tag = "[OK] ALL POS" if all_positive else "  "
        print(f"    -> {tag} min_delta=${min_delta:>+6.1f} total=${total_delta:>+7.1f}")

        # Early stop if we have a passing combo (saves time)
        if all_positive:
            print(f"  [STOP] Found all-folds-positive combo for {asset}")
            break

    # Write per-trade log CSV for offline analysis (RL vs baseline entries)
    if trade_rows:
        out_dir = PROJECT_ROOT / "output" / "walk_forward_trades"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{asset}_rl.csv"
        fieldnames = list(trade_rows[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(trade_rows)
        print(f"  Per-trade log: {out_path} ({len(trade_rows)} rows)")

    print(f"\n  {asset} SWEEP RESULTS:")
    if passing_combos:
        # Pick the one with highest min_delta among passing combos
        winner = max(passing_combos, key=lambda r: r["min_delta"])
        print(f"  [OK] PASSING config: ent_coef={winner['ent_coef']}, skip_penalty={winner['skip_penalty']}")
        print(f"    fold_deltas: {winner['fold_deltas']}")
        print(f"    min_delta: ${winner['min_delta']}, total: ${winner['total_delta']}")
    elif all_results:
        # No passing combo — report best by min_delta
        best = max(all_results, key=lambda r: r["min_delta"])
        print(f"  [NO] NO all-positive combo found. Best by min_delta:")
        print(f"    ent_coef={best['ent_coef']}, skip_penalty={best['skip_penalty']}")
        print(f"    fold_deltas: {best['fold_deltas']}")
        print(f"    min_delta: ${best['min_delta']}, total: ${best['total_delta']}")

    return {
        "asset": asset,
        "passing": passing_combos[:1] if passing_combos else [],
        "best": max(all_results, key=lambda r: r["min_delta"]) if all_results else None,
        "all_results": all_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC,ETH,SOL,XRP")
    ap.add_argument("--timesteps", type=int, default=500000)
    args = ap.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    all_asset_results = []
    for asset in assets:
        r = sweep_asset(asset, args.timesteps)
        all_asset_results.append(r)

    # Save results to JSON
    output_path = PROJECT_ROOT / "output" / "rl_sweep_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_asset_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  FINAL SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Asset':<6} {'Config':<30} {'Min delta':>10} {'Total':>10} {'Status':<12}")
    for r in all_asset_results:
        if r["passing"]:
            c = r["passing"][0]
            status = "ALL POS [OK]"
            config = f"ent={c['ent_coef']} skip={c['skip_penalty']}"
            print(f"  {r['asset']:<6} {config:<30} ${c['min_delta']:>+9.2f} ${c['total_delta']:>+9.2f} {status}")
        elif r["best"]:
            c = r["best"]
            status = "BEST (not all+)"
            config = f"ent={c['ent_coef']} skip={c['skip_penalty']}"
            print(f"  {r['asset']:<6} {config:<30} ${c['min_delta']:>+9.2f} ${c['total_delta']:>+9.2f} {status}")

    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
