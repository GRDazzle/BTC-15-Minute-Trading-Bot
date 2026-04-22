"""Walk-forward evaluation for RL meta-controller v3.

Instead of one 80/20 split, trains and evaluates across multiple
chronological folds — same methodology as XGB walk-forward.

For each fold:
  1. Train RL on all data before the fold
  2. Evaluate on the fold's test period
  3. Compare RL vs XGB baseline on same test data

Usage:
    python scripts/eval_rl_walkforward.py --asset BTC,ETH,SOL,XRP --n-folds 4
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_fold_boundaries(windows, n_folds, min_train_days=14):
    """Split windows into walk-forward folds by date."""
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


def train_and_eval_fold(asset, train_windows, test_windows, fold_idx, n_folds, timesteps):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.meta_controller_v3 import MetaControllerV3

    print(f"    Fold {fold_idx + 1}/{n_folds}: train={len(train_windows)}, test={len(test_windows)}")

    train_env = DummyVecEnv([lambda: MetaControllerV3(
        windows=train_windows, augment=True,
    )])

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=1024, batch_size=256,
        n_epochs=10, gamma=0.99, ent_coef=0.02, clip_range=0.2,
        device="cpu", verbose=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - t0

    # Evaluate
    eval_env = MetaControllerV3(windows=test_windows, augment=False)

    rl_pnl = 0.0
    rl_trades = 0
    rl_wins = 0
    bl_pnl = 0.0
    bl_trades = 0
    bl_wins = 0

    for w in test_windows:
        # RL agent
        eval_env._current_window = w
        eval_env._step_idx = 0
        eval_env._entered = False
        eval_env._entry_price = 0
        eval_env._entry_side = None
        eval_env._entry_contracts = 0
        eval_env._price_history = []
        obs = eval_env._get_obs()

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = eval_env.step(action)

        if eval_env._entered:
            rl_trades += 1
            settle = eval_env._settle()
            rl_pnl += settle
            if settle > 0:
                rl_wins += 1

        # XGB baseline
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
                break

    rl_wr = 100 * rl_wins / rl_trades if rl_trades else 0
    bl_wr = 100 * bl_wins / bl_trades if bl_trades else 0
    delta = rl_pnl - bl_pnl

    print(f"      RL:  {rl_trades:>3} trades, {rl_wr:>5.1f}% WR, ${rl_pnl:>+8.2f}")
    print(f"      XGB: {bl_trades:>3} trades, {bl_wr:>5.1f}% WR, ${bl_pnl:>+8.2f}")
    print(f"      Delta: ${delta:>+8.2f}  ({train_time:.0f}s train)")

    return {
        "rl_pnl": rl_pnl, "rl_trades": rl_trades, "rl_wr": rl_wr,
        "bl_pnl": bl_pnl, "bl_trades": bl_trades, "bl_wr": bl_wr,
        "delta": delta,
    }


def run_walkforward(asset, n_folds, timesteps):
    from rl.trading_env import load_training_windows

    print(f"\n{'='*60}")
    print(f"  Walk-Forward RL Evaluation: {asset}")
    print(f"  {n_folds} folds, {timesteps:,} timesteps/fold")
    print(f"{'='*60}")

    all_windows = load_training_windows(asset)
    print(f"  Total windows: {len(all_windows)}")

    folds = get_fold_boundaries(all_windows, n_folds)
    print(f"  Folds: {len(folds)}")
    for i, (te, ts) in enumerate(folds):
        print(f"    Fold {i + 1}: train <= {te}, test -> {ts}")
    print()

    fold_results = []
    for i, (train_end, test_end) in enumerate(folds):
        train_w = [w for w in all_windows if w["window_start"][:10] <= train_end]
        test_w = [w for w in all_windows
                  if w["window_start"][:10] > train_end and w["window_start"][:10] <= test_end]

        if len(train_w) < 100 or len(test_w) < 20:
            print(f"    Fold {i + 1}: skipped (insufficient data)")
            continue

        result = train_and_eval_fold(asset, train_w, test_w, i, len(folds), timesteps)
        fold_results.append(result)

    if not fold_results:
        print("  No valid folds.")
        return None

    # Aggregate
    rl_total = sum(r["rl_pnl"] for r in fold_results)
    bl_total = sum(r["bl_pnl"] for r in fold_results)
    rl_fold_pnls = [r["rl_pnl"] for r in fold_results]
    bl_fold_pnls = [r["bl_pnl"] for r in fold_results]
    delta_fold_pnls = [r["delta"] for r in fold_results]

    rl_all_positive = all(p > 0 for p in rl_fold_pnls)
    delta_all_positive = all(d > 0 for d in delta_fold_pnls)

    avg_rl_wr = np.mean([r["rl_wr"] for r in fold_results])
    avg_bl_wr = np.mean([r["bl_wr"] for r in fold_results])

    print(f"\n  {'='*50}")
    print(f"  {asset} WALK-FORWARD SUMMARY ({len(fold_results)} folds)")
    print(f"  {'='*50}")
    print(f"  RL total PnL:    ${rl_total:>+9.2f}  fold PnLs: {[round(p,1) for p in rl_fold_pnls]}")
    print(f"  XGB total PnL:   ${bl_total:>+9.2f}  fold PnLs: {[round(p,1) for p in bl_fold_pnls]}")
    print(f"  Delta total:     ${rl_total - bl_total:>+9.2f}  fold deltas: {[round(d,1) for d in delta_fold_pnls]}")
    print(f"  RL avg WR:       {avg_rl_wr:.1f}%")
    print(f"  XGB avg WR:      {avg_bl_wr:.1f}%")
    print(f"  RL all folds +:  {'YES' if rl_all_positive else 'NO'}")
    print(f"  Delta all folds +: {'YES' if delta_all_positive else 'NO'}")

    return {
        "asset": asset,
        "rl_total": rl_total, "bl_total": bl_total,
        "rl_fold_pnls": rl_fold_pnls, "bl_fold_pnls": bl_fold_pnls,
        "delta_fold_pnls": delta_fold_pnls,
        "rl_all_positive": rl_all_positive,
        "delta_all_positive": delta_all_positive,
        "avg_rl_wr": avg_rl_wr, "avg_bl_wr": avg_bl_wr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC", help="Assets (comma-separated)")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=1000000, help="Timesteps per fold")
    args = ap.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    results = []
    for asset in assets:
        r = run_walkforward(asset, args.n_folds, args.timesteps)
        if r:
            results.append(r)

    if results:
        print(f"\n{'='*70}")
        print(f"  FINAL SUMMARY: RL v3 Walk-Forward Evaluation")
        print(f"{'='*70}")
        print(f"  {'Asset':<6} {'RL PnL':>9} {'XGB PnL':>9} {'Delta':>9} {'RL WR':>6} {'XGB WR':>6} {'All+':>5} {'Dlt+':>5}")
        total_rl = total_bl = 0.0
        for r in results:
            delta = r["rl_total"] - r["bl_total"]
            total_rl += r["rl_total"]
            total_bl += r["bl_total"]
            print(f"  {r['asset']:<6} ${r['rl_total']:>+8.2f} ${r['bl_total']:>+8.2f} ${delta:>+8.2f} {r['avg_rl_wr']:>5.1f}% {r['avg_bl_wr']:>5.1f}% {'Y' if r['rl_all_positive'] else 'N':>4} {'Y' if r['delta_all_positive'] else 'N':>4}")
        print(f"  {'TOTAL':<6} ${total_rl:>+8.2f} ${total_bl:>+8.2f} ${total_rl-total_bl:>+8.2f}")


if __name__ == "__main__":
    main()
