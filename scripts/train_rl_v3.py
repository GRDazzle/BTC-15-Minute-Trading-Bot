"""Train RL meta-controller v3: expanded state, continuous sizing, pure profit.

Usage:
    python scripts/train_rl_v3.py --asset BTC,ETH,SOL,XRP --timesteps 1000000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_asset(asset: str, timesteps: int, eval_split: float = 0.2):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.meta_controller_v3 import MetaControllerV3
    from rl.trading_env import load_training_windows

    print(f"\n{'='*60}")
    print(f"  RL Meta-Controller v3: {asset}")
    print(f"  14-dim state, continuous sizing, pure profit reward")
    print(f"{'='*60}")

    print("Loading training windows...")
    all_windows = load_training_windows(asset)
    print(f"  Total windows: {len(all_windows)}")

    if len(all_windows) < 100:
        print(f"  ERROR: Too few windows.")
        return

    split = int(len(all_windows) * (1 - eval_split))
    train_windows = all_windows[:split]
    eval_windows = all_windows[split:]
    print(f"  Train: {len(train_windows)}, Eval: {len(eval_windows)}")

    # Training env WITH augmentation
    train_env = DummyVecEnv([lambda: MetaControllerV3(
        windows=train_windows, augment=True,
    )])

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.02,
        clip_range=0.2,
        device="cpu",
        verbose=0,
    )

    print(f"  Policy params: {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"  Training {timesteps:,} timesteps (augmented)...")

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    print(f"  Training time: {time.time() - t0:.1f}s")

    # Evaluate (NO augmentation)
    print(f"\n  Evaluating on {len(eval_windows)} OOS windows...")
    eval_env = MetaControllerV3(windows=eval_windows, augment=False)

    rl_pnl = 0.0
    rl_trades = 0
    rl_wins = 0
    rl_skips = 0
    rl_contracts_total = 0
    rl_entry_prices = []

    bl_pnl = 0.0
    bl_trades = 0
    bl_wins = 0
    bl_entry_prices = []

    for w in eval_windows:
        # --- RL agent ---
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
            settle_pnl = eval_env._settle()
            rl_pnl += settle_pnl
            if settle_pnl > 0:
                rl_wins += 1
            rl_contracts_total += eval_env._entry_contracts
            rl_entry_prices.append(eval_env._entry_price)
        else:
            rl_skips += 1

        # --- XGB baseline (fixed threshold, fixed 10 contracts) ---
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
                    bl_entry_prices.append(price)
                break

    rl_wr = 100 * rl_wins / rl_trades if rl_trades else 0
    bl_wr = 100 * bl_wins / bl_trades if bl_trades else 0
    rl_avg_price = np.mean(rl_entry_prices) if rl_entry_prices else 0
    bl_avg_price = np.mean(bl_entry_prices) if bl_entry_prices else 0
    rl_avg_contracts = rl_contracts_total / rl_trades if rl_trades else 0

    print(f"\n  {'':>15} {'Trades':>7} {'WR%':>6} {'PnL':>10} {'AvgPrice':>9} {'AvgSize':>8}")
    print(f"  {'RL v3':>15} {rl_trades:>7} {rl_wr:>5.1f}% ${rl_pnl:>+9.2f} {rl_avg_price:>8.1f}c {rl_avg_contracts:>7.1f}")
    print(f"  {'XGB baseline':>15} {bl_trades:>7} {bl_wr:>5.1f}% ${bl_pnl:>+9.2f} {bl_avg_price:>8.1f}c {'10.0':>7}")
    print(f"  {'RL skipped':>15} {rl_skips:>7}")
    delta = rl_pnl - bl_pnl
    print(f"  {'Delta':>15} {'':>7} {'':>6} ${delta:>+9.2f}")

    model_path = PROJECT_ROOT / "models" / f"{asset}_rl_v3.zip"
    model.save(str(model_path))
    print(f"\n  Model saved: {model_path}")

    return {
        "asset": asset,
        "rl_pnl": rl_pnl, "rl_trades": rl_trades, "rl_wr": rl_wr,
        "rl_avg_price": rl_avg_price, "rl_avg_contracts": rl_avg_contracts,
        "bl_pnl": bl_pnl, "bl_trades": bl_trades, "bl_wr": bl_wr,
        "delta": delta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC", help="Assets (comma-separated)")
    ap.add_argument("--timesteps", type=int, default=1000000)
    args = ap.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    results = []
    for asset in assets:
        r = train_asset(asset, args.timesteps)
        if r:
            results.append(r)

    if results:
        print(f"\n{'='*60}")
        print(f"  SUMMARY: RL v3 Meta-Controller vs XGB Baseline")
        print(f"{'='*60}")
        print(f"  {'Asset':<6} {'RL PnL':>9} {'XGB PnL':>9} {'Delta':>9} {'RL WR':>6} {'XGB WR':>6} {'RL Size':>8}")
        for r in results:
            print(f"  {r['asset']:<6} ${r['rl_pnl']:>+8.2f} ${r['bl_pnl']:>+8.2f} ${r['delta']:>+8.2f} {r['rl_wr']:>5.1f}% {r['bl_wr']:>5.1f}% {r['rl_avg_contracts']:>7.1f}")


if __name__ == "__main__":
    main()
