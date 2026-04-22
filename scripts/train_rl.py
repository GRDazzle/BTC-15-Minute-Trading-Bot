"""Train RL agent for Kalshi 15-min binary options trading.

Uses PPO from stable-baselines3 on historical window data.
The agent learns WHEN to enter (wait vs buy) and WHICH SIDE (yes vs no).

Usage:
    python scripts/train_rl.py --asset BTC --timesteps 200000
    python scripts/train_rl.py --asset BTC,ETH,SOL,XRP --timesteps 500000
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
    from rl.trading_env import KalshiTradingEnv, load_training_windows

    print(f"\n{'='*60}")
    print(f"  RL Training: {asset}")
    print(f"{'='*60}")

    # Load historical windows
    print("Loading training windows...")
    all_windows = load_training_windows(asset)
    print(f"  Total windows with outcomes: {len(all_windows)}")

    if len(all_windows) < 100:
        print(f"  ERROR: Too few windows ({len(all_windows)}). Need 100+.")
        return

    # Split train/eval chronologically
    split = int(len(all_windows) * (1 - eval_split))
    train_windows = all_windows[:split]
    eval_windows = all_windows[split:]
    print(f"  Train: {len(train_windows)} windows, Eval: {len(eval_windows)} windows")

    # Count outcomes
    train_yes = sum(1 for w in train_windows if w["outcome"] == "yes")
    print(f"  Train outcomes: {train_yes} YES / {len(train_windows) - train_yes} NO")

    # Create environments
    train_env = DummyVecEnv([lambda: KalshiTradingEnv(asset=asset, windows=train_windows)])
    eval_env = KalshiTradingEnv(asset=asset, windows=eval_windows)

    # PPO agent
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=1.0,         # no discounting — reward only at episode end
        ent_coef=0.05,     # encourage exploration
        clip_range=0.2,
        verbose=0,
    )

    param_count = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy parameters: {param_count:,}")
    print(f"  Training for {timesteps:,} timesteps...")

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    # Evaluate on OOS windows
    print(f"\n  Evaluating on {len(eval_windows)} OOS windows...")
    rl_pnl = 0.0
    rl_trades = 0
    rl_wins = 0
    rl_skips = 0
    baseline_pnl = 0.0
    baseline_trades = 0
    baseline_wins = 0

    for w in eval_windows:
        # RL agent plays the window
        obs, _ = eval_env.reset()
        eval_env._current_window = w
        eval_env._step_idx = 0
        eval_env._position = None
        obs = eval_env._get_obs()

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(int(action))

        if eval_env._position is not None:
            rl_trades += 1
            rl_pnl += reward
            if reward > 0:
                rl_wins += 1
        else:
            rl_skips += 1

        # Baseline: XGB threshold strategy (enter at first dm with p >= 0.75)
        for cp in w["checkpoints"]:
            p = cp.get("xgb_p", 0.5)
            if p >= 0.75 or p <= 0.25:
                side = "yes" if p >= 0.75 else "no"
                price = cp["yes_ask"] if side == "yes" else cp["no_ask"]
                if 55 <= price <= 80:
                    won = side == w["outcome"]
                    cost = (price / 100.0 + 0.02) * 10
                    bpnl = (10.0 - cost) if won else -cost
                    baseline_pnl += bpnl
                    baseline_trades += 1
                    if bpnl > 0:
                        baseline_wins += 1
                break

    rl_wr = 100 * rl_wins / rl_trades if rl_trades else 0
    bl_wr = 100 * baseline_wins / baseline_trades if baseline_trades else 0

    print(f"\n  {'':>15} {'Trades':>8} {'WR%':>6} {'PnL':>10}")
    print(f"  {'RL agent':>15} {rl_trades:>8} {rl_wr:>5.1f}% ${rl_pnl:>+9.2f}")
    print(f"  {'XGB baseline':>15} {baseline_trades:>8} {bl_wr:>5.1f}% ${baseline_pnl:>+9.2f}")
    print(f"  {'RL skipped':>15} {rl_skips:>8}")
    delta = rl_pnl - baseline_pnl
    print(f"  {'Delta':>15} {'':>8} {'':>6} ${delta:>+9.2f}")

    # Save model
    model_path = PROJECT_ROOT / "models" / f"{asset}_rl_ppo.zip"
    model.save(str(model_path))
    print(f"\n  Model saved: {model_path}")

    return {
        "asset": asset,
        "rl_pnl": rl_pnl, "rl_trades": rl_trades, "rl_wr": rl_wr,
        "baseline_pnl": baseline_pnl, "baseline_trades": baseline_trades, "baseline_wr": bl_wr,
        "delta": delta,
    }


def main():
    ap = argparse.ArgumentParser(description="Train RL agent for Kalshi trading")
    ap.add_argument("--asset", default="BTC", help="Assets (comma-separated)")
    ap.add_argument("--timesteps", type=int, default=200000, help="Training timesteps")
    args = ap.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    results = []
    for asset in assets:
        r = train_asset(asset, args.timesteps)
        if r:
            results.append(r)

    if results:
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Asset':<6} {'RL PnL':>10} {'XGB PnL':>10} {'Delta':>10} {'RL WR':>6} {'XGB WR':>6}")
        for r in results:
            print(f"  {r['asset']:<6} ${r['rl_pnl']:>+9.2f} ${r['baseline_pnl']:>+9.2f} ${r['delta']:>+9.2f} {r['rl_wr']:>5.1f}% {r['baseline_wr']:>5.1f}%")


if __name__ == "__main__":
    main()
