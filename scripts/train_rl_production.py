"""Train production RL models using per-asset winning configs from sweep.

Trains on ALL historical windows (no eval split) and saves to
models/{ASSET}_rl_prod.zip for deployment.

Per-asset configs (from sweep):
  BTC: ent=0.01, skip=-0.2
  ETH: ent=0.01, skip=-0.2
  SOL: ent=0.02, skip=-0.5
  XRP: ent=0.01, skip=-0.5
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Winning per-asset hyperparameters (from sweep_rl_per_asset.py results)
PER_ASSET_CONFIG = {
    "BTC": {"ent_coef": 0.01, "skip_penalty": -0.2},
    "ETH": {"ent_coef": 0.01, "skip_penalty": -0.2},
    "SOL": {"ent_coef": 0.02, "skip_penalty": -0.5},
    "XRP": {"ent_coef": 0.01, "skip_penalty": -0.5},
}


def train_asset(asset: str, timesteps: int, holdout_days: int = 0, suffix: str = "prod"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.meta_controller_v3 import MetaControllerV3
    from rl.trading_env import load_training_windows

    cfg = PER_ASSET_CONFIG.get(asset, {"ent_coef": 0.02, "skip_penalty": -0.5})

    print(f"\n{'='*60}")
    print(f"  {asset}: training RL model ({suffix})")
    print(f"  Config: ent_coef={cfg['ent_coef']}, skip_penalty={cfg['skip_penalty']}")
    if holdout_days > 0:
        print(f"  Holdout: excluding last {holdout_days} days from training")
    print(f"{'='*60}")

    windows = load_training_windows(asset)
    if holdout_days > 0 and windows:
        dates_sorted = sorted(set(w["window_start"][:10] for w in windows))
        if len(dates_sorted) > holdout_days:
            cutoff = dates_sorted[-holdout_days]
            pre = [w for w in windows if w["window_start"][:10] < cutoff]
            print(f"  Filtered to {len(pre)} pre-holdout windows (cutoff={cutoff})")
            windows = pre
    print(f"  Training on {len(windows)} windows")

    env = DummyVecEnv([lambda: MetaControllerV3(
        windows=windows,
        augment=True,
        skip_penalty=cfg["skip_penalty"],
        min_dm=7,
    )])

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=1024, batch_size=256,
        n_epochs=10, gamma=0.99,
        ent_coef=cfg["ent_coef"], clip_range=0.2,
        device="cpu", verbose=0,
    )

    print(f"  Training {timesteps:,} timesteps...")
    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    model_path = PROJECT_ROOT / "models" / f"{asset}_rl_{suffix}.zip"
    model.save(str(model_path))
    print(f"  Saved: {model_path}")

    meta_path = PROJECT_ROOT / "models" / f"{asset}_rl_{suffix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "asset": asset,
            "config": cfg,
            "timesteps": timesteps,
            "train_windows": len(windows),
            "training_time_s": round(elapsed, 1),
            "feature_version": "v12_37features",
            "rl_version": "v3_14dim_continuous",
            "min_dm": 7,
            "holdout_days": holdout_days,
            "suffix": suffix,
        }, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC,ETH,SOL,XRP")
    ap.add_argument("--timesteps", type=int, default=1000000)
    ap.add_argument("--holdout-days", type=int, default=0,
                    help="Exclude last N days from training (for OOS test)")
    ap.add_argument("--suffix", default="prod",
                    help="Model file suffix: models/{ASSET}_rl_{suffix}.zip")
    args = ap.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    total_t0 = time.time()
    for asset in assets:
        train_asset(asset, args.timesteps, args.holdout_days, args.suffix)

    print(f"\n{'='*60}")
    print(f"  ALL PRODUCTION RL MODELS TRAINED ({time.time()-total_t0:.0f}s total)")
    print(f"{'='*60}")
    for asset in assets:
        cfg = PER_ASSET_CONFIG.get(asset, {})
        print(f"  {asset}: models/{asset}_rl_prod.zip  ({cfg})")


if __name__ == "__main__":
    main()
