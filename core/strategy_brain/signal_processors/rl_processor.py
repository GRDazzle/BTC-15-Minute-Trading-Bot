"""RL meta-controller for live inference.

Takes XGB's prediction + Kalshi state + time info, returns:
- enter: bool (True if RL says enter now, False if wait/skip)
- contracts: int (1-max_contracts, 0 if skipping)

Matches the state representation used during training in rl/meta_controller_v3.py.
"""
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


class RLProcessor:
    """Meta-controller: given XGB prediction, decide WHEN to enter and HOW MUCH."""

    def __init__(self, asset: str, model_dir: Path | str = Path("models"), max_contracts: int = 10):
        self.asset = asset.upper()
        self.max_contracts = max_contracts
        self.model_path = Path(model_dir) / f"{self.asset}_rl_prod.zip"
        if not self.model_path.exists():
            raise FileNotFoundError(f"RL model not found: {self.model_path}")

        from stable_baselines3 import PPO
        self.model = PPO.load(str(self.model_path), device="cpu")

        # Per-window state — reset at window boundary
        self.price_history: list[int] = []
        self.entered_this_window: bool = False

    def reset_window(self):
        """Called at window boundary."""
        self.price_history = []
        self.entered_this_window = False

    def reload(self) -> None:
        """Reload PPO weights from disk. Preserves per-window state."""
        from stable_baselines3 import PPO
        self.model = PPO.load(str(self.model_path), device="cpu")

    def decide(
        self,
        xgb_p: float,
        yes_ask: int,
        no_ask: int,
        spread: int,
        mins_to_close: float,
        dm: int,
        price_vs_open: float,
        velocity_900s: float,
        z_score_900s: float,
        hour_utc: int,
    ) -> tuple[bool, int]:
        """Return (enter, contracts) decision.

        Returns:
            enter: True if RL wants to enter now
            contracts: 1-max_contracts if entering, 0 otherwise
        """
        if self.entered_this_window:
            return False, 0  # already entered

        # XGB direction
        if xgb_p >= 0.5:
            side_price = yes_ask
        else:
            side_price = no_ask

        # Track price history
        self.price_history.append(side_price)

        # Build state vector matching training (14 dims)
        ph = self.price_history
        p_prev1 = ph[-2] if len(ph) >= 2 else ph[-1]
        p_prev2 = ph[-3] if len(ph) >= 3 else p_prev1
        price_delta_1 = (side_price - p_prev1) / 10.0
        price_delta_2 = (p_prev1 - p_prev2) / 10.0

        hour_sin = math.sin(2 * math.pi * hour_utc / 24.0)
        hour_cos = math.cos(2 * math.pi * hour_utc / 24.0)

        obs = np.array([
            xgb_p * 2 - 1,
            abs(xgb_p - 0.5) * 2,
            side_price / 100.0,
            mins_to_close / 15.0,
            dm / 9.0,
            spread / 10.0,
            price_delta_1,
            price_delta_2,
            0.0,  # has_entered (we only ask RL before entering)
            price_vs_open * 100,
            velocity_900s * 100,
            z_score_900s,
            hour_sin,
            hour_cos,
        ], dtype=np.float32)
        obs = np.clip(obs, -5.0, 5.0)

        action, _ = self.model.predict(obs, deterministic=True)
        action_val = float(action[0]) if hasattr(action, "__len__") else float(action)

        # Continuous action interpretation (same as training env):
        # [0, 0.3] = wait, (0.3, 1.0] = enter with scaled contracts
        if action_val > 0.3:
            frac = (action_val - 0.3) / 0.7
            contracts = max(1, int(frac * self.max_contracts))
            self.entered_this_window = True
            return True, contracts
        return False, 0
