"""RL Meta-Controller v3: expanded state, continuous sizing, pure profit reward.

Changes from v2:
- State: 8 → 14 dims (price trajectory, raw xgb_p, hour, volatility, kalshi volume)
- Action: continuous [0, 1] — 0=skip, >0=enter with fraction of max contracts
- Reward: pure PnL (no shaping tricks, just profit)
- Data augmentation: noise on observations during training
- Keep v2's skip penalty
"""
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MetaControllerV3(gym.Env):
    """RL meta-controller v3: XGB direction, RL decides timing + sizing."""

    metadata = {"render_modes": []}
    STATE_DIM = 14

    def __init__(
        self,
        windows: list[dict] = None,
        max_contracts: int = 10,
        fee_cents: float = 2.0,
        min_price: int = 55,
        max_price: int = 80,
        xgb_threshold: float = 0.75,
        skip_penalty: float = -0.5,
        augment: bool = False,
        min_dm: int = 0,
    ):
        super().__init__()
        # Filter checkpoints to dm >= min_dm so RL only sees decision points
        # matching live strategy's min_dm gate (prevents train/infer mismatch).
        if min_dm > 0 and windows:
            filtered = []
            for w in windows:
                cps = [cp for cp in w.get("checkpoints", []) if cp.get("dm", 0) >= min_dm]
                if cps:
                    filtered.append({**w, "checkpoints": cps})
            self.windows = filtered
        else:
            self.windows = windows or []
        self.max_contracts = max_contracts
        self.fee_cents = fee_cents
        self.min_price = min_price
        self.max_price = max_price
        self.xgb_threshold = xgb_threshold
        self.skip_penalty = skip_penalty
        self.augment = augment  # training-time noise
        self.min_dm = min_dm

        # Continuous action: 0 = skip/wait, (0, 1] = enter with fraction of max contracts
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(self.STATE_DIM,), dtype=np.float32,
        )

        self._window_idx = 0
        self._step_idx = 0
        self._current_window = None
        self._entered = False
        self._entry_price = 0
        self._entry_side = None
        self._entry_contracts = 0
        self._price_history = []  # track prices within this window

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.windows:
            return np.zeros(self.STATE_DIM, dtype=np.float32), {}

        self._current_window = self.windows[self._window_idx % len(self.windows)]
        self._window_idx += 1
        self._step_idx = 0
        self._entered = False
        self._entry_price = 0
        self._entry_side = None
        self._entry_contracts = 0
        self._price_history = []

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        w = self._current_window
        cps = w["checkpoints"]

        if self._step_idx >= len(cps):
            reward = self._settle()
            return np.zeros(self.STATE_DIM, dtype=np.float32), reward, True, False, {}

        cp = cps[self._step_idx]
        xgb_p = cp.get("xgb_p", 0.5)

        # XGB direction
        if xgb_p >= self.xgb_threshold:
            side = "yes"
            price = cp["yes_ask"]
        elif xgb_p <= 1 - self.xgb_threshold:
            side = "no"
            price = cp["no_ask"]
        else:
            side = None
            price = 50

        # Track price history
        self._price_history.append(price)

        # Continuous action: value > 0.3 = ENTER, value = sizing fraction
        action_val = float(action[0]) if hasattr(action, '__len__') else float(action)

        reward = 0.0
        if action_val > 0.3 and not self._entered and side is not None:
            if self.min_price <= price <= self.max_price:
                self._entered = True
                self._entry_price = price
                self._entry_side = side
                # Continuous sizing: action [0.3, 1.0] maps to [1, max_contracts]
                frac = (action_val - 0.3) / 0.7  # 0 to 1
                self._entry_contracts = max(1, int(frac * self.max_contracts))

        self._step_idx += 1
        terminated = self._step_idx >= len(cps)

        if terminated:
            reward = self._settle()

        obs = self._get_obs() if not terminated else np.zeros(self.STATE_DIM, dtype=np.float32)
        return obs, reward, terminated, False, {"pnl": reward}

    def _settle(self) -> float:
        """Pure PnL reward — no shaping."""
        if not self._entered:
            had_signal = any(
                abs(cp.get("xgb_p", 0.5) - 0.5) >= (self.xgb_threshold - 0.5)
                and self.min_price <= cp.get("yes_ask", 50) <= self.max_price
                for cp in self._current_window.get("checkpoints", [])
            )
            return self.skip_penalty if had_signal else 0.0

        outcome = self._current_window["outcome"]
        won = self._entry_side == outcome
        cost = (self._entry_price / 100.0 + self.fee_cents / 100.0) * self._entry_contracts
        return (self._entry_contracts * 1.0 - cost) if won else -cost

    def _get_obs(self) -> np.ndarray:
        w = self._current_window
        cps = w["checkpoints"]

        if self._step_idx >= len(cps):
            return np.zeros(self.STATE_DIM, dtype=np.float32)

        cp = cps[self._step_idx]
        xgb_p = cp.get("xgb_p", 0.5)

        # Side price
        if xgb_p >= 0.5:
            side_price = cp["yes_ask"]
        else:
            side_price = cp["no_ask"]

        # Price trajectory (last 3 prices, padded)
        ph = self._price_history
        p_curr = side_price
        p_prev1 = ph[-1] if len(ph) >= 1 else p_curr
        p_prev2 = ph[-2] if len(ph) >= 2 else p_prev1
        price_delta_1 = (p_curr - p_prev1) / 10.0  # cents / 10
        price_delta_2 = (p_prev1 - p_prev2) / 10.0

        # Hour of day (sin/cos for cyclical)
        try:
            from datetime import datetime, timezone
            ws = w.get("window_start", "")
            hour = datetime.fromisoformat(ws).hour if ws else 12
        except Exception:
            hour = 12
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)

        obs = np.array([
            xgb_p * 2 - 1,                                   # raw xgb_p [-1, 1]
            abs(xgb_p - 0.5) * 2,                            # confidence [0, 1]
            side_price / 100.0,                               # price [0, 1]
            cp.get("mins_to_close", 7.5) / 15.0,             # time left [0, 1]
            cp.get("dm", 5) / 9.0,                            # dm [0, 1]
            cp.get("spread", 2) / 10.0,                       # spread [0, ~1]
            price_delta_1,                                     # price trend 1-step
            price_delta_2,                                     # price trend 2-step
            1.0 if self._entered else 0.0,                    # already entered
            cp.get("price_vs_open", 0) * 100,                 # amplified
            cp.get("velocity_900s", 0) * 100,                  # amplified
            cp.get("z_score_900s", 0),                         # already scaled
            hour_sin,                                          # cyclical hour
            hour_cos,
        ], dtype=np.float32)

        # Data augmentation: add noise during training
        if self.augment:
            noise = np.random.normal(0, 0.02, size=obs.shape).astype(np.float32)
            # Don't augment binary features
            noise[8] = 0  # has_entered
            obs = obs + noise

        return np.clip(obs, -5.0, 5.0)
