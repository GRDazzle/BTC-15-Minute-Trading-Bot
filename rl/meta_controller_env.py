"""RL Meta-Controller: XGB decides direction, RL decides WHEN to enter.

At each checkpoint, XGB has already computed p_bullish. The RL agent
only decides: WAIT or ENTER_NOW. Direction comes from XGB.

This is a much simpler task than full RL — the agent only learns
price-timing, not direction prediction.

State (8 dims):
    xgb_confidence    - abs(xgb_p - 0.5) * 2, how sure XGB is [0, 1]
    entry_price_norm  - current price / 100 for the XGB-chosen side [0, 1]
    mins_to_close     - normalized [0, 1]
    dm_normalized     - decision minute / 9 [0, 1]
    spread_norm       - kalshi spread / 10 [0, 1]
    price_change      - price movement since first checkpoint (cents)
    has_entered       - 1 if already entered, 0 if not
    best_price_so_far - best (lowest) price seen so far for XGB's side

Actions: 0=WAIT, 1=ENTER_NOW
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MetaControllerEnv(gym.Env):
    """RL agent decides WHEN to enter. XGB decides direction."""

    metadata = {"render_modes": []}
    STATE_DIM = 8

    def __init__(
        self,
        windows: list[dict] = None,
        max_contracts: int = 10,
        fee_cents: float = 2.0,
        min_price: int = 55,
        max_price: int = 80,
        xgb_threshold: float = 0.75,
        skip_penalty: float = -0.5,
    ):
        super().__init__()
        self.windows = windows or []
        self.max_contracts = max_contracts
        self.fee_cents = fee_cents
        self.min_price = min_price
        self.max_price = max_price
        self.xgb_threshold = xgb_threshold
        self.skip_penalty = skip_penalty  # penalty for not trading when XGB had a signal

        self.action_space = spaces.Discrete(2)  # 0=WAIT, 1=ENTER
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
        self._first_price = None
        self._best_price = 999

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
        self._first_price = None
        self._best_price = 999

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        w = self._current_window
        cps = w["checkpoints"]

        if self._step_idx >= len(cps):
            reward = self._settle()
            return np.zeros(self.STATE_DIM, dtype=np.float32), reward, True, False, {}

        cp = cps[self._step_idx]
        xgb_p = cp.get("xgb_p", 0.5)

        # Determine XGB's direction
        if xgb_p >= self.xgb_threshold:
            side = "yes"
            price = cp["yes_ask"]
        elif xgb_p <= 1 - self.xgb_threshold:
            side = "no"
            price = cp["no_ask"]
        else:
            side = None
            price = 50

        # Track best price and first price
        if side and self._first_price is None:
            self._first_price = price
        if side and price < self._best_price:
            self._best_price = price

        reward = 0.0

        # Agent action
        if action == 1 and not self._entered and side is not None:
            if self.min_price <= price <= self.max_price:
                self._entered = True
                self._entry_price = price
                self._entry_side = side
                # Small intermediate reward for entering at a good price
                # (below average entry would be positive signal)
                if self._first_price and self._first_price > 0:
                    price_edge = (self._first_price - price) / 100.0
                    reward = price_edge * 0.5  # small shaped reward

        self._step_idx += 1
        terminated = self._step_idx >= len(cps)

        if terminated:
            reward += self._settle()

        obs = self._get_obs() if not terminated else np.zeros(self.STATE_DIM, dtype=np.float32)
        return obs, reward, terminated, False, {}

    def _settle(self) -> float:
        if not self._entered:
            # Penalty for skipping when XGB had a qualifying signal
            # Prevents agent from collapsing to "always wait"
            had_signal = any(
                abs(cp.get("xgb_p", 0.5) - 0.5) >= (self.xgb_threshold - 0.5)
                and self.min_price <= cp.get("yes_ask", 50) <= self.max_price
                for cp in self._current_window.get("checkpoints", [])
            )
            return self.skip_penalty if had_signal else 0.0

        outcome = self._current_window["outcome"]
        won = self._entry_side == outcome
        cost = (self._entry_price / 100.0 + self.fee_cents / 100.0) * self.max_contracts
        return (self.max_contracts * 1.0 - cost) if won else -cost

    def _get_obs(self) -> np.ndarray:
        w = self._current_window
        cps = w["checkpoints"]

        if self._step_idx >= len(cps):
            return np.zeros(self.STATE_DIM, dtype=np.float32)

        cp = cps[self._step_idx]
        xgb_p = cp.get("xgb_p", 0.5)

        # XGB confidence (direction-agnostic)
        confidence = abs(xgb_p - 0.5) * 2.0

        # Price for XGB's preferred side
        if xgb_p >= 0.5:
            side_price = cp["yes_ask"]
        else:
            side_price = cp["no_ask"]

        # Price change since first qualifying checkpoint
        price_change = 0.0
        if self._first_price and self._first_price > 0:
            price_change = (side_price - self._first_price) / 10.0  # cents / 10

        obs = np.array([
            confidence,                                      # [0, 1]
            side_price / 100.0,                              # [0, 1]
            cp.get("mins_to_close", 7.5) / 15.0,            # [0, 1]
            cp.get("dm", 5) / 9.0,                           # [0, 1]
            cp.get("spread", 2) / 10.0,                      # [0, ~1]
            price_change,                                     # [-5, 5] ish
            1.0 if self._entered else 0.0,                   # binary
            (self._best_price - side_price) / 10.0 if self._best_price < 999 else 0.0,
        ], dtype=np.float32)

        return np.clip(obs, -5.0, 5.0)
