"""Gymnasium environment for 15-min Kalshi binary options trading.

Replays historical windows. At each 10s checkpoint (dm=2 to dm=9),
the agent observes features + Kalshi prices and decides: WAIT, BUY_YES, BUY_NO.

State: [xgb_p, lstm_p, kalshi_yes_ask, kalshi_no_ask, kalshi_spread,
        mins_to_close, dm_normalized, has_position, entry_price_norm,
        kalshi_ask_velocity, price_vs_open, velocity_900s, z_score_900s]

Actions: 0=WAIT, 1=BUY_YES, 2=BUY_NO

Reward: PnL at settlement (sparse — only on last step or when position taken).
"""
import csv
import json
import bisect
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class KalshiTradingEnv(gym.Env):
    """Gym environment for one asset's 15-min Kalshi windows."""

    metadata = {"render_modes": []}

    # State dimension
    STATE_DIM = 13
    # Actions: 0=WAIT, 1=BUY_YES, 2=BUY_NO
    N_ACTIONS = 3

    def __init__(
        self,
        asset: str = "BTC",
        windows: list[dict] = None,
        max_contracts: int = 10,
        fee_cents: float = 2.0,
        min_price: int = 55,
        max_price: int = 80,
    ):
        super().__init__()
        self.asset = asset
        self.windows = windows or []
        self.max_contracts = max_contracts
        self.fee_cents = fee_cents
        self.min_price = min_price
        self.max_price = max_price

        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.STATE_DIM,), dtype=np.float32,
        )

        self._window_idx = 0
        self._step_idx = 0
        self._current_window = None
        self._position = None  # None, "yes", "no"
        self._entry_price = 0
        self._entry_contracts = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.windows:
            return np.zeros(self.STATE_DIM, dtype=np.float32), {}

        # Pick next window (sequential for training stability)
        self._current_window = self.windows[self._window_idx % len(self.windows)]
        self._window_idx += 1
        self._step_idx = 0
        self._position = None
        self._entry_price = 0
        self._entry_contracts = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        w = self._current_window
        checkpoints = w["checkpoints"]

        if self._step_idx >= len(checkpoints):
            # Episode over
            reward = self._settle()
            return np.zeros(self.STATE_DIM, dtype=np.float32), reward, True, False, {}

        cp = checkpoints[self._step_idx]

        reward = 0.0
        if action == 1 and self._position is None:  # BUY YES
            price = cp["yes_ask"]
            if self.min_price <= price <= self.max_price:
                self._position = "yes"
                self._entry_price = price
                self._entry_contracts = self.max_contracts
        elif action == 2 and self._position is None:  # BUY NO
            price = cp["no_ask"]
            if self.min_price <= price <= self.max_price:
                self._position = "no"
                self._entry_price = price
                self._entry_contracts = self.max_contracts

        self._step_idx += 1

        # Check if episode is done (last checkpoint)
        terminated = self._step_idx >= len(checkpoints)
        if terminated:
            reward = self._settle()

        obs = self._get_obs() if not terminated else np.zeros(self.STATE_DIM, dtype=np.float32)
        return obs, reward, terminated, False, {"pnl": reward}

    def _settle(self) -> float:
        """Compute PnL at settlement."""
        if self._position is None:
            return 0.0  # didn't trade

        w = self._current_window
        outcome = w["outcome"]  # "yes" or "no"
        won = self._position == outcome

        cost = (self._entry_price / 100.0 + self.fee_cents / 100.0) * self._entry_contracts
        if won:
            return self._entry_contracts * 1.0 - cost
        else:
            return -cost

    def _get_obs(self) -> np.ndarray:
        """Build observation vector from current checkpoint."""
        w = self._current_window
        checkpoints = w["checkpoints"]

        if self._step_idx >= len(checkpoints):
            return np.zeros(self.STATE_DIM, dtype=np.float32)

        cp = checkpoints[self._step_idx]

        # Normalize features to roughly [-1, 1] range
        obs = np.array([
            cp.get("xgb_p", 0.5) * 2 - 1,           # [-1, 1]
            cp.get("lstm_p", 0.5) * 2 - 1,           # [-1, 1]
            cp.get("yes_ask", 50) / 100.0 * 2 - 1,   # [-1, 1]
            cp.get("no_ask", 50) / 100.0 * 2 - 1,    # [-1, 1]
            cp.get("spread", 2) / 10.0,              # ~[0, 1]
            cp.get("mins_to_close", 7.5) / 15.0,     # [0, 1]
            cp.get("dm", 5) / 9.0,                   # [0, 1]
            1.0 if self._position is not None else 0.0,
            self._entry_price / 100.0 if self._position else 0.0,
            cp.get("ask_velocity", 0) / 10.0,        # normalized
            cp.get("price_vs_open", 0) * 100,         # amplified
            cp.get("velocity_900s", 0) * 100,          # amplified
            cp.get("z_score_900s", 0),                 # already scaled
        ], dtype=np.float32)

        return np.clip(obs, -10.0, 10.0)


def load_training_windows(
    asset: str,
    days: int = 45,
    polls_dir: Path = None,
    features_csv: Path = None,
) -> list[dict]:
    """Load historical windows with XGB predictions + Kalshi polls for RL training.

    Returns list of window dicts, each with:
        {window_start, outcome, checkpoints: [{dm, xgb_p, lstm_p, yes_ask, ...}]}
    """
    if polls_dir is None:
        polls_dir = PROJECT_ROOT / "data" / "kalshi_polls"
    if features_csv is None:
        features_csv = PROJECT_ROOT / "ml" / "training_data" / f"{asset}_features_stacked.csv"

    from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker

    kalshi_idx = KalshiPollIndex(polls_dir, asset)

    # Load XGB features (with predictions if available, else use raw features)
    # Group by window_start, collect checkpoints
    windows_data = {}
    with open(features_csv) as f:
        for row in csv.DictReader(f):
            ws = row.get("window_start", "")
            if not ws:
                continue
            dm = int(float(row.get("minute_in_window", 0)))
            xgb_p = float(row.get("lstm_p", 0.5))  # stacked prediction
            price_vs_open = float(row.get("price_vs_open", 0))
            vel_900 = float(row.get("velocity_900s", 0))
            z_900 = float(row.get("z_score_900s", 0))

            if ws not in windows_data:
                windows_data[ws] = {"window_start": ws, "checkpoints": []}
            windows_data[ws]["checkpoints"].append({
                "dm": dm,
                "xgb_p": xgb_p,
                "price_vs_open": price_vs_open,
                "velocity_900s": vel_900,
                "z_score_900s": z_900,
            })

    # Enrich with Kalshi poll data and outcomes
    windows = []
    for ws, w in windows_data.items():
        try:
            ws_dt = datetime.fromisoformat(ws)
            we_dt = ws_dt + timedelta(minutes=15)
        except Exception:
            continue

        event_ticker = window_start_to_event_ticker(asset, we_dt)
        outcome = kalshi_idx.get_outcome(event_ticker)
        if outcome is None:
            continue

        w["outcome"] = outcome

        # Enrich each checkpoint with Kalshi prices
        for cp in w["checkpoints"]:
            entry_time = ws_dt + timedelta(minutes=5 + cp["dm"])
            poll = kalshi_idx.find_poll(event_ticker, entry_time)
            if poll:
                cp["yes_ask"] = poll["yes_ask"]
                cp["no_ask"] = poll.get("no_ask", 100 - poll.get("yes_bid", 50))
                cp["spread"] = poll["yes_ask"] - poll.get("yes_bid", poll["yes_ask"])
                cp["mins_to_close"] = poll.get("mins_to_close", 7.5)
                # Kalshi RT features
                history = kalshi_idx.get_poll_history(event_ticker, entry_time, lookback_seconds=30)
                if history and len(history) >= 2:
                    cp["ask_velocity"] = history[-1].get("yes_ask", 50) - history[0].get("yes_ask", 50)
                else:
                    cp["ask_velocity"] = 0
                cp["lstm_p"] = cp.get("xgb_p", 0.5)  # use stacked prediction
            else:
                cp["yes_ask"] = 50
                cp["no_ask"] = 50
                cp["spread"] = 2
                cp["mins_to_close"] = 7.5
                cp["ask_velocity"] = 0
                cp["lstm_p"] = 0.5

        # Sort checkpoints by dm
        w["checkpoints"].sort(key=lambda x: x["dm"])

        # Filter to only dm >= 2
        w["checkpoints"] = [cp for cp in w["checkpoints"] if cp["dm"] >= 2]

        if w["checkpoints"]:
            windows.append(w)

    return windows
