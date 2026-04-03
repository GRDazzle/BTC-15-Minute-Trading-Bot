"""
LSTM sequence feature extraction (v2).

Extracts fixed-length sequences of 1-second price bars from tick buffers.
Shared between training data generation and live inference.

v2 changes:
- 180s sequence (was 120s)
- 15 features per timestep (was 7)
- Added: price_vs_open (#1 XGBoost feature), rolling momentum (5s/10s/30s),
  rolling volatility (10s/30s), VWAP deviation, tick intensity, cumulative buy ratio
v2.1: Bisect-based tick filtering for O(log n) window extraction.
"""
import bisect
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

LSTM_SEQ_LEN = 180  # 3 minutes of 1-second bars
LSTM_BAR_SECONDS = 1  # 1-second bars
LSTM_FEATURE_NAMES = [
    # Price path (3)
    "return_1s",
    "cumulative_return",
    "price_vs_open",
    # Multi-scale momentum (3)
    "momentum_5s",
    "momentum_10s",
    "momentum_30s",
    # Volatility (2)
    "volatility_10s",
    "volatility_30s",
    # Volume/order flow (4)
    "volume_1s",
    "buy_ratio_1s",
    "cumulative_buy_ratio",
    "tick_intensity_10s",
    # Price structure (1)
    "vwap_deviation",
    # Time (2)
    "hour_sin",
    "hour_cos",
    # Market condition (3)
    "choppiness_30s",
    "volume_60s",
    "vol_acceleration",
    # Crash detection (3)
    "flips_per_tick_180s",
    "momentum_strength_180s",
    "volume_180s",
]
LSTM_NUM_FEATURES = len(LSTM_FEATURE_NAMES)


def extract_lstm_sequence(
    tick_buffer: list[dict],
    timestamp: datetime,
    decision_minute: int = 0,
    seq_len: int = LSTM_SEQ_LEN,
    window_open_price: float | None = None,
) -> Optional[np.ndarray]:
    """Extract a (seq_len, LSTM_NUM_FEATURES) array from tick_buffer.

    Resamples ticks into 1-second bars ending at `timestamp`.
    Returns None if insufficient data (< seq_len/4 actual bars with ticks).
    Missing bars are forward-filled from the last known price.

    Args:
        tick_buffer: List of {"ts": datetime, "price": float, "qty": float, "is_buyer": bool}.
        timestamp: Current evaluation timestamp (UTC).
        decision_minute: dm value within the 15-min window.
        seq_len: Number of 1-second bars to extract.
        window_open_price: Price at minute 0 of the window (for price_vs_open).

    Returns:
        numpy array of shape (seq_len, LSTM_NUM_FEATURES), or None.
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    # Define the time window
    start_time = timestamp - timedelta(seconds=seq_len)

    # Build sorted index and use bisect for O(log n) window extraction
    if not tick_buffer:
        return None

    # Ensure UTC and build timestamp list for bisect
    ts_list = []
    for tick in tick_buffer:
        ts = tick["ts"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts_list.append(ts)

    lo = bisect.bisect_left(ts_list, start_time)
    hi = bisect.bisect_right(ts_list, timestamp)

    relevant_ticks = []
    for i in range(lo, hi):
        tick = tick_buffer[i]
        ts = ts_list[i]
        relevant_ticks.append({
            "ts": ts,
            "price": float(tick["price"]),
            "qty": float(tick.get("qty", 0)),
            "is_buyer": bool(tick.get("is_buyer", False)),
        })

    if not relevant_ticks:
        return None

    # Resample into 1-second bars
    bars = []
    last_price = relevant_ticks[0]["price"]
    tick_idx = 0

    for i in range(seq_len):
        bar_start = start_time + timedelta(seconds=i)
        bar_end = bar_start + timedelta(seconds=1)

        bar_volume = 0.0
        bar_buy_volume = 0.0
        bar_tick_count = 0
        bar_price = last_price  # forward-fill
        had_tick = False

        while tick_idx < len(relevant_ticks) and relevant_ticks[tick_idx]["ts"] < bar_end:
            tick = relevant_ticks[tick_idx]
            if tick["ts"] >= bar_start:
                bar_price = tick["price"]
                bar_volume += tick["qty"]
                bar_tick_count += 1
                if tick["is_buyer"]:
                    bar_buy_volume += tick["qty"]
                had_tick = True
            tick_idx += 1

        bars.append({
            "price": bar_price,
            "volume": bar_volume,
            "buy_volume": bar_buy_volume,
            "tick_count": bar_tick_count,
            "had_tick": had_tick,
        })
        last_price = bar_price

    # Check minimum data density (minimal — sparse Binance.US may have very few ticks)
    bars_with_ticks = sum(1 for b in bars if b["had_tick"])
    if bars_with_ticks < 1:
        return None

    # Pre-compute price array for rolling calculations
    prices = np.array([b["price"] for b in bars], dtype=np.float64)
    volumes = np.array([b["volume"] for b in bars], dtype=np.float64)
    buy_volumes = np.array([b["buy_volume"] for b in bars], dtype=np.float64)
    tick_counts = np.array([b["tick_count"] for b in bars], dtype=np.float64)

    # Pre-compute 1s returns for volatility
    returns_1s = np.zeros(seq_len, dtype=np.float64)
    for i in range(1, seq_len):
        if prices[i - 1] != 0:
            returns_1s[i] = (prices[i] - prices[i - 1]) / prices[i - 1]

    # Compute features
    sequence = np.zeros((seq_len, LSTM_NUM_FEATURES), dtype=np.float32)
    first_price = prices[0]

    # Time features (constant across sequence)
    hour = timestamp.hour
    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)

    # Running VWAP
    cum_vol = 0.0
    cum_vq = 0.0
    # Running buy ratio
    cum_buy_vol = 0.0
    cum_total_vol = 0.0

    for i in range(seq_len):
        price = prices[i]

        # --- return_1s ---
        sequence[i, 0] = returns_1s[i]

        # --- cumulative_return ---
        if first_price != 0:
            sequence[i, 1] = (price - first_price) / first_price

        # --- price_vs_open ---
        if window_open_price is not None and window_open_price != 0:
            sequence[i, 2] = (price - window_open_price) / window_open_price

        # --- momentum_5s ---
        if i >= 5 and prices[i - 5] != 0:
            sequence[i, 3] = (price - prices[i - 5]) / prices[i - 5]

        # --- momentum_10s ---
        if i >= 10 and prices[i - 10] != 0:
            sequence[i, 4] = (price - prices[i - 10]) / prices[i - 10]

        # --- momentum_30s ---
        if i >= 30 and prices[i - 30] != 0:
            sequence[i, 5] = (price - prices[i - 30]) / prices[i - 30]

        # --- volatility_10s (std of returns over last 10 bars) ---
        if i >= 10:
            window_rets = returns_1s[i - 9:i + 1]
            sequence[i, 6] = float(np.std(window_rets))

        # --- volatility_30s ---
        if i >= 30:
            window_rets = returns_1s[i - 29:i + 1]
            sequence[i, 7] = float(np.std(window_rets))

        # --- volume_1s (log-scaled) ---
        sequence[i, 8] = math.log1p(volumes[i])

        # --- buy_ratio_1s ---
        sequence[i, 9] = buy_volumes[i] / volumes[i] if volumes[i] > 0 else 0.5

        # --- cumulative_buy_ratio ---
        cum_buy_vol += buy_volumes[i]
        cum_total_vol += volumes[i]
        sequence[i, 10] = cum_buy_vol / cum_total_vol if cum_total_vol > 0 else 0.5

        # --- tick_intensity_10s (rolling count of ticks in last 10s) ---
        lo = max(0, i - 9)
        sequence[i, 11] = float(tick_counts[lo:i + 1].sum())

        # --- vwap_deviation ---
        cum_vol += volumes[i]
        cum_vq += volumes[i] * price
        if cum_vol > 0:
            vwap = cum_vq / cum_vol
            if vwap != 0:
                sequence[i, 12] = (price - vwap) / vwap

        # --- hour_sin ---
        sequence[i, 13] = hour_sin

        # --- hour_cos ---
        sequence[i, 14] = hour_cos

        # --- choppiness_30s (direction flips per tick in last 30 bars) ---
        if i >= 30:
            window_prices = prices[i - 29:i + 1]
            flips = 0
            for j in range(2, len(window_prices)):
                prev_d = window_prices[j - 1] - window_prices[j - 2]
                curr_d = window_prices[j] - window_prices[j - 1]
                if prev_d * curr_d < 0:
                    flips += 1
            sequence[i, 15] = flips / (len(window_prices) - 2)

        # --- volume_60s (rolling 60-bar log volume) ---
        lo_v = max(0, i - 59)
        sequence[i, 16] = math.log1p(float(volumes[lo_v:i + 1].sum()))

        # --- vol_acceleration (10s vol / 30s vol ratio) ---
        if i >= 30:
            vol_10 = float(np.std(returns_1s[max(0, i - 9):i + 1]))
            vol_30 = float(np.std(returns_1s[i - 29:i + 1]))
            sequence[i, 17] = vol_10 / vol_30 if vol_30 > 0 else 1.0

        # --- flips_per_tick_180s (direction flips over full sequence window) ---
        window_len = min(i + 1, 180)
        if window_len >= 3:
            wp = prices[max(0, i - 179):i + 1]
            fl = 0
            for j in range(2, len(wp)):
                pd = wp[j - 1] - wp[j - 2]
                cd = wp[j] - wp[j - 1]
                if pd * cd < 0:
                    fl += 1
            sequence[i, 18] = fl / (len(wp) - 2)

        # --- momentum_strength_180s (|net move| / range over full window) ---
        if window_len >= 2:
            wp = prices[max(0, i - 179):i + 1]
            net = float(wp[-1] - wp[0])
            rng = float(wp.max() - wp.min())
            sequence[i, 19] = abs(net) / rng if rng > 0 else 0.0

        # --- volume_180s (rolling 180-bar log volume) ---
        lo_v180 = max(0, i - 179)
        sequence[i, 20] = math.log1p(float(volumes[lo_v180:i + 1].sum()))

    return sequence
