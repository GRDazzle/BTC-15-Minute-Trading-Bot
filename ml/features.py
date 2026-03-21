"""
Feature extraction for XGBoost meta-model.

Extracts 31 features from tick_buffer, price_history, timestamp, and
rule-based processor outputs. Used by both training data generation and
live/backtest ML inference.
"""
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from core.strategy_brain.signal_processors.base_processor import (
    TradingSignal,
    SignalDirection,
)

# Ordered feature names (must match training and inference)
FEATURE_NAMES = [
    # Tick-derived (16)
    "velocity_10s",
    "velocity_30s",
    "velocity_60s",
    "acceleration",
    "volatility_30s",
    "volatility_60s",
    "volume_30s",
    "volume_60s",
    "volume_acceleration",
    "buy_volume_ratio_30s",
    "buy_volume_ratio_60s",
    "aggressor_ratio_30s",
    "aggressor_ratio_60s",
    "tick_intensity_30s",
    "large_trade_count",
    "vwap_deviation",
    # Price-history (4)
    "ma_deviation_20",
    "momentum_5",
    "momentum_10",
    "price_range_20",
    # Time (3) -- cyclical hour encoding + minute_in_window
    "hour_sin",
    "hour_cos",
    "minute_in_window",
    # Rule-based processor outputs (2) -- tickvel only (spike is dead for non-BTC)
    "tickvel_direction",
    "tickvel_confidence",
    # New features (6 - replaced dead spike_direction/spike_confidence)
    "rsi_14",
    "bollinger_pos",
    "velocity_5s",
    "return_skew_60s",
    "price_vs_open",
    # Cross-asset (1)
    "btc_velocity_60s",
]


def _direction_to_num(direction: SignalDirection) -> float:
    d = str(direction).upper()
    if "BULLISH" in d:
        return 1.0
    elif "BEARISH" in d:
        return -1.0
    return 0.0


def _get_price_at_offset(tick_buffer: list[dict], seconds_ago: float, now: datetime) -> Optional[float]:
    """Find tick price closest to `seconds_ago` before `now`, within 15s tolerance."""
    target = now - timedelta(seconds=seconds_ago)
    best = None
    best_diff = float("inf")
    for tick in tick_buffer:
        ts = tick["ts"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        diff = abs((ts - target).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best = float(tick["price"])
    if best_diff <= 15:
        return best
    return None


def _ticks_in_window(tick_buffer: list[dict], seconds: float, now: datetime) -> list[dict]:
    """Return ticks within the last `seconds` from `now`."""
    cutoff = now - timedelta(seconds=seconds)
    result = []
    for tick in tick_buffer:
        ts = tick["ts"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            result.append(tick)
    return result


def _compute_rsi(prices: list[float], period: int = 14) -> float:
    """Compute RSI from a list of prices. Returns 50.0 if insufficient data."""
    if len(prices) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(len(prices) - period, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_bollinger_pos(prices: list[float], period: int = 20) -> float:
    """Compute position within Bollinger Bands (0-1 scale).

    Returns 0.5 if insufficient data or zero-width bands.
    """
    if len(prices) < period:
        return 0.5
    window = prices[-period:]
    mean = sum(window) / len(window)
    variance = sum((p - mean) ** 2 for p in window) / len(window)
    std = variance ** 0.5
    if std == 0:
        return 0.5
    upper = mean + 2 * std
    lower = mean - 2 * std
    band_width = upper - lower
    if band_width == 0:
        return 0.5
    pos = (prices[-1] - lower) / band_width
    # Clamp to [0, 1] -- price can be outside bands
    return max(0.0, min(1.0, pos))


def _compute_return_skewness(ticks: list[dict]) -> float:
    """Compute skewness of sequential returns from tick prices."""
    if len(ticks) < 3:
        return 0.0
    prices = [t["price"] for t in ticks]
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] != 0:
            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
    n = len(returns)
    if n < 3:
        return 0.0
    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / n
    std = variance ** 0.5
    if std == 0:
        return 0.0
    skew = sum((r - mean_r) ** 3 for r in returns) / (n * std ** 3)
    return skew


def extract_features(
    tick_buffer: list[dict],
    price_history: list,
    current_price: float,
    timestamp: datetime,
    spike_signal: Optional[TradingSignal] = None,
    tickvel_signal: Optional[TradingSignal] = None,
    decision_minute: int = 0,
    window_open_price: Optional[float] = None,
    btc_velocity_60s: Optional[float] = None,
) -> dict[str, float]:
    """Extract all features for one decision point.

    Args:
        tick_buffer: List of {"ts": datetime, "price": float, ...} dicts.
                     May also contain "qty" and "is_buyer" keys from aggTrades.
        price_history: List of recent prices (Decimal or float).
        current_price: Current price as float.
        timestamp: Current evaluation timestamp (UTC).
        spike_signal: Output from SpikeDetector (unused, kept for API compat).
        tickvel_signal: Output from TickVelocityProcessor (None if no signal).
        decision_minute: dm value (0-9) within the 15-min window.
        window_open_price: Price at start of decision zone (minute 5). For price_vs_open.
        btc_velocity_60s: BTC 60s velocity for cross-asset feature. None for BTC itself.

    Returns:
        Dict mapping feature name -> float value.
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    features: dict[str, float] = {}

    # ---- Tick-derived features ----
    price_5s = _get_price_at_offset(tick_buffer, 5, timestamp)
    price_10s = _get_price_at_offset(tick_buffer, 10, timestamp)
    price_30s = _get_price_at_offset(tick_buffer, 30, timestamp)
    price_60s = _get_price_at_offset(tick_buffer, 60, timestamp)

    features["velocity_10s"] = (
        (current_price - price_10s) / price_10s if price_10s and price_10s != 0 else 0.0
    )
    features["velocity_30s"] = (
        (current_price - price_30s) / price_30s if price_30s and price_30s != 0 else 0.0
    )
    features["velocity_60s"] = (
        (current_price - price_60s) / price_60s if price_60s and price_60s != 0 else 0.0
    )

    # Acceleration: is the 30s move speeding up vs the first 30s?
    vel_first_30s = features["velocity_60s"] - features["velocity_30s"]
    features["acceleration"] = features["velocity_30s"] - vel_first_30s

    # Volatility: std dev of sequential returns
    ticks_30 = _ticks_in_window(tick_buffer, 30, timestamp)
    ticks_60 = _ticks_in_window(tick_buffer, 60, timestamp)

    features["volatility_30s"] = _compute_return_volatility(ticks_30)
    features["volatility_60s"] = _compute_return_volatility(ticks_60)

    # Volume features (only available with aggTrade data that has "qty")
    vol_30 = sum(t.get("qty", 0) for t in ticks_30)
    vol_60 = sum(t.get("qty", 0) for t in ticks_60)
    features["volume_30s"] = vol_30
    features["volume_60s"] = vol_60
    features["volume_acceleration"] = vol_30 / vol_60 if vol_60 > 0 else 1.0

    # Buy volume ratio (order flow)
    buy_30 = sum(t.get("qty", 0) for t in ticks_30 if t.get("is_buyer", False))
    buy_60 = sum(t.get("qty", 0) for t in ticks_60 if t.get("is_buyer", False))
    features["buy_volume_ratio_30s"] = buy_30 / vol_30 if vol_30 > 0 else 0.5
    features["buy_volume_ratio_60s"] = buy_60 / vol_60 if vol_60 > 0 else 0.5

    # Aggressor ratio (trade-count-based order flow)
    buy_count_30 = sum(1 for t in ticks_30 if t.get("is_buyer", False))
    buy_count_60 = sum(1 for t in ticks_60 if t.get("is_buyer", False))
    features["aggressor_ratio_30s"] = buy_count_30 / len(ticks_30) if ticks_30 else 0.5
    features["aggressor_ratio_60s"] = buy_count_60 / len(ticks_60) if ticks_60 else 0.5

    # Tick intensity (number of ticks in last 30s)
    features["tick_intensity_30s"] = float(len(ticks_30))

    # Large trade count: trades > 2x average qty in last 60s
    if ticks_60:
        qtys = [t.get("qty", 0) for t in ticks_60]
        avg_qty = sum(qtys) / len(qtys) if qtys else 0
        features["large_trade_count"] = float(
            sum(1 for q in qtys if avg_qty > 0 and q > 2.0 * avg_qty)
        )
    else:
        features["large_trade_count"] = 0.0

    # VWAP deviation
    if ticks_60:
        total_vq = sum(t["price"] * t.get("qty", 1) for t in ticks_60)
        total_q = sum(t.get("qty", 1) for t in ticks_60)
        vwap = total_vq / total_q if total_q > 0 else current_price
        features["vwap_deviation"] = (current_price - vwap) / vwap if vwap != 0 else 0.0
    else:
        features["vwap_deviation"] = 0.0

    # ---- Price-history features ----
    hist = [float(p) for p in price_history]

    if len(hist) >= 20:
        ma20 = sum(hist[-20:]) / 20.0
        features["ma_deviation_20"] = (current_price - ma20) / ma20 if ma20 != 0 else 0.0
    else:
        features["ma_deviation_20"] = 0.0

    if len(hist) >= 5:
        old5 = hist[-5]
        features["momentum_5"] = (current_price - old5) / old5 if old5 != 0 else 0.0
    else:
        features["momentum_5"] = 0.0

    if len(hist) >= 10:
        old10 = hist[-10]
        features["momentum_10"] = (current_price - old10) / old10 if old10 != 0 else 0.0
    else:
        features["momentum_10"] = 0.0

    if len(hist) >= 20:
        window = hist[-20:]
        rng = max(window) - min(window)
        mean = sum(window) / len(window)
        features["price_range_20"] = rng / mean if mean != 0 else 0.0
    else:
        features["price_range_20"] = 0.0

    # ---- Time features (cyclical encoding) ----
    hour = timestamp.hour
    features["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
    features["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    features["minute_in_window"] = float(decision_minute)

    # ---- Rule-based processor outputs (tickvel only) ----
    if tickvel_signal is not None:
        features["tickvel_direction"] = _direction_to_num(tickvel_signal.direction)
        features["tickvel_confidence"] = tickvel_signal.confidence
    else:
        features["tickvel_direction"] = 0.0
        features["tickvel_confidence"] = 0.0

    # ---- New features ----

    # RSI-14 from price history
    features["rsi_14"] = _compute_rsi(hist, period=14)

    # Bollinger Band position (0-1)
    features["bollinger_pos"] = _compute_bollinger_pos(hist, period=20)

    # Ultra-short velocity (5-second)
    features["velocity_5s"] = (
        (current_price - price_5s) / price_5s if price_5s and price_5s != 0 else 0.0
    )

    # Return skewness over 60s (asymmetric moves)
    features["return_skew_60s"] = _compute_return_skewness(ticks_60)

    # Price vs window open (intra-window trend)
    if window_open_price is not None and window_open_price != 0:
        features["price_vs_open"] = (current_price - window_open_price) / window_open_price
    else:
        features["price_vs_open"] = 0.0

    # Cross-asset BTC velocity
    features["btc_velocity_60s"] = btc_velocity_60s if btc_velocity_60s is not None else 0.0

    return features


def _compute_return_volatility(ticks: list[dict]) -> float:
    """Compute std dev of sequential returns from tick prices."""
    if len(ticks) < 2:
        return 0.0
    prices = [t["price"] for t in ticks]
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] != 0:
            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
    if not returns:
        return 0.0
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    return variance ** 0.5
