"""
Feature extraction for XGBoost meta-model.

Extracts 36 features from tick_buffer, price_history, timestamp, and
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
    # Polymarket-inspired features (5)
    "obv_pvt_divergence",
    "vpt_trend",
    "decay_buy_ratio_60s",
    "mtf_convergence",
    "delta_vol_vs_avg",
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


def _compute_obv_pvt_divergence(ticks: list[dict]) -> float:
    """OBV vs PVT momentum residual.

    OBV = cumsum(qty if price_up, -qty if price_down)
    PVT = cumsum((price_change / prev_price) * qty)
    Normalize both, take residual, compute momentum (recent third vs early third).
    Catches when volume commitment doesn't match price change magnitude.
    """
    if len(ticks) < 3:
        return 0.0
    obv = [0.0]
    pvt = [0.0]
    for i in range(1, len(ticks)):
        price = ticks[i]["price"]
        prev_price = ticks[i - 1]["price"]
        qty = ticks[i].get("qty", 1.0)
        price_change = price - prev_price
        # OBV: full volume in direction of price change
        if price_change > 0:
            obv.append(obv[-1] + qty)
        elif price_change < 0:
            obv.append(obv[-1] - qty)
        else:
            obv.append(obv[-1])
        # PVT: proportional volume
        if prev_price != 0:
            pvt.append(pvt[-1] + (price_change / prev_price) * qty)
        else:
            pvt.append(pvt[-1])
    # Normalize both to [0, 1] range
    obv_range = max(obv) - min(obv)
    pvt_range = max(pvt) - min(pvt)
    if obv_range == 0 or pvt_range == 0:
        return 0.0
    obv_norm = [(v - min(obv)) / obv_range for v in obv]
    pvt_norm = [(v - min(pvt)) / pvt_range for v in pvt]
    # Residual
    residual = [o - p for o, p in zip(obv_norm, pvt_norm)]
    # Momentum: mean of recent third minus mean of early third
    n = len(residual)
    third = max(1, n // 3)
    early = sum(residual[:third]) / third
    recent = sum(residual[-third:]) / third
    return recent - early


def _compute_vpt_trend(ticks: list[dict]) -> float:
    """VPT vs its short-term moving average, normalized by VPT range.

    VPT = cumsum(return * qty). Compare current VPT to window average.
    Positive = volume-weighted price momentum accelerating.
    """
    if len(ticks) < 4:
        return 0.0
    vpt = [0.0]
    for i in range(1, len(ticks)):
        price = ticks[i]["price"]
        prev_price = ticks[i - 1]["price"]
        qty = ticks[i].get("qty", 1.0)
        if prev_price != 0:
            ret = (price - prev_price) / prev_price
        else:
            ret = 0.0
        vpt.append(vpt[-1] + ret * qty)
    vpt_range = max(vpt) - min(vpt)
    if vpt_range == 0:
        return 0.0
    vpt_mean = sum(vpt) / len(vpt)
    trend = (vpt[-1] - vpt_mean) / vpt_range
    return max(-1.0, min(1.0, trend))


def _compute_decay_buy_ratio(ticks: list[dict], decay: float = 0.75) -> float:
    """Recency-weighted buy volume ratio with exponential decay.

    Newest tick gets highest weight. decay=0.75 means newest tick weight
    is ~18x the oldest tick weight in a 60s window with ~40 ticks.
    """
    if not ticks:
        return 0.5
    n = len(ticks)
    weighted_buy = 0.0
    weighted_total = 0.0
    for i, tick in enumerate(ticks):
        weight = decay ** (n - 1 - i)
        qty = tick.get("qty", 1.0)
        weighted_total += qty * weight
        if tick.get("is_buyer", False):
            weighted_buy += qty * weight
    if weighted_total == 0:
        return 0.5
    return weighted_buy / weighted_total


def _compute_mtf_convergence(ticks: list[dict], now, timedelta_cls=timedelta) -> float:
    """Multi-timeframe agreement score using 15s and 45s sub-windows.

    Each sub-signal: price direction + volume direction agreement.
    Both bullish = 1.0, both bearish = 0.0, mixed = 0.5.
    """
    if len(ticks) < 4:
        return 0.5

    def _sub_signal(sub_ticks: list[dict]) -> float:
        """Returns 1.0 (bullish), -1.0 (bearish), or 0.0 (neutral)."""
        if len(sub_ticks) < 2:
            return 0.0
        price_dir = 1.0 if sub_ticks[-1]["price"] > sub_ticks[0]["price"] else (
            -1.0 if sub_ticks[-1]["price"] < sub_ticks[0]["price"] else 0.0
        )
        buy_vol = sum(t.get("qty", 1.0) for t in sub_ticks if t.get("is_buyer", False))
        sell_vol = sum(t.get("qty", 1.0) for t in sub_ticks if not t.get("is_buyer", False))
        vol_dir = 1.0 if buy_vol > sell_vol else (-1.0 if sell_vol > buy_vol else 0.0)
        if price_dir == vol_dir and price_dir != 0:
            return price_dir
        return 0.0

    cutoff_15 = now - timedelta_cls(seconds=15)
    cutoff_45 = now - timedelta_cls(seconds=45)
    ticks_15 = [t for t in ticks if t["ts"] >= cutoff_15]
    ticks_45 = [t for t in ticks if t["ts"] >= cutoff_45]

    sig_15 = _sub_signal(ticks_15)
    sig_45 = _sub_signal(ticks_45)

    if sig_15 > 0 and sig_45 > 0:
        return 1.0
    if sig_15 < 0 and sig_45 < 0:
        return 0.0
    return 0.5


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

    # ---- Polymarket-inspired features ----

    # OBV vs PVT divergence (hidden accumulation/distribution)
    features["obv_pvt_divergence"] = _compute_obv_pvt_divergence(ticks_60)

    # VPT trend (volume-weighted price momentum acceleration)
    features["vpt_trend"] = _compute_vpt_trend(ticks_60)

    # Decay-weighted buy ratio (recency-biased order flow)
    features["decay_buy_ratio_60s"] = _compute_decay_buy_ratio(ticks_60)

    # Multi-timeframe convergence (15s + 45s agreement)
    # Ensure ticks have tz-aware timestamps for comparison
    ticks_for_mtf = []
    for t in ticks_60:
        ts = t["ts"]
        if ts.tzinfo is None:
            t_copy = dict(t)
            t_copy["ts"] = ts.replace(tzinfo=timezone.utc)
            ticks_for_mtf.append(t_copy)
        else:
            ticks_for_mtf.append(t)
    features["mtf_convergence"] = _compute_mtf_convergence(ticks_for_mtf, timestamp)

    # Delta volume vs average (buying pressure change)
    features["delta_vol_vs_avg"] = (
        features["buy_volume_ratio_30s"] - features["buy_volume_ratio_60s"]
    )

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
