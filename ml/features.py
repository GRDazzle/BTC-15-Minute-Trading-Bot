"""
Feature extraction for XGBoost meta-model (v6 - trimmed).

Extracts 22 features from tick_buffer, price_history, timestamp, and
rule-based processor outputs. Used by both training data generation and
live/backtest ML inference.

v6: Trimmed from 42 to 22 features. Removed 20 features with <1% mean
importance across all 4 assets (BTC, ETH, SOL, XRP).
v6.1: Bisect-based lookups for O(log n) price and tick window queries.
"""
import bisect
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from core.strategy_brain.signal_processors.base_processor import TradingSignal

# Ordered feature names (must match training and inference)
FEATURE_NAMES = [
    # Tick-derived (10)
    "velocity_30s",
    "velocity_60s",
    "volatility_30s",
    "volatility_60s",
    "volume_30s",
    "volume_60s",
    "buy_volume_ratio_60s",
    "aggressor_ratio_60s",
    "tick_intensity_30s",
    "large_trade_count",
    # Price structure (2)
    "vwap_deviation",
    "price_range_20",
    # Time (3)
    "hour_sin",
    "hour_cos",
    "minute_in_window",
    # Signal features (2)
    "return_skew_60s",
    "price_vs_open",
    # Longer lookback (5)
    "velocity_300s",
    "velocity_900s",
    "volatility_300s",
    "buy_volume_ratio_300s",
    "momentum_trend",
    # Market condition (4)
    "volume_180s",
    "choppiness_60s",
    "range_pct_180s",
    "vol_acceleration",
    # Crash detection (5)
    "flips_per_tick_180s",
    "momentum_strength_180s",
    "price_vs_sma5",
    "price_vs_sma15",
    "price_vs_sma30",
]


def compute_daily_smas(
    daily_closes: list[tuple[str, float]],
    target_date: str,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute SMA 5, 15, 30 from daily closing prices.

    Args:
        daily_closes: List of (date_str, close_price) sorted by date ascending.
        target_date: Date string (YYYY-MM-DD) to compute SMAs for.

    Returns:
        (sma5, sma15, sma30) — None if insufficient data for that period.
    """
    # Get prices up to (but not including) target_date
    prices = [p for d, p in daily_closes if d < target_date]

    sma5 = sum(prices[-5:]) / 5 if len(prices) >= 5 else None
    sma15 = sum(prices[-15:]) / 15 if len(prices) >= 15 else None
    sma30 = sum(prices[-30:]) / 30 if len(prices) >= 30 else None

    return sma5, sma15, sma30


def load_daily_closes(data_dir, asset: str) -> list[tuple[str, float]]:
    """Load daily closing prices from aggTrade CSVs.

    Returns list of (date_str, close_price) sorted by date.
    """
    import csv as _csv
    from pathlib import Path

    asset_dir = Path(data_dir) / asset.upper()
    if not asset_dir.exists():
        return []

    daily = {}
    for f in sorted(asset_dir.glob("*-aggTrades-*.csv")):
        date_str = f.stem.split("aggTrades-")[1]
        # Read last line to get closing price
        last_price = None
        try:
            with open(f, "r") as fh:
                fh.seek(0, 2)
                fsize = fh.tell()
                fh.seek(max(0, fsize - 1024))
                lines = fh.read().splitlines()
                for line in reversed(lines):
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            last_price = float(parts[1])
                            break
                        except ValueError:
                            continue
        except Exception:
            continue
        if last_price is not None:
            daily[date_str] = last_price

    return sorted(daily.items())


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def build_tick_index(tick_buffer: list[dict]) -> tuple[list[datetime], list[dict]]:
    """Build sorted timestamp index for bisect lookups. O(n log n).

    Call once per checkpoint, then pass to _get_price_at_offset and
    _ticks_in_window for O(log n) queries instead of O(n).

    Returns:
        (sorted_timestamps, sorted_ticks)
    """
    pairs = [(_ensure_utc(t["ts"]), t) for t in tick_buffer]
    pairs.sort(key=lambda x: x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _get_price_at_offset(tick_buffer, seconds_ago: float, now: datetime,
                          ts_index=None, sorted_ticks=None) -> Optional[float]:
    """Find tick price closest to `seconds_ago` before `now`, within 15s tolerance.

    O(log n) when ts_index/sorted_ticks provided, O(n) fallback otherwise.
    """
    now = _ensure_utc(now)
    target = now - timedelta(seconds=seconds_ago)

    if ts_index is not None and sorted_ticks is not None:
        idx = bisect.bisect_left(ts_index, target)
        best_price = None
        best_diff = float("inf")
        for i in (idx - 1, idx):
            if 0 <= i < len(ts_index):
                diff = abs((ts_index[i] - target).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_price = float(sorted_ticks[i]["price"])
        return best_price if best_diff <= 15 else None

    # Fallback: O(n) linear scan
    best = None
    best_diff = float("inf")
    for tick in tick_buffer:
        ts = _ensure_utc(tick["ts"])
        diff = abs((ts - target).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best = float(tick["price"])
    return best if best_diff <= 15 else None


def _ticks_in_window(tick_buffer, seconds: float, now: datetime,
                      ts_index=None, sorted_ticks=None) -> list[dict]:
    """Return ticks within the last `seconds` from `now`.

    O(log n) when ts_index/sorted_ticks provided, O(n) fallback otherwise.
    """
    now = _ensure_utc(now)
    cutoff = now - timedelta(seconds=seconds)

    if ts_index is not None and sorted_ticks is not None:
        lo = bisect.bisect_left(ts_index, cutoff)
        hi = bisect.bisect_right(ts_index, now)
        return sorted_ticks[lo:hi]

    result = []
    for tick in tick_buffer:
        ts = _ensure_utc(tick["ts"])
        if ts >= cutoff:
            result.append(tick)
    return result


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
    ts_index: list[datetime] = None,
    sorted_ticks: list[dict] = None,
    sma5: Optional[float] = None,
    sma15: Optional[float] = None,
    sma30: Optional[float] = None,
) -> dict[str, float]:
    """Extract 31 features for one decision point.

    For fast batch processing, pass ts_index and sorted_ticks from
    build_tick_index(). For live inference with small buffers, omit them.
    SMA values should be precomputed from daily closing prices.
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    features: dict[str, float] = {}

    # ---- Price offsets (O(log n) with index) ----
    price_30s = _get_price_at_offset(tick_buffer, 30, timestamp, ts_index, sorted_ticks)
    price_60s = _get_price_at_offset(tick_buffer, 60, timestamp, ts_index, sorted_ticks)
    price_300s = _get_price_at_offset(tick_buffer, 300, timestamp, ts_index, sorted_ticks)
    price_900s = _get_price_at_offset(tick_buffer, 900, timestamp, ts_index, sorted_ticks)

    features["velocity_30s"] = (
        (current_price - price_30s) / price_30s if price_30s and price_30s != 0 else 0.0
    )
    features["velocity_60s"] = (
        (current_price - price_60s) / price_60s if price_60s and price_60s != 0 else 0.0
    )

    # ---- Tick windows (O(log n) with index) ----
    ticks_30 = _ticks_in_window(tick_buffer, 30, timestamp, ts_index, sorted_ticks)
    ticks_60 = _ticks_in_window(tick_buffer, 60, timestamp, ts_index, sorted_ticks)

    features["volatility_30s"] = _compute_return_volatility(ticks_30)
    features["volatility_60s"] = _compute_return_volatility(ticks_60)

    # Volume
    vol_60 = sum(t.get("qty", 0) for t in ticks_60)
    features["volume_30s"] = sum(t.get("qty", 0) for t in ticks_30)
    features["volume_60s"] = vol_60

    # Buy volume ratio
    buy_60 = sum(t.get("qty", 0) for t in ticks_60 if t.get("is_buyer", False))
    features["buy_volume_ratio_60s"] = buy_60 / vol_60 if vol_60 > 0 else 0.5

    # Aggressor ratio
    buy_count_60 = sum(1 for t in ticks_60 if t.get("is_buyer", False))
    features["aggressor_ratio_60s"] = buy_count_60 / len(ticks_60) if ticks_60 else 0.5

    # Tick intensity
    features["tick_intensity_30s"] = float(len(ticks_30))

    # Large trade count
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
        window = hist[-20:]
        rng = max(window) - min(window)
        mean = sum(window) / len(window)
        features["price_range_20"] = rng / mean if mean != 0 else 0.0
    else:
        features["price_range_20"] = 0.0

    # ---- Time features ----
    hour = timestamp.hour
    features["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
    features["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    features["minute_in_window"] = float(decision_minute)

    # ---- Signal features ----
    features["return_skew_60s"] = _compute_return_skewness(ticks_60)

    if window_open_price is not None and window_open_price != 0:
        features["price_vs_open"] = (current_price - window_open_price) / window_open_price
    else:
        features["price_vs_open"] = 0.0

    # ---- Longer lookback features ----
    features["velocity_300s"] = (
        (current_price - price_300s) / price_300s if price_300s and price_300s != 0 else 0.0
    )
    features["velocity_900s"] = (
        (current_price - price_900s) / price_900s if price_900s and price_900s != 0 else 0.0
    )

    ticks_300 = _ticks_in_window(tick_buffer, 300, timestamp, ts_index, sorted_ticks)
    features["volatility_300s"] = _compute_return_volatility(ticks_300)

    vol_300 = sum(t.get("qty", 0) for t in ticks_300)
    buy_300 = sum(t.get("qty", 0) for t in ticks_300 if t.get("is_buyer", False))
    features["buy_volume_ratio_300s"] = buy_300 / vol_300 if vol_300 > 0 else 0.5

    features["momentum_trend"] = features["velocity_60s"] - features["velocity_300s"]

    # ---- Market condition features ----
    ticks_180 = _ticks_in_window(tick_buffer, 180, timestamp, ts_index, sorted_ticks)

    # Volume over 3 minutes (bridges 60s and 300s gap)
    features["volume_180s"] = sum(t.get("qty", 0) for t in ticks_180)

    # Choppiness: direction flips per tick in last 60s
    # High = trending (consistent direction), Low = choppy/noisy
    if len(ticks_60) >= 3:
        prices_60 = [t["price"] for t in ticks_60]
        flips = 0
        for i in range(2, len(prices_60)):
            prev_dir = prices_60[i - 1] - prices_60[i - 2]
            curr_dir = prices_60[i] - prices_60[i - 1]
            if prev_dir * curr_dir < 0:
                flips += 1
        features["choppiness_60s"] = flips / (len(prices_60) - 2)
    else:
        features["choppiness_60s"] = 0.0

    # Price range as % of mean over 3 minutes
    if ticks_180:
        prices_180 = [t["price"] for t in ticks_180]
        mean_180 = sum(prices_180) / len(prices_180)
        features["range_pct_180s"] = (
            (max(prices_180) - min(prices_180)) / mean_180 if mean_180 != 0 else 0.0
        )
    else:
        features["range_pct_180s"] = 0.0

    # Volatility acceleration: recent vol / longer-term vol
    # >1 = volatility increasing, <1 = calming down
    vol_30 = features["volatility_30s"]
    vol_300 = features["volatility_300s"]
    features["vol_acceleration"] = vol_30 / vol_300 if vol_300 > 0 else 1.0

    # ---- Crash detection features ----

    # Direction flips per tick over 3 minutes (longer-term choppiness)
    if len(ticks_180) >= 3:
        prices_180_list = [t["price"] for t in ticks_180]
        flips_180 = 0
        for i in range(2, len(prices_180_list)):
            prev_dir = prices_180_list[i - 1] - prices_180_list[i - 2]
            curr_dir = prices_180_list[i] - prices_180_list[i - 1]
            if prev_dir * curr_dir < 0:
                flips_180 += 1
        features["flips_per_tick_180s"] = flips_180 / (len(prices_180_list) - 2)
    else:
        features["flips_per_tick_180s"] = 0.0

    # Momentum strength: |net move| / range over 3 minutes
    # High = strong trend, Low = choppy (moved a lot but ended near start)
    if ticks_180 and len(ticks_180) >= 2:
        p_180 = [t["price"] for t in ticks_180]
        net_move = p_180[-1] - p_180[0]
        price_range = max(p_180) - min(p_180)
        features["momentum_strength_180s"] = abs(net_move) / price_range if price_range > 0 else 0.0
    else:
        features["momentum_strength_180s"] = 0.0

    # SMA features: current price relative to moving averages
    # Negative = price below SMA = bearish trend
    if sma5 is not None and sma5 != 0:
        features["price_vs_sma5"] = (current_price - sma5) / sma5
    else:
        features["price_vs_sma5"] = 0.0

    if sma15 is not None and sma15 != 0:
        features["price_vs_sma15"] = (current_price - sma15) / sma15
    else:
        features["price_vs_sma15"] = 0.0

    if sma30 is not None and sma30 != 0:
        features["price_vs_sma30"] = (current_price - sma30) / sma30
    else:
        features["price_vs_sma30"] = 0.0

    return features
