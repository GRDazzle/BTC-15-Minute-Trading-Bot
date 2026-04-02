"""Analyze market conditions during winning vs losing trades.

Compares Coinbase tick data characteristics between wins and losses
to identify patterns that could be used as trade filters.
"""
import csv
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import statistics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtester.data_loader_ticks import load_aggtrades_multi

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
TRADES_CSV = PROJECT_ROOT / "output" / "trades.csv"


def load_trades():
    """Load trade records from CSV."""
    trades = []
    with open(TRADES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["pnl"]:
                continue
            trades.append({
                "ts": datetime.fromisoformat(row["timestamp"]),
                "asset": row["asset"],
                "direction": row["direction"],
                "price": int(row["price_cents"]),
                "dm": int(row["dm"]),
                "pnl": float(row["pnl"]),
                "won": float(row["pnl"]) > 0,
                "window_id": row["window_id"],
            })
    return trades


def analyze_ticks_around_trade(ticks, trade_ts, window_start):
    """Compute market condition metrics from tick data around a trade entry.

    Looks at:
    - 3 minutes before entry (immediate conditions)
    - 5 minutes before entry (broader context)
    - Full window before entry (warmup + decision zone)
    """
    trade_epoch = trade_ts.timestamp()

    # Get ticks in different lookback windows
    ticks_3m = [t for t in ticks if trade_epoch - 180 <= t.ts.timestamp() < trade_epoch]
    ticks_5m = [t for t in ticks if trade_epoch - 300 <= t.ts.timestamp() < trade_epoch]
    ticks_1m = [t for t in ticks if trade_epoch - 60 <= t.ts.timestamp() < trade_epoch]

    if len(ticks_3m) < 3:
        return None

    prices_3m = [t.price for t in ticks_3m]
    prices_5m = [t.price for t in ticks_5m] if ticks_5m else prices_3m
    prices_1m = [t.price for t in ticks_1m] if ticks_1m else prices_3m[:5]

    metrics = {}

    # 1. Tick density (trades per minute)
    metrics["ticks_per_min_3m"] = len(ticks_3m) / 3.0
    metrics["ticks_per_min_5m"] = len(ticks_5m) / 5.0 if ticks_5m else 0
    metrics["ticks_per_min_1m"] = len(ticks_1m) / 1.0 if ticks_1m else 0

    # 2. Volatility (price std dev as % of mean)
    mean_price = statistics.mean(prices_3m)
    metrics["volatility_3m"] = statistics.stdev(prices_3m) / mean_price * 100 if len(prices_3m) > 1 else 0
    metrics["volatility_5m"] = statistics.stdev(prices_5m) / statistics.mean(prices_5m) * 100 if len(prices_5m) > 1 else 0

    # 3. Price range (high-low as % of mean)
    metrics["range_pct_3m"] = (max(prices_3m) - min(prices_3m)) / mean_price * 100
    metrics["range_pct_5m"] = (max(prices_5m) - min(prices_5m)) / statistics.mean(prices_5m) * 100 if prices_5m else 0

    # 4. Volume (total quantity)
    metrics["volume_3m"] = sum(t.qty for t in ticks_3m)
    metrics["volume_5m"] = sum(t.qty for t in ticks_5m) if ticks_5m else 0

    # 5. Direction flips (how choppy is the market)
    flips_3m = 0
    for i in range(2, len(prices_3m)):
        prev_dir = prices_3m[i-1] - prices_3m[i-2]
        curr_dir = prices_3m[i] - prices_3m[i-1]
        if prev_dir * curr_dir < 0:  # direction changed
            flips_3m += 1
    metrics["flips_3m"] = flips_3m
    metrics["flips_per_tick_3m"] = flips_3m / max(1, len(prices_3m) - 2)

    # Same for 1m
    flips_1m = 0
    for i in range(2, len(prices_1m)):
        prev_dir = prices_1m[i-1] - prices_1m[i-2]
        curr_dir = prices_1m[i] - prices_1m[i-1]
        if prev_dir * curr_dir < 0:
            flips_1m += 1
    metrics["flips_1m"] = flips_1m
    metrics["flips_per_tick_1m"] = flips_1m / max(1, len(prices_1m) - 2)

    # 6. Momentum strength (net price move as % of range)
    net_move = prices_3m[-1] - prices_3m[0]
    price_range = max(prices_3m) - min(prices_3m)
    metrics["momentum_strength_3m"] = abs(net_move) / price_range * 100 if price_range > 0 else 0

    # 7. Buy/sell imbalance
    buys_3m = sum(1 for t in ticks_3m if t.is_buyer)
    metrics["buy_ratio_3m"] = buys_3m / len(ticks_3m) if ticks_3m else 0.5

    # 8. Large trade presence (trades > 2x median size)
    qtys = [t.qty for t in ticks_3m]
    median_qty = statistics.median(qtys) if qtys else 0
    if median_qty > 0:
        metrics["large_trade_ratio_3m"] = sum(1 for q in qtys if q > 2 * median_qty) / len(qtys)
    else:
        metrics["large_trade_ratio_3m"] = 0

    # 9. Recent acceleration (is volatility increasing?)
    if len(ticks_3m) > 10:
        first_half = prices_3m[:len(prices_3m)//2]
        second_half = prices_3m[len(prices_3m)//2:]
        vol_first = statistics.stdev(first_half) if len(first_half) > 1 else 0
        vol_second = statistics.stdev(second_half) if len(second_half) > 1 else 0
        metrics["vol_acceleration"] = vol_second / vol_first if vol_first > 0 else 1.0
    else:
        metrics["vol_acceleration"] = 1.0

    return metrics


def main():
    trades = load_trades()
    print(f"Loaded {len(trades)} trades")

    # Load tick data for each asset
    asset_ticks = {}
    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        ticks = load_aggtrades_multi(DATA_DIR, asset, days=7)
        if ticks:
            asset_ticks[asset] = ticks
            print(f"  {asset}: {len(ticks)} ticks loaded")

    # Analyze each trade
    win_metrics = defaultdict(list)
    loss_metrics = defaultdict(list)

    analyzed = 0
    skipped = 0

    for trade in trades:
        asset = trade["asset"]
        if asset not in asset_ticks:
            skipped += 1
            continue

        ticks = asset_ticks[asset]
        window_start = datetime.fromisoformat(trade["window_id"])

        metrics = analyze_ticks_around_trade(ticks, trade["ts"], window_start)
        if metrics is None:
            skipped += 1
            continue

        analyzed += 1
        target = win_metrics if trade["won"] else loss_metrics
        for k, v in metrics.items():
            target[k].append(v)

    print(f"\nAnalyzed {analyzed} trades ({skipped} skipped due to missing data)")
    print(f"Wins: {sum(len(v) for v in win_metrics.values()) // len(win_metrics)}")
    print(f"Losses: {sum(len(v) for v in loss_metrics.values()) // len(loss_metrics)}")

    # Compare metrics
    print(f"\n{'='*80}")
    print(f"{'Metric':<25} {'Win Mean':>10} {'Loss Mean':>10} {'Delta':>10} {'Signal?':>10}")
    print(f"{'='*80}")

    for metric in sorted(win_metrics.keys()):
        w_vals = win_metrics[metric]
        l_vals = loss_metrics[metric]
        w_mean = statistics.mean(w_vals) if w_vals else 0
        l_mean = statistics.mean(l_vals) if l_vals else 0
        delta = l_mean - w_mean
        delta_pct = delta / w_mean * 100 if w_mean != 0 else 0

        # Flag significant differences (>10% delta)
        signal = "***" if abs(delta_pct) > 20 else "**" if abs(delta_pct) > 10 else ""

        print(f"{metric:<25} {w_mean:>10.4f} {l_mean:>10.4f} {delta:>+10.4f} {signal:>10}")

    # Detailed breakdown for top signals
    print(f"\n{'='*80}")
    print("Detailed Distributions (Win vs Loss)")
    print(f"{'='*80}")

    for metric in ["volatility_3m", "flips_per_tick_3m", "momentum_strength_3m",
                    "ticks_per_min_3m", "vol_acceleration", "range_pct_3m"]:
        if metric not in win_metrics:
            continue
        w_vals = sorted(win_metrics[metric])
        l_vals = sorted(loss_metrics[metric])

        print(f"\n  {metric}:")
        print(f"    Win  - p25={w_vals[len(w_vals)//4]:.4f}  p50={w_vals[len(w_vals)//2]:.4f}  p75={w_vals[3*len(w_vals)//4]:.4f}")
        print(f"    Loss - p25={l_vals[len(l_vals)//4]:.4f}  p50={l_vals[len(l_vals)//2]:.4f}  p75={l_vals[3*len(l_vals)//4]:.4f}")

    # Check if any metric can separate wins from losses
    print(f"\n{'='*80}")
    print("Potential Filters (thresholds that improve WR)")
    print(f"{'='*80}")

    for metric in ["volatility_3m", "flips_per_tick_3m", "momentum_strength_3m",
                    "ticks_per_min_3m", "vol_acceleration", "range_pct_3m",
                    "flips_per_tick_1m", "large_trade_ratio_3m"]:
        if metric not in win_metrics:
            continue

        all_vals = [(v, True) for v in win_metrics[metric]] + [(v, False) for v in loss_metrics[metric]]
        all_vals.sort(key=lambda x: x[0])

        # Try different percentile cutoffs
        print(f"\n  {metric}:")
        for pct in [25, 50, 75]:
            idx = len(all_vals) * pct // 100
            threshold = all_vals[idx][0]

            # Filter: only trade when metric <= threshold
            below_w = sum(1 for v, won in all_vals if v <= threshold and won)
            below_l = sum(1 for v, won in all_vals if v <= threshold and not won)
            below_t = below_w + below_l
            below_wr = below_w / below_t * 100 if below_t else 0

            # Filter: only trade when metric > threshold
            above_w = sum(1 for v, won in all_vals if v > threshold and won)
            above_l = sum(1 for v, won in all_vals if v > threshold and not won)
            above_t = above_w + above_l
            above_wr = above_w / above_t * 100 if above_t else 0

            print(f"    thresh={threshold:.4f} (p{pct}):  below={below_wr:.0f}% ({below_t}t)  above={above_wr:.0f}% ({above_t}t)")


if __name__ == "__main__":
    main()
