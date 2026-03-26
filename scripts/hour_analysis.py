"""Hour-by-hour PnL analysis using real Kalshi prices."""
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import date
from loguru import logger
logger.remove()

from backtester.data_loader_ticks import load_aggtrades_multi, generate_tick_windows
from backtester.data_loader_kalshi import load_kalshi_windows, get_kalshi_prices
from backtester.data_loader import load_fear_greed
from backtester.simulator import BacktestSimulator
from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.signal_processors.ml_processor import MLProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
FG_CSV = PROJECT_ROOT / "data" / "historical" / "fear_greed.csv"
FEE = 2

# Current params from PnL sweep
PARAMS = {
    "BTC": (0.35, 0.58, 65, 50, 20),
    "ETH": (0.15, 0.55, 75, 25, 10),
    "SOL": (0.20, 0.58, 85, 25, 10),
    "XRP": (0.20, 0.58, 90, 25, 10),
}

all_hour_data = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "avg_entry": []}))

for asset in ["BTC", "ETH", "SOL", "XRP"]:
    ml_w, thresh, max_price, init_bal, max_contracts = PARAMS[asset]
    fusion_w = 1.0 - ml_w

    kalshi_windows = load_kalshi_windows(KALSHI_DIR, asset)
    if not kalshi_windows:
        continue

    kalshi_close_times = set(kalshi_windows.keys())
    kalshi_dates = sorted(set(ct.date() for ct in kalshi_windows.keys()))
    days_back = (date.today() - kalshi_dates[0]).days + 2

    ticks = load_aggtrades_multi(DATA_DIR, asset, days=days_back)
    if not ticks:
        continue
    windows = generate_tick_windows(ticks, min_warmup_ticks=1, min_during_ticks=1)
    windows = [w for w in windows if w.window_end in kalshi_close_times]

    fg = load_fear_greed(FG_CSV) if FG_CSV.exists() else {}
    ml = MLProcessor(asset=asset, model_dir=PROJECT_ROOT / "models", confidence_threshold=0.60)
    procs = [
        SpikeDetectionProcessor(spike_threshold=0.003, velocity_threshold=0.0015, lookback_periods=20, min_confidence=0.55),
        TickVelocityProcessor(velocity_threshold_60s=0.001, velocity_threshold_30s=0.0007, min_ticks=5, min_confidence=0.55),
    ]
    sim = BacktestSimulator(procs, SignalFusionEngine(), ml_processor=ml, min_dm=2)
    window_data = sim.run_ticks_collect_probabilities(windows, fg)

    balance = init_bal
    print(f"Processing {asset}: {len(window_data)} windows...")

    for win in window_data:
        we = win.get("window_end")
        if we is None:
            continue
        kw = kalshi_windows.get(we)
        if kw is None or kw.outcome is None:
            continue

        hour = we.hour

        predicted = None
        confidence = 0.0
        entry_price = 0
        side = ""

        for cp in win["checkpoints"]:
            if cp.get("dm", 0) > 8:
                continue
            ep = ml_w * cp["ml_p"] + fusion_w * cp["fusion_p"]
            signal_ts = cp.get("signal_ts")
            if signal_ts is None:
                continue
            kp = get_kalshi_prices(kw, signal_ts)
            if kp is None:
                continue

            if ep >= thresh:
                p = kp["yes_ask"]
                if p <= 0 or p >= 100 or p < 15 or p > max_price:
                    continue
                predicted = "BULLISH"
                confidence = ep
                side = "yes"
                entry_price = p
                break
            elif ep <= 1.0 - thresh:
                p = kp["no_ask"]
                if p <= 0 or p >= 100 or p < 15 or p > max_price:
                    continue
                predicted = "BEARISH"
                confidence = 1.0 - ep
                side = "no"
                entry_price = p
                break

        if predicted is None:
            continue

        cost_per = (entry_price + FEE) / 100.0
        max_by_bal = int(balance / cost_per) if cost_per > 0 else 0
        scale = min(1.0, confidence * confidence)
        contracts = min(max(1, int(max_by_bal * scale)), max_contracts)

        cost = (entry_price / 100.0) * contracts
        fees = (FEE / 100.0) * contracts
        balance -= (cost + fees)

        won = side == kw.outcome
        revenue = contracts * 1.0 if won else 0.0
        pnl = revenue - cost - fees
        balance += revenue

        hd = all_hour_data[hour][asset]
        if won:
            hd["wins"] += 1
        else:
            hd["losses"] += 1
        hd["pnl"] += pnl
        hd["avg_entry"].append(entry_price)

# Print results
print()
print("=" * 110)
print("  HOUR-BY-HOUR PnL ANALYSIS (current params, all Kalshi data)")
print("=" * 110)

header = f"{'Hour':>5} | {'BTC WR%':>7} {'PnL':>7} {'#':>3} | {'ETH WR%':>7} {'PnL':>7} {'#':>3} | {'SOL WR%':>7} {'PnL':>7} {'#':>3} | {'XRP WR%':>7} {'PnL':>7} {'#':>3} | {'ALL WR%':>7} {'PnL':>8} {'#':>4}"
print(header)
print("-" * 110)

hour_totals = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})

for h in range(24):
    parts = []
    for asset_name in ["BTC", "ETH", "SOL", "XRP"]:
        hd = all_hour_data[h][asset_name]
        total = hd["wins"] + hd["losses"]
        if total > 0:
            wr = hd["wins"] / total * 100
            parts.append(f"{wr:5.0f}% {hd['pnl']:+7.1f} {total:3d}")
            hour_totals[h]["wins"] += hd["wins"]
            hour_totals[h]["losses"] += hd["losses"]
            hour_totals[h]["pnl"] += hd["pnl"]
        else:
            parts.append(f"{'--':>19}")

    ht = hour_totals[h]
    total_all = ht["wins"] + ht["losses"]
    if total_all > 0:
        wr_all = ht["wins"] / total_all * 100
        all_str = f"{wr_all:5.0f}% {ht['pnl']:+8.1f} {total_all:4d}"
    else:
        all_str = "--"

    print(f"{h:02d}:00 | {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} | {all_str}")

# Summary
print()
print("=== RANKED BY COMBINED PnL ===")
sorted_hours = sorted(hour_totals.items(), key=lambda x: x[1]["pnl"], reverse=True)
for h, ht in sorted_hours:
    total = ht["wins"] + ht["losses"]
    wr = ht["wins"] / total * 100 if total else 0
    marker = " *** SKIP" if ht["pnl"] < 0 else ""
    print(f"  {h:02d}:00 UTC  PnL=${ht['pnl']:+8.2f}  WR={wr:5.1f}% ({total:3d} trades){marker}")

# Impact of skipping negative-PnL hours
skip_hours = set(h for h, ht in sorted_hours if ht["pnl"] < 0)
print(f"\n=== SKIP NEGATIVE-PNL HOURS: {sorted(skip_hours)} ===")
kept_w = sum(ht["wins"] for h, ht in hour_totals.items() if h not in skip_hours)
kept_l = sum(ht["losses"] for h, ht in hour_totals.items() if h not in skip_hours)
kept_pnl = sum(ht["pnl"] for h, ht in hour_totals.items() if h not in skip_hours)
all_w = sum(ht["wins"] for ht in hour_totals.values())
all_l = sum(ht["losses"] for ht in hour_totals.values())
all_pnl = sum(ht["pnl"] for ht in hour_totals.values())
print(f"  All hours:       PnL=${all_pnl:+.2f}  WR={all_w/(all_w+all_l)*100:.1f}% ({all_w+all_l} trades)")
print(f"  Skip negative:   PnL=${kept_pnl:+.2f}  WR={kept_w/(kept_w+kept_l)*100:.1f}% ({kept_w+kept_l} trades)")
print(f"  Improvement:     PnL=${kept_pnl-all_pnl:+.2f}  ({all_w+all_l - kept_w - kept_l} trades removed)")

# Also test skip worst 4 hours
skip4 = set(h for h, _ in sorted_hours[-4:])
print(f"\n=== SKIP WORST 4 HOURS: {sorted(skip4)} ===")
k4_w = sum(ht["wins"] for h, ht in hour_totals.items() if h not in skip4)
k4_l = sum(ht["losses"] for h, ht in hour_totals.items() if h not in skip4)
k4_pnl = sum(ht["pnl"] for h, ht in hour_totals.items() if h not in skip4)
print(f"  All hours:       PnL=${all_pnl:+.2f}  WR={all_w/(all_w+all_l)*100:.1f}% ({all_w+all_l} trades)")
print(f"  Skip worst 4:    PnL=${k4_pnl:+.2f}  WR={k4_w/(k4_w+k4_l)*100:.1f}% ({k4_w+k4_l} trades)")
print(f"  Improvement:     PnL=${k4_pnl-all_pnl:+.2f}  ({all_w+all_l - k4_w - k4_l} trades removed)")
