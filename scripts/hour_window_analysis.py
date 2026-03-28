"""
Hour-by-hour PnL analysis using current 3-way ensemble config.

Tests which trading hours are profitable and which should be avoided.
Uses current config params (ml_weight, threshold, max_price, min_price).
"""
import json
import sys
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from ml.lstm_model import load_model as load_lstm
from ml.lstm_features import extract_lstm_sequence

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
FG_CSV = PROJECT_ROOT / "data" / "historical" / "fear_greed.csv"
MODEL_DIR = PROJECT_ROOT / "models"

FEE = 2
K = 4.5


def main():
    with open(PROJECT_ROOT / "config" / "trading.json") as f:
        cfg = json.load(f)

    all_trades = []

    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        ac = cfg["assets"].get(asset, {})
        ens = ac.get("ensemble", {})
        ml_w = ens.get("ml_weight", 0.2)
        thresh = ens.get("threshold", 0.7)
        max_price = ens.get("max_price_cents", 90)
        min_price = ac.get("min_price_cents", 55)

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
        ml = MLProcessor(asset=asset, model_dir=MODEL_DIR, confidence_threshold=0.60)
        procs = [
            SpikeDetectionProcessor(spike_threshold=0.003, velocity_threshold=0.0015,
                                    lookback_periods=20, min_confidence=0.55),
            TickVelocityProcessor(velocity_threshold_60s=0.001, velocity_threshold_30s=0.0007,
                                  min_ticks=5, min_confidence=0.55),
        ]
        sim = BacktestSimulator(procs, SignalFusionEngine(), ml_processor=ml, min_dm=2)
        window_data = sim.run_ticks_collect_probabilities(windows, fg)

        # Load LSTM
        lstm_path = MODEL_DIR / f"{asset}_lstm.pt"
        lstm_model, lstm_meta = None, {}
        if lstm_path.exists():
            lstm_model, lstm_meta = load_lstm(str(lstm_path))
        scaler_mean = np.array(lstm_meta.get("scaler_mean", []), dtype=np.float32)
        scaler_std = np.array(lstm_meta.get("scaler_std", []), dtype=np.float32)
        if len(scaler_std) > 0:
            scaler_std[scaler_std == 0] = 1.0

        print(f"Processing {asset}: {len(window_data)} windows...")

        for win in window_data:
            we = win.get("window_end")
            if we is None:
                continue
            kw = kalshi_windows.get(we)
            if kw is None or kw.outcome is None:
                continue

            hour = we.hour

            orig_win = None
            for w in windows:
                if w.window_end == we:
                    orig_win = w
                    break

            tick_buffer = deque(maxlen=5000)
            tfi = 0
            if orig_win:
                for t in orig_win.ticks_before:
                    tick_buffer.append({"ts": t.ts, "price": t.price,
                                        "qty": t.qty, "is_buyer": t.is_buyer})

            predicted = None
            entry_price = 0
            side = ""

            for cp in win["checkpoints"]:
                if cp.get("dm", 0) > 8:
                    continue
                signal_ts = cp.get("signal_ts")
                if signal_ts is None:
                    continue
                kp = get_kalshi_prices(kw, signal_ts)
                if kp is None:
                    continue

                xgb_p = cp["ml_p"]
                fusion_p = cp["fusion_p"]

                # Feed ticks and get LSTM
                lstm_p = 0.5
                if lstm_model is not None and orig_win and signal_ts:
                    while tfi < len(orig_win.ticks_during) and orig_win.ticks_during[tfi].ts < signal_ts:
                        t = orig_win.ticks_during[tfi]
                        tick_buffer.append({"ts": t.ts, "price": t.price,
                                            "qty": t.qty, "is_buyer": t.is_buyer})
                        tfi += 1
                    seq = extract_lstm_sequence(
                        list(tick_buffer), signal_ts,
                        decision_minute=cp.get("dm", 0),
                        window_open_price=orig_win.price_open,
                    )
                    if seq is not None and len(scaler_mean) > 0:
                        seq_n = (seq - scaler_mean) / scaler_std
                        seq_n = np.nan_to_num(seq_n, nan=0.0)
                        with torch.no_grad():
                            lstm_p = lstm_model(
                                torch.tensor(seq_n, dtype=torch.float32).unsqueeze(0)
                            ).item()

                # Dynamic 3-way weight
                xgb_conf = abs(xgb_p - 0.5) * 2.0
                lstm_conf = abs(lstm_p - 0.5) * 2.0
                dyn_xgb = ml_w + (0.60 - ml_w) * (xgb_conf ** K)
                dyn_lstm = 0.10 + 0.30 * (lstm_conf ** K)
                dyn_fus = max(0.0, 1.0 - dyn_xgb - dyn_lstm)
                ep = dyn_xgb * xgb_p + dyn_lstm * lstm_p + dyn_fus * fusion_p

                if ep >= thresh:
                    p = kp["yes_ask"]
                    if 0 < p < 100 and min_price <= p <= max_price:
                        predicted = "BULLISH"
                        side = "yes"
                        entry_price = p
                        break
                elif ep <= 1.0 - thresh:
                    p = kp["no_ask"]
                    if 0 < p < 100 and min_price <= p <= max_price:
                        predicted = "BEARISH"
                        side = "no"
                        entry_price = p
                        break

            if predicted is None:
                continue

            cost = entry_price / 100.0
            fees = FEE / 100.0
            won = side == kw.outcome
            pnl = (1.0 - cost - fees) if won else -(cost + fees)

            all_trades.append({
                "asset": asset, "hour": hour, "won": won,
                "pnl": pnl, "entry_price": entry_price,
            })

    # Print results
    print("\n" + "=" * 95)
    print(f"  HOUR-BY-HOUR PnL ANALYSIS (3-way dynamic ensemble, current config)")
    print("=" * 95)
    print(f"{'Hour':>4} | {'ALL':>25} | {'BTC':>18} | {'ETH':>18} | {'SOL':>18} | {'XRP':>18}")
    print("-" * 95)

    hour_pnl = defaultdict(float)
    hour_trades = defaultdict(int)
    hour_wins = defaultdict(int)

    for h in range(24):
        ht = [t for t in all_trades if t["hour"] == h]
        if not ht:
            print(f"{h:02d}:00 | {'--':>25} |")
            continue

        for t in ht:
            hour_pnl[h] += t["pnl"]
            hour_trades[h] += 1
            if t["won"]:
                hour_wins[h] += 1

        parts = []
        for subset in [["BTC", "ETH", "SOL", "XRP"], ["BTC"], ["ETH"], ["SOL"], ["XRP"]]:
            trades = [t for t in ht if t["asset"] in subset]
            if not trades:
                parts.append("--")
                continue
            w = sum(1 for t in trades if t["won"])
            p = sum(t["pnl"] for t in trades)
            wr = w / len(trades) * 100
            parts.append(f"{wr:.0f}% ${p:+.0f} ({len(trades)})")

        print(f"{h:02d}:00 | {parts[0]:>25} | {parts[1]:>18} | {parts[2]:>18} | {parts[3]:>18} | {parts[4]:>18}")

    print()
    print("=== BEST HOURS ===")
    for h in sorted(hour_pnl, key=lambda x: hour_pnl[x], reverse=True)[:8]:
        wr = hour_wins[h] / hour_trades[h] * 100 if hour_trades[h] else 0
        print(f"  {h:02d}:00 UTC  PnL=${hour_pnl[h]:+.2f}  WR={wr:.0f}% ({hour_trades[h]} trades)")

    print()
    print("=== WORST HOURS ===")
    for h in sorted(hour_pnl, key=lambda x: hour_pnl[x])[:8]:
        wr = hour_wins[h] / hour_trades[h] * 100 if hour_trades[h] else 0
        print(f"  {h:02d}:00 UTC  PnL=${hour_pnl[h]:+.2f}  WR={wr:.0f}% ({hour_trades[h]} trades)")

    total_pnl = sum(t["pnl"] for t in all_trades)
    total_w = sum(1 for t in all_trades if t["won"])
    print(f"\nTotal: {total_w}W/{len(all_trades)-total_w}L = {total_w/len(all_trades)*100:.1f}% WR  PnL=${total_pnl:+.2f}")


if __name__ == "__main__":
    main()
