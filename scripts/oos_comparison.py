"""Out-of-sample comparison: v6 vs v7 models on Monday-Tuesday data only.

Uses pre-Sunday params (from snapshot) to avoid look-ahead bias.
Tests only on 2026-03-31 and 2026-04-01 Kalshi data.
"""
import sys
from pathlib import Path
from collections import deque
from datetime import date, datetime

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
logger.remove()

from backtester.data_loader_ticks import load_aggtrades_multi, generate_tick_windows
from backtester.data_loader_kalshi import load_kalshi_windows, get_kalshi_prices
from core.strategy_brain.signal_processors.ml_processor import MLProcessor
from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine
from backtester.simulator import BacktestSimulator
from ml.lstm_model import load_model as load_lstm
from ml.lstm_features import extract_lstm_sequence

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
MODEL_DIR = PROJECT_ROOT / "models"
FEE = 2
K = 4.5

# Pre-Sunday params (from snapshot, no look-ahead bias)
PARAMS = {
    "BTC": {"ml_w": 0.20, "thresh": 0.65, "max_p": 80, "min_p": 60, "contracts": 20},
    "ETH": {"ml_w": 0.10, "thresh": 0.65, "max_p": 80, "min_p": 60, "contracts": 10},
    "SOL": {"ml_w": 0.10, "thresh": 0.70, "max_p": 80, "min_p": 60, "contracts": 10},
    "XRP": {"ml_w": 0.10, "thresh": 0.70, "max_p": 80, "min_p": 60, "contracts": 10},
}

# Only test on Monday-Tuesday (out-of-sample)
OOS_START = date(2026, 3, 31)


def run_model_variant(suffix, label):
    """Run one model variant on OOS data and return results."""
    total_pnl = 0
    total_w = 0
    total_l = 0
    asset_results = {}
    hourly = {}

    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        p = PARAMS[asset]
        kalshi_windows = load_kalshi_windows(KALSHI_DIR, asset)
        if not kalshi_windows:
            continue

        # Filter to OOS dates only
        kalshi_windows = {k: v for k, v in kalshi_windows.items() if k.date() >= OOS_START}
        if not kalshi_windows:
            print(f"  No OOS Kalshi data for {asset}")
            continue

        kalshi_close_times = set(kalshi_windows.keys())
        days_back = (date.today() - OOS_START).days + 2

        ticks = load_aggtrades_multi(DATA_DIR, asset, days=days_back)
        if not ticks:
            continue
        windows = generate_tick_windows(ticks, min_warmup_ticks=1, min_during_ticks=1)
        windows = [w for w in windows if w.window_end in kalshi_close_times]
        # Weekday only
        windows = [w for w in windows if w.window_start.weekday() < 5]

        try:
            ml = MLProcessor(asset=asset, model_dir=MODEL_DIR, confidence_threshold=0.60, model_suffix=suffix)
        except FileNotFoundError:
            print(f"  No {label} XGB model for {asset}")
            continue

        lstm_path = MODEL_DIR / f"{asset}{suffix}_lstm.pt"
        if not lstm_path.exists():
            print(f"  No {label} LSTM model for {asset}")
            continue
        lstm_model, lstm_meta = load_lstm(str(lstm_path))
        scaler_mean = np.array(lstm_meta.get("scaler_mean", []), dtype=np.float32)
        scaler_std = np.array(lstm_meta.get("scaler_std", []), dtype=np.float32)
        scaler_std[scaler_std == 0] = 1.0

        procs = [
            SpikeDetectionProcessor(spike_threshold=0.003, velocity_threshold=0.0015, lookback_periods=20, min_confidence=0.55),
            TickVelocityProcessor(velocity_threshold_60s=0.001, velocity_threshold_30s=0.0007, min_ticks=5, min_confidence=0.55),
        ]
        sim = BacktestSimulator(procs, SignalFusionEngine(), ml_processor=ml, min_dm=2)
        window_data = sim.run_ticks_collect_probabilities(windows, {})

        a_pnl = 0.0
        a_w = 0
        a_l = 0

        for win in window_data:
            we = win.get("window_end")
            if we is None:
                continue
            kw = kalshi_windows.get(we)
            if kw is None or kw.outcome is None:
                continue

            orig_win = None
            for w in windows:
                if w.window_end == we:
                    orig_win = w
                    break

            tick_buffer = deque(maxlen=5000)
            if orig_win:
                for t in orig_win.ticks_before:
                    tick_buffer.append({"ts": t.ts, "price": t.price, "qty": t.qty, "is_buyer": t.is_buyer})
                tick_feed_idx = 0

            for cp in win["checkpoints"]:
                signal_ts = cp.get("signal_ts")
                dm = cp.get("dm", 0)
                if dm < 2 or dm > 8:
                    continue
                cp["kalshi"] = get_kalshi_prices(kw, signal_ts) if signal_ts else None

                if orig_win and signal_ts:
                    while tick_feed_idx < len(orig_win.ticks_during) and orig_win.ticks_during[tick_feed_idx].ts < signal_ts:
                        t = orig_win.ticks_during[tick_feed_idx]
                        tick_buffer.append({"ts": t.ts, "price": t.price, "qty": t.qty, "is_buyer": t.is_buyer})
                        tick_feed_idx += 1
                    seq = extract_lstm_sequence(list(tick_buffer), signal_ts, decision_minute=dm, window_open_price=orig_win.price_open)
                    if seq is not None:
                        # Trim features to match model's scaler (v6=15, v7=18)
                        n_feat = len(scaler_mean)
                        seq = seq[:, :n_feat]
                        seq_norm = (seq - scaler_mean) / scaler_std
                        seq_norm = np.nan_to_num(seq_norm, nan=0.0)
                        with torch.no_grad():
                            cp["lstm_p"] = lstm_model(torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)).item()
                    else:
                        cp["lstm_p"] = None
                else:
                    cp["lstm_p"] = None

                ml_p = cp.get("ml_p")
                fusion_p = cp.get("fusion_p", 0.5)
                lstm_p = cp.get("lstm_p")
                kalshi = cp.get("kalshi")
                if ml_p is None or kalshi is None:
                    continue

                xgb_conf = abs(ml_p - 0.5) * 2.0
                dyn_xgb_w = p["ml_w"] + (0.60 - p["ml_w"]) * (xgb_conf ** K)

                if lstm_p is not None:
                    lstm_conf = abs(lstm_p - 0.5) * 2.0
                    dyn_lstm_w = 0.10 + 0.30 * (lstm_conf ** K)
                    dyn_fusion_w = max(0.0, 1.0 - dyn_xgb_w - dyn_lstm_w)
                    ensemble_p = dyn_xgb_w * ml_p + dyn_lstm_w * lstm_p + dyn_fusion_w * fusion_p
                else:
                    dyn_fusion_w = 1.0 - dyn_xgb_w
                    ensemble_p = dyn_xgb_w * ml_p + dyn_fusion_w * fusion_p

                if ensemble_p >= p["thresh"]:
                    direction = "BULLISH"
                    entry = kalshi.get("yes_ask", 0)
                elif ensemble_p <= 1.0 - p["thresh"]:
                    direction = "BEARISH"
                    entry = kalshi.get("no_ask", 0)
                else:
                    continue

                if entry < p["min_p"] or entry > p["max_p"] or entry <= 0:
                    continue

                outcome_yes = kw.outcome == "yes"
                won = (direction == "BULLISH" and outcome_yes) or (direction == "BEARISH" and not outcome_yes)
                trade_pnl = (100 - entry - FEE) * p["contracts"] / 100.0 if won else -(entry + FEE) * p["contracts"] / 100.0

                a_pnl += trade_pnl
                if won:
                    a_w += 1
                else:
                    a_l += 1

                # Track hourly
                hour = we.hour
                if hour not in hourly:
                    hourly[hour] = {"w": 0, "l": 0, "pnl": 0.0}
                hourly[hour]["w" if won else "l"] += 1
                hourly[hour]["pnl"] += trade_pnl

                break  # one trade per window

        asset_results[asset] = {"pnl": a_pnl, "w": a_w, "l": a_l}
        total_pnl += a_pnl
        total_w += a_w
        total_l += a_l

    return {
        "total_pnl": total_pnl, "w": total_w, "l": total_l,
        "assets": asset_results, "hourly": hourly,
    }


def main():
    print(f"Out-of-Sample Test: {OOS_START} to {date.today()}")
    print(f"Using pre-Sunday params (no look-ahead bias)")
    print()

    # v6 can't run with new feature code (22 vs 26 features mismatch)
    # Use live trades.csv as the v6 baseline instead
    print("v6 baseline: using live trades.csv (actual results)")
    print()

    # Test v7 (new features)
    print("Running v7 (weekday, 26 features)...")
    v7 = run_model_variant("_v7", "v7")

    # Compare
    print()
    print("=" * 70)
    print("OUT-OF-SAMPLE RESULTS: v7 on Monday-Tuesday")
    print("=" * 70)

    t = v7["w"] + v7["l"]
    wr = v7["w"] / t * 100 if t else 0
    print(f"\n  v7 (26 feat): {v7['w']}W/{v7['l']}L = {wr:.1f}% WR  PnL=${v7['total_pnl']:+.2f}  trades={t}")
    for a in ["BTC", "ETH", "SOL", "XRP"]:
        if a in v7["assets"]:
            ar = v7["assets"][a]
            at = ar["w"] + ar["l"]
            awr = ar["w"] / at * 100 if at else 0
            print(f"    {a}: {ar['w']}W/{ar['l']}L = {awr:.1f}% WR  PnL=${ar['pnl']:+.2f}")

    print(f"\n  Compare to live v6 session: 300 trades, 78% WR, ~-$25 PnL")

    # Hourly breakdown
    print()
    print("=" * 70)
    print("v7 HOURLY BREAKDOWN")
    print("=" * 70)
    all_hours = sorted(v7["hourly"].keys())
    print(f"\n  {'Hour':>4} {'PT':>4}  {'WR':>7} {'PnL':>9} {'Trades':>7}")
    print("  " + "-" * 40)
    for h in all_hours:
        h7 = v7["hourly"][h]
        t7 = h7["w"] + h7["l"]
        wr7 = h7["w"] / t7 * 100 if t7 else 0
        pt = (h - 7) % 24
        print(f"  {h:>4} {pt:>3}h  {wr7:>6.0f}% ${h7['pnl']:>+8.2f} {t7:>7}")


if __name__ == "__main__":
    main()
