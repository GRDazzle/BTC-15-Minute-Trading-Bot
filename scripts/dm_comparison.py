"""Compare dm 2 vs dm 3 entry gate on backtest data."""
import sys
from pathlib import Path
from collections import deque
from datetime import date

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

PARAMS = {
    "BTC": {"ml_w": 0.30, "thresh": 0.70, "max_p": 80, "min_p": 60, "contracts": 20},
    "ETH": {"ml_w": 0.20, "thresh": 0.65, "max_p": 75, "min_p": 60, "contracts": 10},
    "SOL": {"ml_w": 0.20, "thresh": 0.65, "max_p": 80, "min_p": 60, "contracts": 10},
    "XRP": {"ml_w": 0.20, "thresh": 0.70, "max_p": 80, "min_p": 60, "contracts": 10},
}

results = {}

for min_dm_test in [2, 3]:
    total_pnl = 0
    total_w = 0
    total_l = 0
    asset_results = {}

    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        p = PARAMS[asset]
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

        ml = MLProcessor(asset=asset, model_dir=MODEL_DIR, confidence_threshold=0.60, model_suffix="_weekday")
        lstm_path = MODEL_DIR / f"{asset}_weekday_lstm.pt"
        lstm_model, lstm_meta = load_lstm(str(lstm_path))
        scaler_mean = np.array(lstm_meta.get("scaler_mean", []), dtype=np.float32)
        scaler_std = np.array(lstm_meta.get("scaler_std", []), dtype=np.float32)
        scaler_std[scaler_std == 0] = 1.0

        procs = [
            SpikeDetectionProcessor(spike_threshold=0.003, velocity_threshold=0.0015, lookback_periods=20, min_confidence=0.55),
            TickVelocityProcessor(velocity_threshold_60s=0.001, velocity_threshold_30s=0.0007, min_ticks=5, min_confidence=0.55),
        ]
        sim = BacktestSimulator(procs, SignalFusionEngine(), ml_processor=ml, min_dm=min_dm_test)
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
                if dm < min_dm_test or dm > 8:
                    continue
                cp["kalshi"] = get_kalshi_prices(kw, signal_ts) if signal_ts else None

                if orig_win and signal_ts:
                    while tick_feed_idx < len(orig_win.ticks_during) and orig_win.ticks_during[tick_feed_idx].ts < signal_ts:
                        t = orig_win.ticks_during[tick_feed_idx]
                        tick_buffer.append({"ts": t.ts, "price": t.price, "qty": t.qty, "is_buyer": t.is_buyer})
                        tick_feed_idx += 1
                    seq = extract_lstm_sequence(list(tick_buffer), signal_ts, decision_minute=dm, window_open_price=orig_win.price_open)
                    if seq is not None:
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
                break  # one trade per window

        asset_results[asset] = {"pnl": a_pnl, "w": a_w, "l": a_l}
        total_pnl += a_pnl
        total_w += a_w
        total_l += a_l

    results[min_dm_test] = {"total_pnl": total_pnl, "w": total_w, "l": total_l, "assets": asset_results}

print("=== Backtest: dm 2 vs dm 3 ===")
print()
for dm in [2, 3]:
    r = results[dm]
    t = r["w"] + r["l"]
    wr = r["w"] / t * 100 if t else 0
    print(f"min_dm={dm}: {r['w']}W/{r['l']}L = {wr:.1f}% WR  PnL=${r['total_pnl']:+.2f}  trades={t}")
    for a in ["BTC", "ETH", "SOL", "XRP"]:
        ar = r["assets"][a]
        at = ar["w"] + ar["l"]
        awr = ar["w"] / at * 100 if at else 0
        print(f"  {a}: {ar['w']}W/{ar['l']}L = {awr:.1f}% WR  PnL=${ar['pnl']:+.2f}")
    print()

d = results[3]["total_pnl"] - results[2]["total_pnl"]
print(f"Difference: ${d:+.2f} ({'better' if d > 0 else 'worse'} with dm 3)")
