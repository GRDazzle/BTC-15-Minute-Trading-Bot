"""
Sweep 3 ensemble combinations with dynamic weighting against Kalshi PnL.

1. XGB + LSTM + Fusion (3-way)
2. LSTM + Fusion (2-way, no XGB)
3. LSTM + XGB (2-way, no Fusion)

All use dynamic weight scaling: w = min_w + (max_w - min_w) * confidence^k
"""
import argparse
import csv
import json
import sys
import time
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
from ml.lstm_features import extract_lstm_sequence, LSTM_SEQ_LEN

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
FG_CSV = PROJECT_ROOT / "data" / "historical" / "fear_greed.csv"
MODEL_DIR = PROJECT_ROOT / "models"

FEE_CENTS = 2
K = 4.5  # Exponential scaling constant


def sweep_combo(
    mode: str,  # "3way", "lstm_fusion", "lstm_xgb"
    min_w_a: float, max_w_a: float,
    min_w_b: float, max_w_b: float,
    threshold: float, max_price: int,
    enriched_windows: list[dict],
    config: dict,
    max_dm: int | None = None,
) -> dict:
    """Evaluate one dynamic weight combo.

    For 3way: A=xgb, B=lstm, remainder=fusion
    For lstm_fusion: A=lstm, B=0, remainder=fusion
    For lstm_xgb: A=lstm, B=xgb, remainder=0
    """
    balance = config["initial_balance"]
    max_contracts = config["max_contracts_per_trade"]
    min_price_cents = config["min_price_cents"]

    traded_count = 0
    win_count = 0
    total_pnl = 0.0

    for win in enriched_windows:
        outcome = win["outcome"]
        if not outcome:
            continue

        predicted = None
        entry_price = 0
        side = ""

        for cp in win["checkpoints"]:
            if cp.get("kalshi") is None:
                continue
            if max_dm is not None and cp["dm"] > max_dm:
                continue

            xgb_p = cp["ml_p"]
            fusion_p = cp["fusion_p"]
            lstm_p = cp.get("lstm_p")
            if lstm_p is None:
                continue

            # Compute dynamic weights based on mode
            if mode == "3way":
                xgb_conf = abs(xgb_p - 0.5) * 2.0
                lstm_conf = abs(lstm_p - 0.5) * 2.0
                w_xgb = min_w_a + (max_w_a - min_w_a) * (xgb_conf ** K)
                w_lstm = min_w_b + (max_w_b - min_w_b) * (lstm_conf ** K)
                w_fusion = max(0.0, 1.0 - w_xgb - w_lstm)
                ensemble_p = w_xgb * xgb_p + w_lstm * lstm_p + w_fusion * fusion_p

            elif mode == "lstm_fusion":
                lstm_conf = abs(lstm_p - 0.5) * 2.0
                w_lstm = min_w_a + (max_w_a - min_w_a) * (lstm_conf ** K)
                w_fusion = 1.0 - w_lstm
                ensemble_p = w_lstm * lstm_p + w_fusion * fusion_p

            elif mode == "lstm_xgb":
                lstm_conf = abs(lstm_p - 0.5) * 2.0
                xgb_conf = abs(xgb_p - 0.5) * 2.0
                w_lstm = min_w_a + (max_w_a - min_w_a) * (lstm_conf ** K)
                w_xgb = min_w_b + (max_w_b - min_w_b) * (xgb_conf ** K)
                total_w = w_lstm + w_xgb
                if total_w > 0:
                    w_lstm /= total_w
                    w_xgb /= total_w
                ensemble_p = w_lstm * lstm_p + w_xgb * xgb_p

            if ensemble_p >= threshold:
                p = cp["kalshi"]["yes_ask"]
                if p <= 0 or p >= 100 or p < min_price_cents or p > max_price:
                    continue
                predicted = "BULLISH"
                side = "yes"
                entry_price = p
                break
            elif ensemble_p <= 1.0 - threshold:
                p = cp["kalshi"]["no_ask"]
                if p <= 0 or p >= 100 or p < min_price_cents or p > max_price:
                    continue
                predicted = "BEARISH"
                side = "no"
                entry_price = p
                break

        if predicted is None:
            continue

        cost_per = (entry_price + FEE_CENTS) / 100.0
        max_by_bal = int(balance / cost_per) if cost_per > 0 else 0
        contracts = min(max(1, max_by_bal), max_contracts)

        cost = (entry_price / 100.0) * contracts
        fees = (FEE_CENTS / 100.0) * contracts

        won = (side == outcome)
        revenue = contracts * 1.0 if won else 0.0
        pnl = revenue - cost - fees
        total_pnl += pnl
        traded_count += 1
        if won:
            win_count += 1

    wr = win_count / traded_count * 100 if traded_count > 0 else 0
    return {
        "mode": mode,
        "min_w_a": min_w_a, "max_w_a": max_w_a,
        "min_w_b": min_w_b, "max_w_b": max_w_b,
        "threshold": threshold, "max_price": max_price,
        "total_pnl": total_pnl, "win_rate": wr,
        "traded_count": traded_count, "win_count": win_count,
    }


def run_asset(asset: str, min_dm: int = 2, max_dm: int | None = 8, model_suffix: str = "", day_filter: str = "all"):
    """Run all 3 ensemble combinations for one asset."""
    # Load Kalshi data
    kalshi_windows = load_kalshi_windows(KALSHI_DIR, asset)
    if not kalshi_windows:
        print(f"  No Kalshi data for {asset}")
        return
    kalshi_close_times = set(kalshi_windows.keys())
    kalshi_dates = sorted(set(ct.date() for ct in kalshi_windows.keys()))
    days_back = (date.today() - kalshi_dates[0]).days + 2

    # Load ticks
    ticks = load_aggtrades_multi(DATA_DIR, asset, days=days_back)
    if not ticks:
        return

    windows = generate_tick_windows(ticks, min_warmup_ticks=1, min_during_ticks=1)
    windows = [w for w in windows if w.window_end in kalshi_close_times]

    # Filter by day of week
    if day_filter == "weekday":
        before = len(windows)
        windows = [w for w in windows if w.window_start.weekday() < 5]
        print(f"  Day filter weekday: {before} -> {len(windows)} windows")
    elif day_filter == "weekend":
        before = len(windows)
        windows = [w for w in windows if w.window_start.weekday() >= 5]
        print(f"  Day filter weekend: {before} -> {len(windows)} windows")

    # Run XGB + Fusion backtest
    fg = load_fear_greed(FG_CSV) if FG_CSV.exists() else {}
    ml = MLProcessor(asset=asset, model_dir=MODEL_DIR, confidence_threshold=0.60, model_suffix=model_suffix)
    procs = [
        SpikeDetectionProcessor(spike_threshold=0.003, velocity_threshold=0.0015,
                                lookback_periods=20, min_confidence=0.55),
        TickVelocityProcessor(velocity_threshold_60s=0.001, velocity_threshold_30s=0.0007,
                              min_ticks=5, min_confidence=0.55),
    ]
    sim = BacktestSimulator(procs, SignalFusionEngine(), ml_processor=ml, min_dm=min_dm)
    window_data = sim.run_ticks_collect_probabilities(windows, fg)

    # Load LSTM model
    lstm_path = MODEL_DIR / f"{asset}{model_suffix}_lstm.pt"
    if not lstm_path.exists():
        print(f"  No LSTM model for {asset}")
        return
    lstm_model, lstm_meta = load_lstm(str(lstm_path))
    scaler_mean = np.array(lstm_meta.get("scaler_mean", []), dtype=np.float32)
    scaler_std = np.array(lstm_meta.get("scaler_std", []), dtype=np.float32)
    scaler_std[scaler_std == 0] = 1.0

    # Enrich with Kalshi prices + LSTM predictions
    enriched = []
    from collections import deque

    for win in window_data:
        we = win.get("window_end")
        if we is None:
            continue
        kw = kalshi_windows.get(we)
        if kw is None or kw.outcome is None:
            continue

        # Build tick buffer for LSTM from the window's tick data
        # Find matching original window
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
            cp["kalshi"] = get_kalshi_prices(kw, signal_ts) if signal_ts else None

            # Feed ticks up to checkpoint and get LSTM prediction
            if orig_win and signal_ts:
                while tick_feed_idx < len(orig_win.ticks_during) and orig_win.ticks_during[tick_feed_idx].ts < signal_ts:
                    t = orig_win.ticks_during[tick_feed_idx]
                    tick_buffer.append({"ts": t.ts, "price": t.price, "qty": t.qty, "is_buyer": t.is_buyer})
                    tick_feed_idx += 1

                seq = extract_lstm_sequence(
                    list(tick_buffer), signal_ts,
                    decision_minute=cp.get("dm", 0),
                    window_open_price=orig_win.price_open if orig_win else None,
                )
                if seq is not None:
                    seq_norm = (seq - scaler_mean) / scaler_std
                    seq_norm = np.nan_to_num(seq_norm, nan=0.0)
                    with torch.no_grad():
                        lstm_p = lstm_model(torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)).item()
                    cp["lstm_p"] = lstm_p
                else:
                    cp["lstm_p"] = None
            else:
                cp["lstm_p"] = None

        enriched.append({
            "window_start": win["window_start"],
            "window_end": we,
            "actual_direction": win["actual_direction"],
            "outcome": kw.outcome,
            "checkpoints": win["checkpoints"],
        })

    config = {
        "initial_balance": 50.0 if asset == "BTC" else 25.0,
        "max_contracts_per_trade": 20 if asset == "BTC" else 10,
        "min_price_cents": 55,
    }

    print(f"\n{'='*70}")
    print(f"  Ensemble Combo Sweep: {asset} ({len(enriched)} windows)")
    print(f"{'='*70}")

    THRESHOLDS = [0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
    MAX_PRICES = [70, 75, 80]  # Hard cap at 80c
    MIN_WS = [0.10, 0.20, 0.30]
    MAX_WS = [0.40, 0.50, 0.60, 0.70, 0.80]

    results_by_mode = {}

    for mode_name, sweep_fn in [
        ("XGB+LSTM+Fusion", "3way"),
        ("LSTM+Fusion", "lstm_fusion"),
        ("LSTM+XGB", "lstm_xgb"),
    ]:
        t0 = time.time()
        best = None

        if sweep_fn == "3way":
            for min_a in MIN_WS:
                for max_a in MAX_WS:
                    if max_a <= min_a:
                        continue
                    for min_b in MIN_WS:
                        for max_b in MAX_WS:
                            if max_b <= min_b:
                                continue
                            for thresh in THRESHOLDS:
                                for max_p in MAX_PRICES:
                                    r = sweep_combo(sweep_fn, min_a, max_a, min_b, max_b,
                                                    thresh, max_p, enriched, config, max_dm)
                                    if best is None or r["total_pnl"] > best["total_pnl"]:
                                        best = r
        elif sweep_fn == "lstm_fusion":
            for min_a in MIN_WS:
                for max_a in MAX_WS:
                    if max_a <= min_a:
                        continue
                    for thresh in THRESHOLDS:
                        for max_p in MAX_PRICES:
                            r = sweep_combo(sweep_fn, min_a, max_a, 0, 0,
                                            thresh, max_p, enriched, config, max_dm)
                            if best is None or r["total_pnl"] > best["total_pnl"]:
                                best = r
        elif sweep_fn == "lstm_xgb":
            for min_a in MIN_WS:
                for max_a in MAX_WS:
                    if max_a <= min_a:
                        continue
                    for min_b in MIN_WS:
                        for max_b in MAX_WS:
                            if max_b <= min_b:
                                continue
                            for thresh in THRESHOLDS:
                                for max_p in MAX_PRICES:
                                    r = sweep_combo(sweep_fn, min_a, max_a, min_b, max_b,
                                                    thresh, max_p, enriched, config, max_dm)
                                    if best is None or r["total_pnl"] > best["total_pnl"]:
                                        best = r

        elapsed = time.time() - t0
        results_by_mode[mode_name] = best
        print(f"\n  {mode_name}: PnL=${best['total_pnl']:+.2f} WR={best['win_rate']:.1f}% "
              f"trades={best['traded_count']} thresh={best['threshold']} maxP={best['max_price']}c "
              f"({elapsed:.1f}s)")
        if sweep_fn == "3way":
            print(f"    xgb: min={best['min_w_a']:.2f} max={best['max_w_a']:.2f}  "
                  f"lstm: min={best['min_w_b']:.2f} max={best['max_w_b']:.2f}")
        elif sweep_fn == "lstm_fusion":
            print(f"    lstm: min={best['min_w_a']:.2f} max={best['max_w_a']:.2f}")
        elif sweep_fn == "lstm_xgb":
            print(f"    lstm: min={best['min_w_a']:.2f} max={best['max_w_a']:.2f}  "
                  f"xgb: min={best['min_w_b']:.2f} max={best['max_w_b']:.2f}")

    # Summary
    print(f"\n  --- {asset} Summary ---")
    for mode_name, r in results_by_mode.items():
        print(f"  {mode_name:20s}: PnL=${r['total_pnl']:+8.2f}  WR={r['win_rate']:5.1f}%  trades={r['traded_count']}")

    # Derive config key from model_suffix
    if model_suffix == "_weekday":
        config_key = "ensemble_weekday"
    elif model_suffix == "_weekend":
        config_key = "ensemble_weekend"
    else:
        config_key = "ensemble"

    # Write best 3-way params to config
    best_3way = results_by_mode.get("XGB+LSTM+Fusion")
    if best_3way:
        _write_config(asset, best_3way, config_key=config_key)


def _write_config(asset: str, best: dict, config_key: str = "ensemble"):
    """Update config/trading.json with best 3-way ensemble params."""
    config_path = PROJECT_ROOT / "config" / "trading.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    asset_cfg = cfg.get("assets", {}).get(asset, {})
    ens = asset_cfg.get(config_key, {})

    # ml_weight in config acts as xgb_min_w for dynamic weighting
    ens["ml_weight"] = best["min_w_a"]
    ens["threshold"] = best["threshold"]
    ens["max_price_cents"] = min(best["max_price"], 80)  # Never exceed 80c
    ens["pnl_sweep_total_pnl"] = round(best["total_pnl"], 2)
    ens["pnl_sweep_source"] = "ensemble_combo_sweep"
    ens["win_rate"] = round(best["win_rate"], 1)
    ens["traded_count"] = best["traded_count"]

    asset_cfg[config_key] = ens
    cfg.setdefault("assets", {})[asset] = asset_cfg

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    print(f"  Config updated: {asset} ml_w={best['min_w_a']:.2f} thresh={best['threshold']} maxP={best['max_price']}c")


def main():
    parser = argparse.ArgumentParser(description="Ensemble combo sweep")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--min-dm", type=int, default=2)
    parser.add_argument("--max-dm", type=int, default=8)
    parser.add_argument("--model-suffix", type=str, default="",
                        help="Model filename suffix (e.g. '_weekday')")
    parser.add_argument("--day-filter", choices=["all", "weekday", "weekend"], default="all",
                        help="Filter windows by day of week")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    for asset in assets:
        run_asset(asset, min_dm=args.min_dm, max_dm=args.max_dm,
                  model_suffix=args.model_suffix, day_filter=args.day_filter)


if __name__ == "__main__":
    main()
