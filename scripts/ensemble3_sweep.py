"""
3-Way Ensemble Sweep: XGBoost + LSTM + Fusion.

Collects XGBoost and LSTM probabilities in one pass, then sweeps
(xgb_w, lstm_w, threshold) combos scored by both accuracy and PnL.

Does NOT modify config/trading.json -- analysis only.

Usage:
  python scripts/ensemble3_sweep.py --asset BTC
  python scripts/ensemble3_sweep.py --asset BTC,ETH,SOL,XRP --min-dm 2 --max-dm 8
"""
import argparse
import csv
import json
import sys
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader_ticks import load_aggtrades_multi, generate_tick_windows, resample_ticks
from backtester.data_loader_kalshi import load_kalshi_windows, get_kalshi_prices
from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine
from ml.features import FEATURE_NAMES, extract_features

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ensemble3_sweep"

FEE = 2

# Sweep grid
XGB_WEIGHTS = [round(w * 0.05, 2) for w in range(21)]   # 0.0 to 1.0
LSTM_WEIGHTS = [round(w * 0.05, 2) for w in range(21)]  # 0.0 to 1.0
THRESHOLDS = [0.55, 0.57, 0.58, 0.60, 0.62, 0.65, 0.70]
MAX_PRICES = [60, 65, 70, 75, 80, 85, 90]


def build_processors():
    return [
        SpikeDetectionProcessor(spike_threshold=0.003, velocity_threshold=0.0015, lookback_periods=20, min_confidence=0.55),
        TickVelocityProcessor(velocity_threshold_60s=0.001, velocity_threshold_30s=0.0007, min_ticks=5, min_confidence=0.55),
    ]


def collect_probabilities(asset, windows, ml_processor, lstm_processor, min_dm=2):
    """Replay windows collecting XGBoost, LSTM, and fusion probabilities."""
    processors = build_processors()
    fusion_engine = SignalFusionEngine()
    tickvel = processors[1]

    window_data = []

    for win_idx, window in enumerate(windows):
        price_history = deque(maxlen=200)
        tick_buffer = deque(maxlen=300)
        raw_tick_buffer = deque(maxlen=1200)

        # Feed warmup
        if window.ticks_before:
            warmup_bars = resample_ticks(
                window.ticks_before,
                window.ticks_before[0].ts,
                window.ticks_before[-1].ts + timedelta(seconds=1),
                interval_ms=250,
            )
            for bar in warmup_bars:
                price_history.append(Decimal(str(bar["price"])))
                tick_buffer.append(bar)

        for tick in window.ticks_before:
            raw_tick_buffer.append({
                "ts": tick.ts, "price": tick.price,
                "qty": tick.qty, "is_buyer": tick.is_buyer,
            })

        # Pre-resample decision zone
        if window.ticks_during:
            decision_bars = resample_ticks(
                window.ticks_during,
                window.ticks_during[0].ts,
                window.ticks_during[-1].ts + timedelta(seconds=1),
                interval_ms=250,
            )
        else:
            decision_bars = []

        raw_tick_idx = 0
        bar_idx = 0

        checkpoints = []
        decision_start = window.window_start + timedelta(minutes=5)
        check_interval = timedelta(seconds=10)
        current_check = decision_start + check_interval

        while current_check < window.window_end:
            next_check = current_check + check_interval

            # Feed bars
            while bar_idx < len(decision_bars) and decision_bars[bar_idx]["ts"] < current_check:
                bar = decision_bars[bar_idx]
                price_history.append(Decimal(str(bar["price"])))
                tick_buffer.append(bar)
                bar_idx += 1

            # Feed raw ticks
            while raw_tick_idx < len(window.ticks_during) and window.ticks_during[raw_tick_idx].ts < current_check:
                tick = window.ticks_during[raw_tick_idx]
                raw_tick_buffer.append({
                    "ts": tick.ts, "price": tick.price,
                    "qty": tick.qty, "is_buyer": tick.is_buyer,
                })
                raw_tick_idx += 1

            if len(price_history) < 20:
                current_check = next_check
                continue

            current_price = float(price_history[-1])
            elapsed_s = (current_check - window.window_start).total_seconds()
            dm = int((elapsed_s - 300) / 60)

            if dm < min_dm:
                current_check = next_check
                continue

            # Momentum
            if len(price_history) >= 6:
                prev = float(price_history[-6])
                momentum = (current_price - prev) / prev if prev != 0 else 0.0
            else:
                momentum = 0.0

            metadata = {
                "tick_buffer": list(tick_buffer),
                "raw_tick_buffer": list(raw_tick_buffer),
                "spot_price": current_price,
                "momentum": momentum,
                "sentiment_score": 50,
                "decision_minute": dm,
                "window_open_price": window.price_open,
            }

            # XGBoost
            ml_p = 0.5
            if ml_processor is not None:
                try:
                    raw_p = ml_processor.predict_proba(
                        Decimal(str(current_price)), list(price_history), metadata
                    )
                    if raw_p is not None:
                        ml_p = raw_p
                except Exception:
                    pass

            # LSTM
            lstm_p = 0.5
            if lstm_processor is not None:
                try:
                    raw_p = lstm_processor.predict_proba(
                        Decimal(str(current_price)), list(price_history), metadata
                    )
                    if raw_p is not None:
                        lstm_p = raw_p
                except Exception:
                    pass

            # Fusion
            fusion_p = 0.5
            try:
                tickvel_signal = tickvel.process(
                    Decimal(str(current_price)), list(price_history), metadata
                )
                signals = [s for s in [tickvel_signal] if s is not None]
                if signals:
                    fused = fusion_engine.fuse_signals(signals)
                    direction_str = str(fused.direction).upper()
                    if "BULLISH" in direction_str:
                        fusion_p = 0.5 + fused.confidence / 200.0
                    elif "BEARISH" in direction_str:
                        fusion_p = 0.5 - fused.confidence / 200.0
            except Exception:
                pass

            checkpoints.append({
                "dm": dm,
                "ml_p": ml_p,
                "lstm_p": lstm_p,
                "fusion_p": fusion_p,
                "signal_ts": current_check,
            })

            current_check = next_check

        if checkpoints:
            window_data.append({
                "window_start": window.window_start,
                "window_end": window.window_end,
                "actual_direction": window.actual_direction,
                "checkpoints": checkpoints,
            })

    return window_data


def sweep_accuracy(window_data, max_dm=None):
    """Sweep 3-way weights by accuracy (Binance direction only)."""
    results = []
    valid_combos = [
        (xw, lw, th) for xw in XGB_WEIGHTS for lw in LSTM_WEIGHTS for th in THRESHOLDS
        if xw + lw <= 1.0
    ]

    for xgb_w, lstm_w, thresh in valid_combos:
        fusion_w = 1.0 - xgb_w - lstm_w
        correct = 0
        traded = 0
        total = len(window_data)

        for win in window_data:
            actual = win["actual_direction"]
            predicted = None

            for cp in win["checkpoints"]:
                if max_dm is not None and cp["dm"] > max_dm:
                    continue
                ep = xgb_w * cp["ml_p"] + lstm_w * cp["lstm_p"] + fusion_w * cp["fusion_p"]
                if ep >= thresh:
                    predicted = "BULLISH"
                    break
                elif ep <= 1.0 - thresh:
                    predicted = "BEARISH"
                    break

            if predicted is not None:
                traded += 1
                if predicted == actual:
                    correct += 1

        if traded > 0:
            accuracy = correct / traded * 100
            results.append({
                "xgb_weight": xgb_w,
                "lstm_weight": lstm_w,
                "fusion_weight": round(fusion_w, 2),
                "threshold": thresh,
                "accuracy": round(accuracy, 1),
                "net_correct": correct - (traded - correct),
                "traded_count": traded,
                "traded_pct": round(traded / total * 100, 1),
            })

    return results


def sweep_pnl(enriched_windows, config, max_dm=None):
    """Sweep 3-way weights by dollar PnL using Kalshi data."""
    results = []

    # Use top accuracy combos as candidates (filter: acc >= 70%, traded >= 20)
    # plus a few max_price options
    valid_combos = [
        (xw, lw, th) for xw in XGB_WEIGHTS for lw in LSTM_WEIGHTS for th in THRESHOLDS
        if xw + lw <= 1.0
    ]

    balance_init = config["initial_balance"]
    max_contracts = config["max_contracts_per_trade"]

    for xgb_w, lstm_w, thresh in valid_combos:
        fusion_w = 1.0 - xgb_w - lstm_w

        for max_price in MAX_PRICES:
            min_price = config["min_price_cents"]
            balance = balance_init
            traded = 0
            wins = 0
            total_pnl = 0.0

            for win in enriched_windows:
                outcome = win.get("outcome")
                if not outcome:
                    continue

                predicted = None
                confidence = 0.0
                entry_price = 0
                side = ""

                for cp in win["checkpoints"]:
                    if max_dm is not None and cp["dm"] > max_dm:
                        continue
                    if cp.get("kalshi") is None:
                        continue

                    ep = xgb_w * cp["ml_p"] + lstm_w * cp["lstm_p"] + fusion_w * cp["fusion_p"]

                    if ep >= thresh:
                        p = cp["kalshi"]["yes_ask"]
                        if p <= 0 or p >= 100 or p < min_price or p > max_price:
                            continue
                        predicted = "BULLISH"
                        confidence = ep
                        side = "yes"
                        entry_price = p
                        break
                    elif ep <= 1.0 - thresh:
                        p = cp["kalshi"]["no_ask"]
                        if p <= 0 or p >= 100 or p < min_price or p > max_price:
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

                won = side == outcome
                revenue = contracts * 1.0 if won else 0.0
                pnl = revenue - cost - fees
                balance += revenue

                traded += 1
                total_pnl += pnl
                if won:
                    wins += 1

            if traded >= 5:
                results.append({
                    "xgb_weight": xgb_w,
                    "lstm_weight": lstm_w,
                    "fusion_weight": round(fusion_w, 2),
                    "threshold": thresh,
                    "max_price": max_price,
                    "total_pnl": round(total_pnl, 2),
                    "win_rate": round(wins / traded * 100, 1),
                    "traded_count": traded,
                    "wins": wins,
                    "losses": traded - wins,
                    "final_balance": round(balance, 2),
                })

    return results


def load_config(asset):
    config_path = PROJECT_ROOT / "config" / "trading.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return {"initial_balance": 25.0, "max_contracts_per_trade": 10, "max_price_cents": 90, "min_price_cents": 65}

    defaults = cfg.get("defaults", {})
    asset_cfg = cfg.get("assets", {}).get(asset.upper(), {})
    return {
        "initial_balance": asset_cfg.get("initial_balance", defaults.get("initial_balance", 25.0)),
        "max_contracts_per_trade": asset_cfg.get("max_contracts_per_trade", defaults.get("max_contracts_per_trade", 10)),
        "max_price_cents": asset_cfg.get("max_price_cents", defaults.get("max_price_cents", 90)),
        "min_price_cents": asset_cfg.get("min_price_cents", defaults.get("min_price_cents", 65)),
    }


def run_asset(asset, min_dm=2, max_dm=8):
    """Full 3-way ensemble analysis for one asset."""
    from core.strategy_brain.signal_processors.ml_processor import MLProcessor
    from core.strategy_brain.signal_processors.lstm_processor import LSTMProcessor

    print(f"\n{'='*70}")
    print(f"  3-Way Ensemble Sweep: {asset}")
    print(f"{'='*70}")

    # Load models
    try:
        ml_proc = MLProcessor(asset=asset, model_dir=MODEL_DIR, confidence_threshold=0.60)
        print(f"  XGBoost model loaded")
    except FileNotFoundError:
        print(f"  No XGBoost model for {asset}, skipping")
        return

    try:
        lstm_proc = LSTMProcessor(asset=asset, model_dir=MODEL_DIR, confidence_threshold=0.60)
        print(f"  LSTM model loaded")
    except FileNotFoundError:
        print(f"  No LSTM model for {asset}, skipping")
        return

    # Load Kalshi data
    kalshi_windows = load_kalshi_windows(KALSHI_DIR, asset)
    if not kalshi_windows:
        print(f"  No Kalshi data for {asset}")
        return

    kalshi_close_times = set(kalshi_windows.keys())
    kalshi_dates = sorted(set(ct.date() for ct in kalshi_windows.keys()))
    days_back = (date.today() - kalshi_dates[0]).days + 2

    # Load tick data
    print(f"  Loading aggTrades (last {days_back} days)...")
    ticks = load_aggtrades_multi(DATA_DIR, asset, days=days_back)
    if not ticks:
        print(f"  No tick data for {asset}")
        return

    windows = generate_tick_windows(ticks, min_warmup_ticks=1, min_during_ticks=1)
    windows = [w for w in windows if w.window_end in kalshi_close_times]
    print(f"  {len(windows)} Kalshi-matched windows, {len(kalshi_dates)} days")

    if not windows:
        print(f"  No overlapping windows")
        return

    # Phase 1: Collect probabilities
    print(f"\n  Phase 1: Collecting XGBoost + LSTM + Fusion probabilities...")
    t0 = time.time()
    window_data = collect_probabilities(asset, windows, ml_proc, lstm_proc, min_dm=min_dm)
    t1 = time.time()
    total_cp = sum(len(w["checkpoints"]) for w in window_data)
    print(f"  Collected {total_cp} checkpoints across {len(window_data)} windows in {t1-t0:.1f}s")

    # Phase 2: Enrich with Kalshi data
    print(f"\n  Phase 2: Enriching with Kalshi prices...")
    enriched = []
    for win in window_data:
        kw = kalshi_windows.get(win["window_end"])
        if kw is None or kw.outcome is None:
            continue
        for cp in win["checkpoints"]:
            ts = cp.get("signal_ts")
            if ts:
                cp["kalshi"] = get_kalshi_prices(kw, ts)
            else:
                cp["kalshi"] = None
        enriched.append({**win, "outcome": kw.outcome})
    print(f"  {len(enriched)} windows with outcomes")

    config = load_config(asset)

    # Phase 3: Accuracy sweep
    print(f"\n  Phase 3: Accuracy sweep...")
    t2 = time.time()
    acc_results = sweep_accuracy(window_data, max_dm=max_dm)
    t3 = time.time()
    print(f"  {len(acc_results)} combos swept in {t3-t2:.1f}s")

    # Top 10 by accuracy (with min trades)
    acc_qualified = [r for r in acc_results if r["traded_count"] >= 20 and r["traded_pct"] >= 5]
    acc_qualified.sort(key=lambda r: r["net_correct"], reverse=True)

    print(f"\n  --- Top 10 by Net Correct (accuracy sweep) ---")
    print(f"  {'XGB_W':>5} {'LSTM_W':>6} {'FUS_W':>5} {'Thr':>5} | {'Acc%':>5} {'Net':>5} {'Trd':>4} {'Trd%':>5}")
    print(f"  {'-'*55}")
    for r in acc_qualified[:10]:
        print(f"  {r['xgb_weight']:5.2f} {r['lstm_weight']:6.2f} {r['fusion_weight']:5.2f} {r['threshold']:5.2f} | "
              f"{r['accuracy']:5.1f} {r['net_correct']:5d} {r['traded_count']:4d} {r['traded_pct']:5.1f}")

    # Compare: XGBoost-only vs LSTM-only vs best 3-way
    print(f"\n  --- Model Comparison ---")
    for label, filter_fn in [
        ("XGBoost only", lambda r: r["lstm_weight"] == 0 and r["xgb_weight"] > 0),
        ("LSTM only", lambda r: r["xgb_weight"] == 0 and r["lstm_weight"] > 0),
        ("3-Way best", lambda r: r["xgb_weight"] > 0 and r["lstm_weight"] > 0),
    ]:
        subset = [r for r in acc_qualified if filter_fn(r)]
        if subset:
            best = max(subset, key=lambda r: r["net_correct"])
            print(f"  {label:15s}: xgb={best['xgb_weight']:.2f} lstm={best['lstm_weight']:.2f} "
                  f"fus={best['fusion_weight']:.2f} thresh={best['threshold']:.2f} "
                  f"acc={best['accuracy']:.1f}% net={best['net_correct']} trades={best['traded_count']}")

    # Phase 4: PnL sweep
    print(f"\n  Phase 4: PnL sweep...")
    t4 = time.time()
    pnl_results = sweep_pnl(enriched, config, max_dm=max_dm)
    t5 = time.time()
    print(f"  {len(pnl_results)} combos swept in {t5-t4:.1f}s")

    pnl_results.sort(key=lambda r: r["total_pnl"], reverse=True)

    print(f"\n  --- Top 15 by PnL ---")
    print(f"  {'Rank':>4} | {'PnL':>8} | {'WR%':>5} | {'Trd':>4} | {'W':>3} {'L':>3} | "
          f"{'XGB':>5} {'LSTM':>5} {'FUS':>5} {'Thr':>5} {'MaxP':>4}")
    print(f"  {'-'*75}")
    for i, r in enumerate(pnl_results[:15], 1):
        print(f"  {i:4d} | ${r['total_pnl']:+7.2f} | {r['win_rate']:5.1f} | {r['traded_count']:4d} | "
              f"{r['wins']:3d} {r['losses']:3d} | "
              f"{r['xgb_weight']:5.2f} {r['lstm_weight']:5.2f} {r['fusion_weight']:5.2f} {r['threshold']:5.2f} {r['max_price']:4d}")

    # PnL comparison
    print(f"\n  --- PnL Model Comparison ---")
    for label, filter_fn in [
        ("XGBoost only", lambda r: r["lstm_weight"] == 0 and r["xgb_weight"] > 0),
        ("LSTM only", lambda r: r["xgb_weight"] == 0 and r["lstm_weight"] > 0),
        ("3-Way best", lambda r: r["xgb_weight"] > 0 and r["lstm_weight"] > 0),
    ]:
        subset = [r for r in pnl_results if filter_fn(r)]
        if subset:
            best = max(subset, key=lambda r: r["total_pnl"])
            print(f"  {label:15s}: PnL=${best['total_pnl']:+.2f} WR={best['win_rate']:.1f}% "
                  f"trades={best['traded_count']} xgb={best['xgb_weight']:.2f} lstm={best['lstm_weight']:.2f} "
                  f"fus={best['fusion_weight']:.2f} thresh={best['threshold']:.2f} maxP={best['max_price']}c")

    # Export CSVs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    acc_csv = OUTPUT_DIR / f"{asset}_accuracy.csv"
    if acc_results:
        with open(acc_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(acc_results[0].keys()))
            writer.writeheader()
            writer.writerows(sorted(acc_results, key=lambda r: r["net_correct"], reverse=True))
        print(f"\n  Exported accuracy: {acc_csv}")

    pnl_csv = OUTPUT_DIR / f"{asset}_pnl.csv"
    if pnl_results:
        with open(pnl_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pnl_results[0].keys()))
            writer.writeheader()
            writer.writerows(pnl_results)
        print(f"  Exported PnL: {pnl_csv}")


def main():
    logger.remove()
    log_path = PROJECT_ROOT / "logs" / "ensemble3_sweep.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), mode="w", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="3-way ensemble sweep (XGBoost + LSTM + Fusion)")
    parser.add_argument("--asset", required=True, help="Asset(s), comma-separated")
    parser.add_argument("--min-dm", type=int, default=2, help="Min decision minute")
    parser.add_argument("--max-dm", type=int, default=8, help="Max decision minute")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    for asset in assets:
        run_asset(asset, min_dm=args.min_dm, max_dm=args.max_dm)


if __name__ == "__main__":
    main()
