"""
Sweep dynamic ML weight parameters against Kalshi PnL data.

Dynamic weight: ml_w = min_w + (max_w - min_w) * confidence^k
where confidence = abs(ml_p - 0.5) * 2

Sweeps: k (exponent), min_w, max_w, threshold, max_price
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader_ticks import load_aggtrades_multi, generate_tick_windows
from backtester.data_loader_kalshi import load_kalshi_windows
from backtester.data_loader import load_fear_greed
from backtester.simulator import BacktestSimulator
from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.signal_processors.ml_processor import MLProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine
from backtester.data_loader_kalshi import get_kalshi_prices

logger.remove()

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
FG_CSV = PROJECT_ROOT / "data" / "historical" / "fear_greed.csv"
MODEL_DIR = PROJECT_ROOT / "models"

FEE_CENTS = 2


def sweep_dynamic_pnl(
    min_w: float,
    max_w: float,
    k: float,
    threshold: float,
    max_price: int,
    enriched_windows: list[dict],
    config: dict,
    max_dm: int | None = None,
) -> dict:
    """Evaluate one dynamic weight combo by dollar PnL."""
    balance = config["initial_balance"]
    max_contracts = config["max_contracts_per_trade"]
    min_price = config["min_price_cents"]

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
        confidence = 0.0

        for cp in win["checkpoints"]:
            if cp.get("kalshi") is None:
                continue
            if max_dm is not None and cp["dm"] > max_dm:
                continue

            ml_p = cp["ml_p"]
            fusion_p = cp["fusion_p"]

            # Dynamic weight
            raw_conf = abs(ml_p - 0.5) * 2.0
            dynamic_ml_w = min_w + (max_w - min_w) * (raw_conf ** k)
            ensemble_p = dynamic_ml_w * ml_p + (1.0 - dynamic_ml_w) * fusion_p

            if ensemble_p >= threshold:
                p = cp["kalshi"]["yes_ask"]
                if p <= 0 or p >= 100 or p < min_price or p > max_price:
                    continue
                predicted = "BULLISH"
                confidence = ensemble_p
                side = "yes"
                entry_price = p
                break
            elif ensemble_p <= 1.0 - threshold:
                p = cp["kalshi"]["no_ask"]
                if p <= 0 or p >= 100 or p < min_price or p > max_price:
                    continue
                predicted = "BEARISH"
                confidence = 1.0 - ensemble_p
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
        "min_w": min_w,
        "max_w": max_w,
        "k": k,
        "threshold": threshold,
        "max_price": max_price,
        "total_pnl": total_pnl,
        "win_rate": wr,
        "traded_count": traded_count,
        "win_count": win_count,
    }


def run_asset(asset: str, min_dm: int = 2, max_dm: int | None = 8):
    """Run dynamic weight sweep for one asset."""
    from datetime import date

    kalshi_windows = load_kalshi_windows(KALSHI_DIR, asset)
    if not kalshi_windows:
        print(f"  No Kalshi data for {asset}")
        return

    kalshi_close_times = set(kalshi_windows.keys())
    kalshi_dates = sorted(set(ct.date() for ct in kalshi_windows.keys()))
    days_back = (date.today() - kalshi_dates[0]).days + 2

    ticks = load_aggtrades_multi(DATA_DIR, asset, days=days_back)
    if not ticks:
        print(f"  No aggTrades for {asset}")
        return

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
    sim = BacktestSimulator(procs, SignalFusionEngine(), ml_processor=ml, min_dm=min_dm)
    window_data = sim.run_ticks_collect_probabilities(windows, fg)
    # Enrich with Kalshi prices
    enriched = []
    for win in window_data:
        we = win.get("window_end")
        if we is None:
            continue
        kw = kalshi_windows.get(we)
        if kw is None or kw.outcome is None:
            continue
        for cp in win["checkpoints"]:
            signal_ts = cp.get("signal_ts")
            cp["kalshi"] = get_kalshi_prices(kw, signal_ts) if signal_ts else None
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
        "min_price_cents": 15,
    }

    print(f"\n{'='*70}")
    print(f"  Dynamic Weight Sweep: {asset} ({len(enriched)} windows)")
    print(f"{'='*70}")

    # Sweep parameters
    all_results = []
    K_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]
    MIN_W_VALUES = [0.10, 0.20, 0.30]
    MAX_W_VALUES = [0.70, 0.80, 0.90, 1.00]
    THRESHOLDS = [0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
    MAX_PRICES = [70, 75, 80, 85, 90, 95]

    total_combos = len(K_VALUES) * len(MIN_W_VALUES) * len(MAX_W_VALUES) * len(THRESHOLDS) * len(MAX_PRICES)
    print(f"  Sweeping {total_combos} combos...")

    t0 = time.time()
    for k_val in K_VALUES:
        for min_w in MIN_W_VALUES:
            for max_w in MAX_W_VALUES:
                if max_w <= min_w:
                    continue
                for thresh in THRESHOLDS:
                    for max_p in MAX_PRICES:
                        r = sweep_dynamic_pnl(min_w, max_w, k_val, thresh, max_p,
                                              enriched, config, max_dm=max_dm)
                        all_results.append(r)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # Sort by PnL
    all_results.sort(key=lambda r: r["total_pnl"], reverse=True)

    # Also get the fixed-weight best for comparison
    fixed_best = None
    for mlw in [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        for thresh in THRESHOLDS:
            for max_p in MAX_PRICES:
                r = sweep_dynamic_pnl(mlw, mlw, 1.0, thresh, max_p,
                                      enriched, config, max_dm=max_dm)
                r["ml_weight"] = mlw
                if fixed_best is None or r["total_pnl"] > fixed_best["total_pnl"]:
                    fixed_best = r

    # Display results
    print(f"\n  --- Top 10 Dynamic Weight Results ---")
    print(f"  {'Rank':>4} | {'PnL':>8} | {'WR%':>5} | {'Trd':>4} | {'min_w':>5} {'max_w':>5} {'k':>4} {'Thr':>5} {'MaxP':>4}")
    print(f"  {'-'*65}")
    for i, r in enumerate(all_results[:10]):
        print(f"  {i+1:4d} | ${r['total_pnl']:+7.2f} | {r['win_rate']:5.1f} | {r['traded_count']:4d} | "
              f"{r['min_w']:5.2f} {r['max_w']:5.2f} {r['k']:4.1f} {r['threshold']:5.2f} {r['max_price']:4d}")

    print(f"\n  --- Fixed Weight Best ---")
    print(f"  PnL=${fixed_best['total_pnl']:+.2f} WR={fixed_best['win_rate']:.1f}% trades={fixed_best['traded_count']} "
          f"ml_w={fixed_best['ml_weight']} thresh={fixed_best['threshold']} maxP={fixed_best['max_price']}c")

    best = all_results[0]
    delta = best["total_pnl"] - fixed_best["total_pnl"]
    print(f"\n  Dynamic vs Fixed: ${delta:+.2f}")

    # Save CSV
    out_dir = PROJECT_ROOT / "output" / "dynamic_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{asset}_dynamic_sweep.csv"
    fields = ["min_w", "max_w", "k", "threshold", "max_price", "total_pnl", "win_rate", "traded_count"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  Exported: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Dynamic ML weight sweep")
    parser.add_argument("--asset", required=True, help="Asset(s), comma-separated")
    parser.add_argument("--min-dm", type=int, default=2)
    parser.add_argument("--max-dm", type=int, default=8)
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    for asset in assets:
        run_asset(asset, min_dm=args.min_dm, max_dm=args.max_dm)


if __name__ == "__main__":
    main()
