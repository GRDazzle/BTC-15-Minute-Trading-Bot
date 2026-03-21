"""
Take-profit backtest: compare holding to settlement vs selling at a target price.

Uses the existing PnL backtest infrastructure + Kalshi polling data to check
whether selling at (e.g.) 95c when the price reaches that level would have
improved total PnL.

For each historical trade:
  1. Replay signal pipeline to get entry (same as backtest_kalshi_pnl.py)
  2. After entry, scan subsequent Kalshi polls for take-profit trigger
  3. Compare: hold-to-settlement PnL vs take-profit PnL

Usage:
  python scripts/backtest_take_profit.py --asset BTC --tp 95
  python scripts/backtest_take_profit.py --asset BTC,ETH,SOL,XRP --tp 90,92,95,97
"""
import argparse
import sys
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader import load_fear_greed
from backtester.data_loader_ticks import load_aggtrades_multi, generate_tick_windows
from backtester.data_loader_kalshi import load_kalshi_windows, get_kalshi_prices, KalshiWindow
from backtester.simulator import BacktestSimulator

from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades"
KALSHI_DATA_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
FG_CSV = PROJECT_ROOT / "data" / "historical" / "fear_greed.csv"
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"

KALSHI_FEE_CENTS = 2


def load_config(asset: str) -> dict:
    import json
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        return {"initial_balance": 25.0, "max_contracts_per_trade": 10,
                "max_price_cents": 90, "min_price_cents": 55, "ensemble": None}
    defaults = config.get("defaults", {})
    asset_cfg = config.get("assets", {}).get(asset.upper(), {})
    return {
        "initial_balance": asset_cfg.get("initial_balance", defaults.get("initial_balance", 25.0)),
        "max_contracts_per_trade": asset_cfg.get("max_contracts_per_trade", defaults.get("max_contracts_per_trade", 10)),
        "max_price_cents": asset_cfg.get("max_price_cents", defaults.get("max_price_cents", 90)),
        "min_price_cents": asset_cfg.get("min_price_cents", defaults.get("min_price_cents", 15)),
        "ensemble": asset_cfg.get("ensemble"),
    }


def calculate_contracts(balance, price_cents, confidence, max_contracts):
    cost_per = (price_cents + KALSHI_FEE_CENTS) / 100.0
    if cost_per <= 0 or balance <= 0:
        return 0
    max_by_balance = int(balance / cost_per)
    scale = min(1.0, confidence)
    desired = max(1, int(max_by_balance * scale))
    return min(desired, max_contracts)


def find_take_profit_price(
    kw: KalshiWindow,
    entry_ts: datetime,
    side: str,
    tp_cents: int,
) -> int | None:
    """Scan polls after entry_ts for take-profit trigger.

    For YES side: look for yes_bid >= tp_cents (we'd sell YES at that bid)
    For NO side: look for no_bid >= tp_cents (we'd sell NO at that bid)

    Returns the bid price at which TP was triggered, or None if never hit.
    """
    if not kw.polls or not kw._poll_timestamps:
        return None

    entry_epoch = entry_ts.timestamp()
    start_idx = bisect_left(kw._poll_timestamps, entry_epoch)

    bid_key = f"{side}_bid"

    for i in range(start_idx, len(kw.polls)):
        poll = kw.polls[i]
        bid = poll.get(bid_key, 0)
        if bid >= tp_cents:
            return bid

    return None


def build_processors():
    return [
        SpikeDetectionProcessor(
            spike_threshold=0.003, velocity_threshold=0.0015,
            lookback_periods=20, min_confidence=0.55,
        ),
        TickVelocityProcessor(
            velocity_threshold_60s=0.001, velocity_threshold_30s=0.0007,
            min_ticks=5, min_confidence=0.55,
        ),
    ]


def run_asset(asset: str, tp_levels: list[int], min_dm: int):
    """Run take-profit comparison for one asset across multiple TP levels."""
    from core.strategy_brain.signal_processors.ml_processor import MLProcessor

    config = load_config(asset)
    ensemble_cfg = config.get("ensemble")

    # Load data
    print(f"\nLoading data for {asset}...")
    kalshi_windows = load_kalshi_windows(KALSHI_DATA_DIR, asset)
    if not kalshi_windows:
        print(f"No Kalshi data for {asset}")
        return

    ticks = load_aggtrades_multi(DATA_DIR, asset)
    if not ticks:
        print(f"No aggTrades data for {asset}")
        return

    tick_windows = generate_tick_windows(ticks)
    fg_scores = load_fear_greed(FG_CSV) if FG_CSV.exists() else {}

    # Load ML + ensemble
    ml_processor = None
    ensemble_weights = None
    if ensemble_cfg:
        try:
            ml_processor = MLProcessor(asset=asset, model_dir=MODEL_DIR, confidence_threshold=0.60)
            ml_w = ensemble_cfg.get("ml_weight", 0.65)
            threshold = ensemble_cfg.get("threshold", 0.70)
            ensemble_weights = (ml_w, threshold)
        except (FileNotFoundError, ImportError):
            pass

    # Collect probabilities
    print(f"Collecting probabilities for {asset}...")
    processors = build_processors()
    fusion_engine = SignalFusionEngine()
    simulator = BacktestSimulator(
        processors, fusion_engine, ml_processor=ml_processor, min_dm=min_dm,
    )
    window_data = simulator.run_ticks_collect_probabilities(tick_windows, fg_scores)

    if not ensemble_weights:
        print(f"No ensemble config for {asset}, skipping")
        return

    ml_w, ens_threshold = ensemble_weights
    fusion_w = 1.0 - ml_w
    max_contracts = config["max_contracts_per_trade"]
    max_price = config["max_price_cents"]
    min_price = config["min_price_cents"]

    # For each TP level, simulate full PnL
    # Also track "hold" (no TP) as baseline
    all_tp_levels = [None] + tp_levels  # None = hold to settlement

    results_by_tp: dict[int | None, dict] = {}

    for tp in all_tp_levels:
        balance = config["initial_balance"]
        wins = 0
        losses = 0
        tp_exits = 0
        traded = 0
        total_pnl = 0.0

        for win in window_data:
            window_end = win.get("window_end")
            if window_end is None:
                continue
            kw = kalshi_windows.get(window_end)
            if kw is None:
                continue
            if not kw.outcome:
                continue

            # Find first signal crossing threshold
            predicted = None
            confidence = 0.0
            signal_ts = None
            side = ""
            entry_price = 0

            for cp in win["checkpoints"]:
                signal_ts_cp = cp.get("signal_ts")
                if signal_ts_cp is None:
                    continue

                ensemble_p = ml_w * cp["ml_p"] + fusion_w * cp["fusion_p"]
                if ensemble_p >= ens_threshold:
                    predicted = "BULLISH"
                    confidence = ensemble_p
                    side = "yes"
                    signal_ts = signal_ts_cp
                    break
                elif ensemble_p <= 1.0 - ens_threshold:
                    predicted = "BEARISH"
                    confidence = 1.0 - ensemble_p
                    side = "no"
                    signal_ts = signal_ts_cp
                    break

            if predicted is None or signal_ts is None:
                continue

            # Get entry price
            kalshi_prices = get_kalshi_prices(kw, signal_ts)
            if kalshi_prices is None:
                continue

            entry_price = kalshi_prices["yes_ask"] if side == "yes" else kalshi_prices["no_ask"]
            if entry_price <= 0 or entry_price >= 100:
                continue
            if entry_price > max_price or entry_price < min_price:
                continue

            contracts = calculate_contracts(balance, entry_price, confidence, max_contracts)
            if contracts < 1:
                continue

            cost = (entry_price / 100.0) * contracts
            fees = (KALSHI_FEE_CENTS / 100.0) * contracts
            balance -= (cost + fees)
            traded += 1

            # Check take-profit
            took_profit = False
            if tp is not None:
                tp_bid = find_take_profit_price(kw, signal_ts, side, tp)
                if tp_bid is not None:
                    # Sell at tp_bid, pay fee again on the sell side
                    sell_revenue = (tp_bid / 100.0) * contracts
                    sell_fees = (KALSHI_FEE_CENTS / 100.0) * contracts
                    pnl = sell_revenue - cost - fees - sell_fees
                    balance += sell_revenue - sell_fees
                    took_profit = True
                    tp_exits += 1
                    wins += 1  # TP is always a win

            if not took_profit:
                # Hold to settlement
                won = (side == kw.outcome)
                revenue = contracts * 1.00 if won else 0.0
                pnl = revenue - cost - fees
                balance += revenue
                if won:
                    wins += 1
                else:
                    losses += 1

            total_pnl += pnl

        results_by_tp[tp] = {
            "traded": traded,
            "wins": wins,
            "losses": losses,
            "tp_exits": tp_exits,
            "total_pnl": round(total_pnl, 2),
            "final_balance": round(balance, 2),
            "win_rate": round(wins / traded * 100, 1) if traded else 0.0,
        }

    # Print comparison
    kalshi_dates = sorted(set(ct.date() for ct in kalshi_windows.keys()))
    print(f"\n{'='*85}")
    print(f"  {asset} Take-Profit Comparison ({len(kalshi_dates)} days of Kalshi data)")
    print(f"  Ensemble: ml_w={ml_w:.2f} thresh={ens_threshold:.2f} "
          f"Kelly=[{min_price}c,{max_price}c] max_contracts={max_contracts}")
    print(f"{'='*85}")
    header = (f"  {'Strategy':<18} | {'Trades':>6} | {'Wins':>4} | {'Loss':>4} | "
              f"{'TP Exits':>8} | {'WinRate':>7} | {'PnL':>10} | {'Final$':>8}")
    print(header)
    print(f"  {'-'*81}")

    hold = results_by_tp[None]
    print(f"  {'Hold (baseline)':<18} | {hold['traded']:6} | {hold['wins']:4} | "
          f"{hold['losses']:4} | {'--':>8} | {hold['win_rate']:6.1f}% | "
          f"{'+'if hold['total_pnl']>=0 else ''}${hold['total_pnl']:8.2f} | "
          f"${hold['final_balance']:7.2f}")

    for tp in tp_levels:
        r = results_by_tp[tp]
        diff = r["total_pnl"] - hold["total_pnl"]
        diff_str = f"({'+'if diff>=0 else ''}{diff:.2f})"
        print(f"  {'TP @'+str(tp)+'c':<18} | {r['traded']:6} | {r['wins']:4} | "
              f"{r['losses']:4} | {r['tp_exits']:8} | {r['win_rate']:6.1f}% | "
              f"{'+'if r['total_pnl']>=0 else ''}${r['total_pnl']:8.2f} | "
              f"${r['final_balance']:7.2f}  {diff_str}")

    print()


def main():
    logger.remove()
    log_path = PROJECT_ROOT / "logs" / "backtest_take_profit.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), mode="w", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="Take-profit backtest")
    parser.add_argument("--asset", required=True, help="Asset(s), comma-separated")
    parser.add_argument("--tp", default="90,92,95,97",
                        help="Take-profit levels in cents, comma-separated (default: 90,92,95,97)")
    parser.add_argument("--min-dm", type=int, default=2, help="Min decision minute")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    tp_levels = [int(x.strip()) for x in args.tp.split(",")]

    for asset in assets:
        run_asset(asset, tp_levels, args.min_dm)


if __name__ == "__main__":
    main()
