"""
Re-entry take-profit backtest.

Strategy:
  - Enter on first ensemble signal (same as live)
  - Take profit when bid >= entry_price + tp_cents (e.g. 20c profit/contract)
  - After TP exit, can re-enter on the next signal checkpoint in the same window
  - Max 1 position at a time
  - If no TP hit, hold to settlement as usual

Uses Kalshi polling data for bid/ask prices and settlement outcomes.

Usage:
  python scripts/backtest_reentry.py --asset BTC --tp-cents 20
  python scripts/backtest_reentry.py --asset BTC,ETH,SOL,XRP --tp-cents 10,15,20,25,30
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


def find_tp_exit(
    kw: KalshiWindow,
    entry_ts: datetime,
    side: str,
    tp_target: int,
) -> tuple[int, datetime] | None:
    """Scan polls after entry for bid >= tp_target.

    Returns (exit_bid, exit_ts) or None.
    """
    if not kw.polls or not kw._poll_timestamps:
        return None

    entry_epoch = entry_ts.timestamp()
    start_idx = bisect_left(kw._poll_timestamps, entry_epoch)
    bid_key = f"{side}_bid"

    for i in range(start_idx, len(kw.polls)):
        poll = kw.polls[i]
        bid = poll.get(bid_key, 0)
        if bid >= tp_target:
            return (bid, poll["ts"])

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


def simulate_reentry(
    window_data: list[dict],
    kalshi_windows: dict[datetime, KalshiWindow],
    ml_w: float,
    ens_threshold: float,
    config: dict,
    tp_cents: int | None,
) -> dict:
    """Simulate re-entry strategy for one TP level.

    tp_cents=None means hold-to-settlement baseline (no TP, no re-entry).
    """
    fusion_w = 1.0 - ml_w
    balance = config["initial_balance"]
    max_contracts = config["max_contracts_per_trade"]
    max_price = config["max_price_cents"]
    min_price = config["min_price_cents"]

    total_entries = 0
    total_tp_exits = 0
    total_settlements = 0
    wins = 0
    losses = 0
    total_pnl = 0.0
    windows_traded = 0
    max_entries_in_window = 0

    for win in window_data:
        window_end = win.get("window_end")
        if window_end is None:
            continue
        kw = kalshi_windows.get(window_end)
        if kw is None or not kw.outcome:
            continue

        checkpoints = win["checkpoints"]
        if not checkpoints:
            continue

        # Track entries within this window
        entries_this_window = 0
        cp_idx = 0  # current checkpoint index

        while cp_idx < len(checkpoints):
            # Find next actionable signal from cp_idx onward
            entry_cp = None
            for i in range(cp_idx, len(checkpoints)):
                cp = checkpoints[i]
                signal_ts = cp.get("signal_ts")
                if signal_ts is None:
                    continue

                ensemble_p = ml_w * cp["ml_p"] + fusion_w * cp["fusion_p"]
                if ensemble_p >= ens_threshold:
                    entry_cp = {"predicted": "BULLISH", "confidence": ensemble_p,
                                "side": "yes", "signal_ts": signal_ts, "idx": i}
                    break
                elif ensemble_p <= 1.0 - ens_threshold:
                    entry_cp = {"predicted": "BEARISH", "confidence": 1.0 - ensemble_p,
                                "side": "no", "signal_ts": signal_ts, "idx": i}
                    break

            if entry_cp is None:
                break  # no more signals in this window

            # Get entry price
            kalshi_prices = get_kalshi_prices(kw, entry_cp["signal_ts"])
            if kalshi_prices is None:
                cp_idx = entry_cp["idx"] + 1
                continue

            side = entry_cp["side"]
            entry_price = kalshi_prices["yes_ask"] if side == "yes" else kalshi_prices["no_ask"]

            if entry_price <= 0 or entry_price >= 100 or entry_price > max_price or entry_price < min_price:
                cp_idx = entry_cp["idx"] + 1
                continue

            contracts = calculate_contracts(balance, entry_price, entry_cp["confidence"], max_contracts)
            if contracts < 1:
                cp_idx = entry_cp["idx"] + 1
                continue

            # Enter
            cost = (entry_price / 100.0) * contracts
            fees = (KALSHI_FEE_CENTS / 100.0) * contracts
            balance -= (cost + fees)
            total_entries += 1
            entries_this_window += 1

            # Try take profit (only if tp_cents is set)
            took_profit = False
            if tp_cents is not None:
                tp_target = entry_price + tp_cents
                if tp_target > 99:
                    tp_target = 99  # can't sell above 99c

                result = find_tp_exit(kw, entry_cp["signal_ts"], side, tp_target)
                if result is not None:
                    exit_bid, exit_ts = result
                    sell_revenue = (exit_bid / 100.0) * contracts
                    sell_fees = (KALSHI_FEE_CENTS / 100.0) * contracts
                    pnl = sell_revenue - cost - fees - sell_fees
                    balance += sell_revenue - sell_fees
                    total_pnl += pnl
                    total_tp_exits += 1
                    wins += 1
                    took_profit = True

                    # Advance cp_idx to first checkpoint AFTER exit_ts
                    exit_epoch = exit_ts.timestamp()
                    found_next = False
                    for j in range(entry_cp["idx"] + 1, len(checkpoints)):
                        cp_ts = checkpoints[j].get("signal_ts")
                        if cp_ts is not None and cp_ts.timestamp() > exit_epoch:
                            cp_idx = j
                            found_next = True
                            break
                    if not found_next:
                        break  # no more checkpoints after exit
                    continue  # loop back to find next signal

            if not took_profit:
                # Hold to settlement
                won = (side == kw.outcome)
                revenue = contracts * 1.00 if won else 0.0
                pnl = revenue - cost - fees
                balance += revenue
                total_pnl += pnl
                total_settlements += 1
                if won:
                    wins += 1
                else:
                    losses += 1
                break  # settlement ends the window (position held to close)

        if entries_this_window > 0:
            windows_traded += 1
            max_entries_in_window = max(max_entries_in_window, entries_this_window)

    win_rate = wins / total_entries * 100 if total_entries else 0.0

    return {
        "total_entries": total_entries,
        "windows_traded": windows_traded,
        "tp_exits": total_tp_exits,
        "settlements": total_settlements,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "final_balance": round(balance, 2),
        "max_entries_in_window": max_entries_in_window,
        "avg_entries_per_window": round(total_entries / windows_traded, 2) if windows_traded else 0,
    }


def run_asset(asset: str, tp_levels: list[int], min_dm: int):
    """Run re-entry backtest for one asset."""
    from core.strategy_brain.signal_processors.ml_processor import MLProcessor

    config = load_config(asset)
    ensemble_cfg = config.get("ensemble")

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

    if not ensemble_weights:
        print(f"No ensemble config for {asset}, skipping")
        return

    ml_w, ens_threshold = ensemble_weights

    print(f"Collecting probabilities for {asset}...")
    processors = build_processors()
    fusion_engine = SignalFusionEngine()
    simulator = BacktestSimulator(
        processors, fusion_engine, ml_processor=ml_processor, min_dm=min_dm,
    )
    window_data = simulator.run_ticks_collect_probabilities(tick_windows, fg_scores)

    # Run baseline (hold) + each TP level
    all_tp = [None] + tp_levels

    results_by_tp: dict[int | None, dict] = {}
    for tp in all_tp:
        results_by_tp[tp] = simulate_reentry(
            window_data, kalshi_windows, ml_w, ens_threshold, config, tp,
        )

    # Print comparison
    kalshi_dates = sorted(set(ct.date() for ct in kalshi_windows.keys()))
    print(f"\n{'='*110}")
    print(f"  {asset} Re-Entry Take-Profit ({len(kalshi_dates)} days)")
    print(f"  Ensemble: ml_w={ml_w:.2f} thresh={ens_threshold:.2f} "
          f"Kelly=[{config['min_price_cents']}c,{config['max_price_cents']}c] "
          f"max_contracts={config['max_contracts_per_trade']}")
    print(f"{'='*110}")
    header = (
        f"  {'Strategy':<16} | {'Entries':>7} | {'TP':>4} | {'Settl':>5} | "
        f"{'Wins':>4} | {'Loss':>4} | {'WR':>6} | {'PnL':>10} | "
        f"{'Final$':>8} | {'Avg/Win':>7} | {'MaxE':>4}"
    )
    print(header)
    print(f"  {'-'*106}")

    hold = results_by_tp[None]
    avg_per_win_hold = hold["total_pnl"] / hold["wins"] if hold["wins"] else 0
    print(
        f"  {'Hold (1x entry)':<16} | {hold['total_entries']:7} | {'--':>4} | "
        f"{hold['settlements']:5} | {hold['wins']:4} | {hold['losses']:4} | "
        f"{hold['win_rate']:5.1f}% | {'+'if hold['total_pnl']>=0 else ''}${hold['total_pnl']:8.2f} | "
        f"${hold['final_balance']:7.2f} | ${avg_per_win_hold:6.3f} | {'--':>4}"
    )

    for tp in tp_levels:
        r = results_by_tp[tp]
        diff = r["total_pnl"] - hold["total_pnl"]
        diff_str = f"({'+'if diff>=0 else ''}{diff:.2f})"
        avg_per_win = r["total_pnl"] / r["wins"] if r["wins"] else 0
        print(
            f"  {'TP +'+str(tp)+'c reentry':<16} | {r['total_entries']:7} | "
            f"{r['tp_exits']:4} | {r['settlements']:5} | {r['wins']:4} | "
            f"{r['losses']:4} | {r['win_rate']:5.1f}% | "
            f"{'+'if r['total_pnl']>=0 else ''}${r['total_pnl']:8.2f} | "
            f"${r['final_balance']:7.2f} | ${avg_per_win:6.3f} | {r['max_entries_in_window']:4}"
            f"  {diff_str}"
        )

    print()


def main():
    logger.remove()
    log_path = PROJECT_ROOT / "logs" / "backtest_reentry.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), mode="w", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="Re-entry take-profit backtest")
    parser.add_argument("--asset", required=True, help="Asset(s), comma-separated")
    parser.add_argument("--tp-cents", default="10,15,20,25,30",
                        help="Take-profit in cents of profit per contract (default: 10,15,20,25,30)")
    parser.add_argument("--min-dm", type=int, default=2, help="Min decision minute")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    tp_levels = [int(x.strip()) for x in args.tp_cents.split(",")]

    for asset in assets:
        run_asset(asset, tp_levels, args.min_dm)


if __name__ == "__main__":
    main()
