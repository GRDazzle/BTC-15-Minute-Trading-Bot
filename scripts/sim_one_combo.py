"""Run BacktestSimulator with a single Formula C combo, same CB+KR slicer
semantics as live. Reports trade count + PnL.

Purpose: validate sweep-vs-replay alignment after Phase 4a/Phase 6 migration.
Replay already verified byte-identical to baseline; this confirms sweep's
simulator, given the same slicer-backed metadata, produces similar trade
decisions to replay over the same range.

Usage:
    python scripts/sim_one_combo.py --asset BTC --from 2026-04-22 --to 2026-04-24 \\
        --threshold 0.62 --k-xgb 4.5 --k-lstm 3.0 --min-dm 2 --max-price 80
"""
import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger as _l
_l.remove()
_l.add(sys.stderr, level="WARNING")

from backtester.data_loader_ticks import load_aggtrades_multi, generate_tick_windows
from backtester.simulator import BacktestSimulator
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine
from core.strategy_brain.signal_processors.ml_processor import MLProcessor
from core.strategy_brain.signal_processors.lstm_processor import LSTMProcessor
from core.tick_window_slicer import TickWindowSlicer
from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
KRAKEN_DIR = PROJECT_ROOT / "data" / "aggtrades_kraken"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
MODEL_DIR = PROJECT_ROOT / "models"

KALSHI_FEE_CENTS = 3  # per contract


def formula_c(ml_p: float, lstm_p: float | None, k_xgb: float, k_lstm: float) -> float:
    """2-way XGB+LSTM confidence-weighted (same as live + replay + sweep)."""
    xgb_conf = abs(ml_p - 0.5) * 2.0
    if lstm_p is None:
        return ml_p
    lstm_conf = abs(lstm_p - 0.5) * 2.0
    wx = xgb_conf ** k_xgb
    wl = lstm_conf ** k_lstm
    tot = wx + wl
    if tot < 1e-9:
        return 0.5
    return (wx * ml_p + wl * lstm_p) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--from", dest="from_date", required=True)
    ap.add_argument("--to", dest="to_date", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--k-xgb", dest="k_xgb", type=float, required=True)
    ap.add_argument("--k-lstm", dest="k_lstm", type=float, required=True)
    ap.add_argument("--min-dm", dest="min_dm", type=int, required=True)
    ap.add_argument("--max-price", dest="max_price", type=int, required=True)
    ap.add_argument("--min-price", dest="min_price", type=int, default=55)
    ap.add_argument("--contracts", type=int, default=None, help="override max contracts; default read from config")
    ap.add_argument("--xgb-suffix", default="")
    ap.add_argument("--lstm-suffix", default="")
    args = ap.parse_args()

    asset = args.asset.upper()
    from_d = date.fromisoformat(args.from_date)
    to_d = date.fromisoformat(args.to_date)
    days_back = (date.today() - from_d).days + 2

    # Load ticks
    print(f"Loading ticks for {asset} (last {days_back} days)...")
    ticks = load_aggtrades_multi(DATA_DIR, asset, days=days_back)
    kr_ticks = load_aggtrades_multi(KRAKEN_DIR, asset, days=days_back)
    print(f"  {len(ticks)} Coinbase + {len(kr_ticks)} Kraken ticks loaded")

    # Build slicer (same pattern as pnl_sweep and live)
    slicer = TickWindowSlicer()
    if ticks:
        slicer.extend("coinbase", [{
            "ts": t.ts, "price": t.price, "qty": t.qty, "is_buyer": t.is_buyer,
        } for t in ticks])
    if kr_ticks:
        slicer.extend("kraken", [{
            "ts": t.ts, "price": t.price, "qty": t.qty, "is_buyer": t.is_buyer,
        } for t in kr_ticks])

    # Generate windows, filter to date range
    windows = generate_tick_windows(ticks, min_warmup_ticks=1, min_during_ticks=1)
    start_utc = datetime.combine(from_d, datetime.min.time(), tzinfo=timezone.utc)
    end_utc = datetime.combine(to_d + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    windows = [w for w in windows if start_utc <= w.window_start < end_utc]
    print(f"  {len(windows)} windows in range {from_d} -> {to_d}")

    # Load processors
    ml = MLProcessor(asset=asset, model_dir=MODEL_DIR, confidence_threshold=0.60, model_suffix=args.xgb_suffix)
    lstm = None
    try:
        lstm = LSTMProcessor(asset=asset, model_dir=MODEL_DIR, model_suffix=args.lstm_suffix)
    except FileNotFoundError:
        print(f"  no LSTM model for {asset}{args.lstm_suffix}; running XGB only")

    # Kalshi settlements
    poll_idx = KalshiPollIndex(KALSHI_DIR, asset)

    # Simulator — check_interval 2s, skip fusion, slicer-backed tick buffer
    sim = BacktestSimulator(
        processors=[],
        fusion_engine=SignalFusionEngine(),
        ml_processor=ml,
        lstm_processor=lstm,
        min_dm=args.min_dm,
        check_interval_seconds=2,
        skip_fusion=True,
        batch_lstm=True,
        slicer=slicer,
        slicer_lookback_seconds=900,
    )

    print("\nCollecting XGB + LSTM probabilities...")
    window_data = sim.run_ticks_collect_probabilities(windows, fg_scores={})

    # Apply Formula C + threshold per checkpoint, take first actionable per window
    trades = 0
    wins = 0
    pnl_total = 0.0

    for win in window_data:
        event_ticker = window_start_to_event_ticker(asset, win["window_end"])
        outcome = poll_idx.get_outcome(event_ticker)
        for cp in win["checkpoints"]:
            dm = cp["dm"]
            if dm < args.min_dm:
                continue
            ensemble_p = formula_c(cp["ml_p"], cp.get("lstm_p"), args.k_xgb, args.k_lstm)
            threshold = args.threshold
            if ensemble_p >= threshold:
                direction, side = "BULLISH", "yes"
                conf = ensemble_p
            elif ensemble_p <= 1.0 - threshold:
                direction, side = "BEARISH", "no"
                conf = 1.0 - ensemble_p
            else:
                continue

            # Kalshi price snapshot nearest the checkpoint
            poll = poll_idx.find_poll(event_ticker, cp["signal_ts"])
            if poll is None:
                continue
            k_yes_ask = int(poll["yes_ask"])
            k_no_ask = int(poll.get("no_ask", 100 - int(poll.get("yes_bid", 0))))
            entry_price = k_yes_ask if side == "yes" else k_no_ask
            if entry_price > args.max_price or entry_price < args.min_price:
                continue

            # Fixed-size bet (same style as replay harness)
            contracts = args.contracts or 20
            cost = contracts * (entry_price + KALSHI_FEE_CENTS) / 100.0

            if outcome is None:
                break  # no settlement; skip the window
            won = (direction == "BULLISH" and outcome == "yes") or \
                  (direction == "BEARISH" and outcome == "no")
            pnl = contracts - cost if won else -cost
            pnl_total += pnl
            trades += 1
            if won:
                wins += 1
            break  # one trade per window

    wr = 100.0 * wins / trades if trades else 0.0
    print("\n=" * 30)
    print(f"Combo: threshold={args.threshold} k_xgb={args.k_xgb} k_lstm={args.k_lstm} "
          f"min_dm={args.min_dm} max_price={args.max_price}")
    print(f"  windows={len(windows)}  trades={trades}  wins={wins}  WR={wr:.1f}%  PnL={pnl_total:+.2f}")


if __name__ == "__main__":
    main()
