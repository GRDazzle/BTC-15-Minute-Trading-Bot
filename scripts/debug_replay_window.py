"""Debug a single window: dump ensemble_p at every check_ts so we can see
where replay agrees/disagrees with live.

Usage:
    python scripts/debug_replay_window.py --asset SOL \\
        --window-start 2026-04-24T21:15:00 \\
        --window-end   2026-04-24T21:30:00
"""
import argparse
import csv
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger as _l
_l.remove()
_l.add(sys.stderr, level="WARNING")

from core.strategy_brain.signal_processors.ml_processor import MLProcessor
from core.strategy_brain.signal_processors.lstm_processor import LSTMProcessor
from core.tick_window_slicer import TickWindowSlicer
from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
KRAKEN_DIR = PROJECT_ROOT / "data" / "aggtrades_kraken"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
MODEL_DIR = PROJECT_ROOT / "models"


def _load_csv_ticks(base, asset, quote, dates):
    out = []
    for d in dates:
        path = base / asset / f"{asset}{quote}-aggTrades-{d}.csv"
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 7:
                    continue
                try:
                    ts_raw = int(row[5])
                except (ValueError, IndexError):
                    continue
                ts = datetime.fromtimestamp(ts_raw / 1_000_000, tz=timezone.utc) if ts_raw > 10**14 else datetime.fromtimestamp(ts_raw / 1000, tz=timezone.utc)
                out.append({
                    "ts": ts,
                    "price": float(row[1]),
                    "qty": float(row[2]),
                    "is_buyer": row[6].strip().lower() not in ("true", "1"),
                })
    out.sort(key=lambda t: t["ts"])
    return out


def _formula_c(ml_p, lstm_p, k_xgb, k_lstm):
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
    ap.add_argument("--window-start", required=True)
    ap.add_argument("--window-end", required=True)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--k-xgb", type=float, default=6.0)
    ap.add_argument("--k-lstm", type=float, default=6.0)
    ap.add_argument("--min-dm", type=int, default=7)
    ap.add_argument("--check-interval", type=int, default=2)
    args = ap.parse_args()

    asset = args.asset.upper()
    ws = datetime.fromisoformat(args.window_start).replace(tzinfo=timezone.utc)
    we = datetime.fromisoformat(args.window_end).replace(tzinfo=timezone.utc)
    decision_start = ws + timedelta(minutes=5)

    # Load ticks (previous + current day to cover the 900s lookback)
    dates = [
        (ws - timedelta(days=1)).strftime("%Y-%m-%d"),
        ws.strftime("%Y-%m-%d"),
    ]
    cb = _load_csv_ticks(DATA_DIR, asset, "USDT", dates)
    kr = _load_csv_ticks(KRAKEN_DIR, asset, "USD", dates)
    print(f"[{asset}] loaded {len(cb)} CB + {len(kr)} KR ticks")

    slicer = TickWindowSlicer()
    if cb: slicer.extend("coinbase", cb)
    if kr: slicer.extend("kraken", kr)

    # Processors
    ml_proc = MLProcessor(asset=asset, model_dir=MODEL_DIR)
    try:
        lstm_proc = LSTMProcessor(asset=asset, model_dir=MODEL_DIR)
    except FileNotFoundError:
        lstm_proc = None

    poll_idx = KalshiPollIndex(KALSHI_DIR, asset)
    event_ticker = window_start_to_event_ticker(asset, we)
    print(f"[{asset}] event_ticker={event_ticker}")

    # Window open price
    wop_tick = slicer.get_first_at_or_after(ws, ("coinbase", "kraken"))
    if not wop_tick:
        print("no tick at window start"); return
    wop = wop_tick["price"]
    print(f"[{asset}] window_open_price={wop}")

    # Step through 2s checkpoints in decision zone
    check_ts = decision_start + timedelta(seconds=args.check_interval)
    print()
    print(f"{'check_ts':<28} {'dm':>3} {'ml_p':>7} {'lstm_p':>7} {'ens_p':>7} {'y_ask':>5} {'n_ask':>5} {'would_fire':>12}")
    print("-" * 90)
    while check_ts < we:
        dm = int((check_ts - decision_start).total_seconds() // 60)
        if dm < args.min_dm:
            check_ts += timedelta(seconds=args.check_interval); continue
        buf = slicer.get_merged_window(check_ts, 900, ("coinbase", "kraken"))
        if len(buf) < 20:
            check_ts += timedelta(seconds=args.check_interval); continue
        kr_buf = slicer.get_merged_window(check_ts, 900, ("kraken",))
        current_price = buf[-1]["price"]
        price_history = [t["price"] for t in buf]

        poll = poll_idx.find_poll(event_ticker, check_ts)
        if not poll:
            check_ts += timedelta(seconds=args.check_interval); continue
        k_yes_ask = int(poll["yes_ask"])
        k_yes_bid = int(poll["yes_bid"])
        k_no_ask = int(poll.get("no_ask", 100 - k_yes_bid))
        k_mtc = float(poll.get("mins_to_close", (we - check_ts).total_seconds() / 60))
        poll_hist = poll_idx.get_poll_history(event_ticker, check_ts, 60)

        sma5 = sum(price_history[-5:]) / min(5, len(price_history))
        sma15 = sum(price_history[-15:]) / min(15, len(price_history))
        sma30 = sum(price_history[-30:]) / min(30, len(price_history))

        metadata = {
            "tick_buffer": buf,
            "raw_tick_buffer": buf,
            "kraken_tick_buffer": kr_buf,
            "kraken_current_price": kr_buf[-1]["price"] if kr_buf else None,
            "window_open_price": wop,
            "sma5": sma5, "sma15": sma15, "sma30": sma30,
            "kalshi_yes_ask": k_yes_ask, "kalshi_yes_bid": k_yes_bid,
            "kalshi_no_ask": k_no_ask, "kalshi_mins_to_close": k_mtc,
            "kalshi_poll_history": poll_hist,
            "decision_minute": dm,
        }
        ml_p = ml_proc.predict_proba(Decimal(str(current_price)), price_history, metadata)
        lstm_p = None
        if lstm_proc:
            lstm_p = lstm_proc.predict_proba(Decimal(str(current_price)), price_history, metadata)
        ens_p = _formula_c(ml_p, lstm_p, args.k_xgb, args.k_lstm) if ml_p is not None else None

        if ens_p is None:
            fire = "skip"
        elif ens_p >= args.threshold:
            fire = "BULL"
        elif ens_p <= 1.0 - args.threshold:
            fire = "BEAR"
        else:
            fire = "."

        lstm_str = f"{lstm_p:.4f}" if lstm_p is not None else "None"
        ml_str = f"{ml_p:.4f}" if ml_p is not None else "None"
        ens_str = f"{ens_p:.4f}" if ens_p is not None else "None"
        print(f"{check_ts.isoformat():<28} {dm:>3} {ml_str:>7} {lstm_str:>7} {ens_str:>7} "
              f"{k_yes_ask:>5} {k_no_ask:>5} {fire:>12}")
        check_ts += timedelta(seconds=args.check_interval)


if __name__ == "__main__":
    main()
