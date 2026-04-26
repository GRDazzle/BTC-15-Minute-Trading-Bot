"""Diagnostic: run the replay-path inference for a single (asset, timestamp)
and compare to what live logged at the same moment.

Usage:
  python scripts/diag_live_vs_replay_single.py \\
    --asset XRP --eval-ts 2026-04-24T14:12:06 --dm 7 \\
    --window-start 2026-04-24T14:00:00

Reconstructs the tick buffer from CSV archives exactly as replay_live_inference
does, runs MLProcessor + LSTMProcessor, and prints ml_p / lstm_p / ens_p.
Also prints live's logged values for comparison (from signal_log.csv).
"""
import argparse
import bisect
import csv
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.add(sys.stderr, level="WARNING")

from core.strategy_brain.signal_processors.ml_processor import MLProcessor
from core.strategy_brain.signal_processors.lstm_processor import LSTMProcessor
from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker

COINBASE_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
KRAKEN_DIR = PROJECT_ROOT / "data" / "aggtrades_kraken"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
MODEL_DIR = PROJECT_ROOT / "models"


def _parse_ts(ts_raw: int) -> datetime:
    if ts_raw > 10**14:
        return datetime.fromtimestamp(ts_raw / 1_000_000, tz=timezone.utc)
    return datetime.fromtimestamp(ts_raw / 1000, tz=timezone.utc)


def _load_aggtrades(base_dir, asset, quote, date):
    path = base_dir / asset / f"{asset}{quote}-aggTrades-{date:%Y-%m-%d}.csv"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            try:
                price = float(row[1])
                qty = float(row[2])
                ts = _parse_ts(int(row[5]))
            except (ValueError, IndexError):
                continue
            is_buyer_maker = row[6].strip().lower() in ("true", "1")
            out.append({
                "ts": ts, "price": price, "qty": qty,
                "is_buyer": not is_buyer_maker,
            })
    return out


def _formula_c(ml_p, lstm_p, k_xgb, k_lstm):
    if lstm_p is None:
        return ml_p, 1.0, 0.0
    xgb_conf = abs(ml_p - 0.5) * 2.0
    lstm_conf = abs(lstm_p - 0.5) * 2.0
    xgb_w_raw = xgb_conf ** k_xgb
    lstm_w_raw = lstm_conf ** k_lstm
    total = xgb_w_raw + lstm_w_raw
    if total < 1e-9:
        return 0.5, 0.0, 0.0
    ens = (xgb_w_raw * ml_p + lstm_w_raw * lstm_p) / total
    return ens, xgb_w_raw / total, lstm_w_raw / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--eval-ts", required=True, help="ISO timestamp to evaluate at")
    ap.add_argument("--window-start", required=True, help="ISO window_start")
    ap.add_argument("--dm", type=int, required=True)
    ap.add_argument("--k-xgb", type=float, default=3.0)
    ap.add_argument("--k-lstm", type=float, default=8.0)
    ap.add_argument("--model-suffix", default="_align2s")
    args = ap.parse_args()

    eval_ts = datetime.fromisoformat(args.eval_ts).replace(tzinfo=timezone.utc) \
        if "+" not in args.eval_ts else datetime.fromisoformat(args.eval_ts)
    window_start = datetime.fromisoformat(args.window_start).replace(tzinfo=timezone.utc) \
        if "+" not in args.window_start else datetime.fromisoformat(args.window_start)
    window_end = window_start + timedelta(minutes=15)

    print(f"=== DIAGNOSTIC: {args.asset} at {eval_ts.isoformat()} ===\n")

    # Load both days of ticks (prev + current for warmup)
    prev_day = (eval_ts - timedelta(days=1)).date()
    curr_day = eval_ts.date()
    cb_ticks = _load_aggtrades(COINBASE_DIR, args.asset, "USDT", prev_day) \
             + _load_aggtrades(COINBASE_DIR, args.asset, "USDT", curr_day)
    kr_ticks = _load_aggtrades(KRAKEN_DIR, args.asset, "USD", prev_day) \
             + _load_aggtrades(KRAKEN_DIR, args.asset, "USD", curr_day)
    merged = sorted(cb_ticks + kr_ticks, key=lambda t: t["ts"])
    merged_ts = [t["ts"] for t in merged]
    kr_only = sorted(kr_ticks, key=lambda t: t["ts"])
    kr_ts = [t["ts"] for t in kr_only]
    print(f"Loaded ticks: CB={len(cb_ticks)} KR={len(kr_ticks)} merged={len(merged)}")

    # Slice tick buffer: last 900s ending at eval_ts
    cutoff = eval_ts - timedelta(seconds=900)
    lo = bisect.bisect_left(merged_ts, cutoff)
    hi = bisect.bisect_right(merged_ts, eval_ts)
    tick_buffer = merged[lo:hi]
    kr_lo = bisect.bisect_left(kr_ts, cutoff)
    kr_hi = bisect.bisect_right(kr_ts, eval_ts)
    kraken_tick_buffer = kr_only[kr_lo:kr_hi]
    print(f"Tick buffer (last 900s): {len(tick_buffer)} ticks (Kraken: {len(kraken_tick_buffer)})")
    if tick_buffer:
        first_ts = tick_buffer[0]["ts"]
        last_ts = tick_buffer[-1]["ts"]
        span_s = (last_ts - first_ts).total_seconds()
        # Compute 180s-slice (LSTM window)
        lstm_cutoff = last_ts - timedelta(seconds=180)
        lstm_slice = [t for t in tick_buffer if t["ts"] >= lstm_cutoff]
        cb_in_lstm = sum(1 for t in lstm_slice if not any(kt for kt in kraken_tick_buffer if kt is t))
        print(f"  buffer span: {first_ts.strftime('%H:%M:%S')} -> {last_ts.strftime('%H:%M:%S')} ({span_s:.1f}s)")
        print(f"  last 180s (LSTM input): {len(lstm_slice)} ticks")
        # Truncate to maxlen=5000 and see how that changes things
        if len(tick_buffer) > 5000:
            truncated_5k = tick_buffer[-5000:]
            first_5k = truncated_5k[0]["ts"]
            span_5k = (last_ts - first_5k).total_seconds()
            print(f"  [if live buffer were capped at 5000 merged ticks]: span would be {span_5k:.1f}s starting at {first_5k.strftime('%H:%M:%S')}")
    if not tick_buffer:
        print("No ticks — cannot run inference")
        return

    current_price = tick_buffer[-1]["price"]
    price_history = [t["price"] for t in tick_buffer]

    # Window open price
    wop_idx = bisect.bisect_left(merged_ts, window_start)
    window_open_price = merged[wop_idx]["price"] if wop_idx < len(merged) else tick_buffer[0]["price"]
    print(f"Replay WOP: {window_open_price} (first tick at/after window_start {window_start}: "
          f"{merged[wop_idx]['ts'].strftime('%H:%M:%S.%f')[:-3] if wop_idx < len(merged) else 'N/A'})")

    # Kalshi poll snapshot
    poll_idx = KalshiPollIndex(KALSHI_DIR, args.asset)
    event_ticker = window_start_to_event_ticker(args.asset, window_end)
    poll = poll_idx.find_poll(event_ticker, eval_ts)
    k_yes_ask = int(poll["yes_ask"]) if poll else 50
    k_yes_bid = int(poll["yes_bid"]) if poll else 50
    k_no_ask = int(poll.get("no_ask", 100 - k_yes_bid)) if poll else 50
    k_mtc = float(poll.get("mins_to_close", 0.0)) if poll else 0.0
    poll_history = poll_idx.get_poll_history(event_ticker, eval_ts, 60)
    print(f"Kalshi poll at {eval_ts}: yes_ask={k_yes_ask} no_ask={k_no_ask} mtc={k_mtc:.1f}")

    # SMAs
    sma5 = sum(price_history[-5:]) / min(5, len(price_history))
    sma15 = sum(price_history[-15:]) / min(15, len(price_history))
    sma30 = sum(price_history[-30:]) / min(30, len(price_history))

    metadata = {
        "tick_buffer": tick_buffer,
        "raw_tick_buffer": tick_buffer,
        "kraken_tick_buffer": kraken_tick_buffer,
        "kraken_current_price": kraken_tick_buffer[-1]["price"] if kraken_tick_buffer else None,
        "window_open_price": window_open_price,
        "sma5": sma5, "sma15": sma15, "sma30": sma30,
        "kalshi_yes_ask": k_yes_ask, "kalshi_yes_bid": k_yes_bid,
        "kalshi_no_ask": k_no_ask, "kalshi_mins_to_close": k_mtc,
        "kalshi_poll_history": poll_history,
        "decision_minute": args.dm,
    }

    # Load processors
    ml_proc = MLProcessor(asset=args.asset, model_dir=MODEL_DIR, model_suffix=args.model_suffix)
    lstm_proc = LSTMProcessor(asset=args.asset, model_dir=MODEL_DIR, model_suffix=args.model_suffix)

    # Inference — first with full replay buffer
    ml_p = ml_proc.predict_proba(Decimal(str(current_price)), price_history, metadata)
    lstm_p = lstm_proc.predict_proba(Decimal(str(current_price)), price_history, metadata)

    print(f"\n=== Replay-path (full 900s buffer) ===")
    print(f"  ml_p    = {ml_p:.4f}")
    print(f"  lstm_p  = {lstm_p:.4f}" if lstm_p is not None else "  lstm_p  = None")

    # Mimic live's buffer cap: last 5000 CB ticks + all KR ticks in 900s,
    # then merge+sort. This matches live's state.raw_tick_buffer(maxlen=5000) +
    # state.kraken_tick_buffer(maxlen=5000) merged behavior.
    cb_only = [t for t in tick_buffer if t not in kraken_tick_buffer]  # approximate
    # simpler: use just coinbase stream
    cb_buffer = [t for t in tick_buffer if t in tick_buffer]  # placeholder, all ticks
    # Actually split by source: CB ticks are ones from the Coinbase stream (no way to
    # distinguish here since both merged have same dict shape). Use cb_ticks from
    # earlier load directly:
    cb_ts = [t["ts"] for t in sorted(cb_ticks, key=lambda t: t["ts"])]
    cb_sorted = sorted(cb_ticks, key=lambda t: t["ts"])
    # Last 5000 CB ticks with ts <= eval_ts
    cb_hi = bisect.bisect_right(cb_ts, eval_ts)
    cb_lo = max(0, cb_hi - 5000)
    cb_capped = cb_sorted[cb_lo:cb_hi]
    # All KR ticks in last 900s with ts <= eval_ts (kraken maxlen=5000 usually
    # covers way more than 900s, so 900s slice fits)
    kr_capped = [t for t in kraken_tick_buffer if (eval_ts - t["ts"]).total_seconds() <= 900]
    live_like_buffer = sorted(cb_capped + kr_capped, key=lambda t: t["ts"])
    print(f"\n=== Replay-path (live-like cap: 5000 CB + KR in 900s) ===")
    print(f"  CB capped: {len(cb_capped)} ticks (span: "
          f"{cb_capped[0]['ts'].strftime('%H:%M:%S') if cb_capped else 'N/A'} -> "
          f"{cb_capped[-1]['ts'].strftime('%H:%M:%S') if cb_capped else 'N/A'})")
    print(f"  KR capped: {len(kr_capped)} ticks")
    print(f"  Total merged: {len(live_like_buffer)} ticks")

    metadata_live_like = dict(metadata)
    metadata_live_like["tick_buffer"] = live_like_buffer
    metadata_live_like["raw_tick_buffer"] = live_like_buffer

    ml_p_2 = ml_proc.predict_proba(Decimal(str(current_price)), price_history, metadata_live_like)
    lstm_p_2 = lstm_proc.predict_proba(Decimal(str(current_price)), price_history, metadata_live_like)

    print(f"  ml_p    = {ml_p_2:.4f}")
    print(f"  lstm_p  = {lstm_p_2:.4f}" if lstm_p_2 is not None else "  lstm_p  = None")

    if lstm_p is not None:
        ens, xgb_share, lstm_share = _formula_c(ml_p, lstm_p, args.k_xgb, args.k_lstm)
        print(f"  k_xgb={args.k_xgb} k_lstm={args.k_lstm}")
        print(f"  ens_p   = {ens:.4f}  (xgb_share={xgb_share:.3f} lstm_share={lstm_share:.3f})")

    # Find closest live signal_log entry
    print(f"\n=== Live signal_log (closest entries) ===")
    live_path = PROJECT_ROOT / "output" / "signal_log.csv"
    target_prefix = eval_ts.strftime("%Y-%m-%dT%H:%M:")
    target_near = eval_ts.strftime("%Y-%m-%dT%H:%M")
    with open(live_path) as f:
        rows = []
        for r in csv.DictReader(f):
            if r.get("asset") == args.asset and r.get("timestamp", "").startswith(target_near):
                rows.append(r)
    if rows:
        for r in rows[-5:]:
            print(f"  {r['timestamp'][:19]} dm={r['dm']} ml_p={r['ml_p']} lstm_p={r['lstm_p']} ens={r['ensemble_p']} {r['direction']} action={r['action']}")
    else:
        print(f"  (no live entries near {target_near} for {args.asset})")


if __name__ == "__main__":
    main()
