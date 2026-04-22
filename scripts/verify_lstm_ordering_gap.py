"""Verify the training-vs-live LSTM ordering hypothesis.

For a chosen timestamp T:
  1. Load last ~300s of Coinbase ticks from data/aggtrades_coinbase/
  2. Load last ~300s of Kraken ticks from data/aggtrades_kraken/
  3. Build two tick buffers:
     (a) TRAINING-STYLE: Coinbase ticks, then Kraken appended after
         (mimics generate_lstm_training_data.py's order)
     (b) LIVE-STYLE: merged + sorted by ts
         (mimics kalshi_strategy.py:1928)
  4. Run the live LSTM on both
  5. Report the prediction delta

If (a) and (b) give different outputs, the ordering bug is real.
"""
import argparse
import csv
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.lstm_features import extract_lstm_sequence
from ml.lstm_model import load_model


def load_aggtrade_csv(path: Path, start_ts: datetime, end_ts: datetime) -> list[dict]:
    """Load ticks within [start_ts, end_ts] from aggTrades CSV.

    Format: id, price, qty, unused, unused, ts_micros, is_buyer, trade_type
    (based on inspected rows).
    """
    ticks = []
    if not path.exists():
        return ticks
    start_us = int(start_ts.timestamp() * 1_000_000)
    end_us = int(end_ts.timestamp() * 1_000_000)
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            try:
                price = float(row[1])
                qty = float(row[2])
                ts_us = int(row[5])
                # Column 6 is is_buyer_maker (maker was buyer = taker SOLD). Invert for is_buyer.
                is_buyer = row[6].strip().lower() not in ("true", "1")
            except (ValueError, IndexError):
                continue
            if ts_us < start_us:
                continue
            if ts_us > end_us:
                break
            ts = datetime.fromtimestamp(ts_us / 1_000_000, tz=timezone.utc)
            ticks.append({"ts": ts, "price": price, "qty": qty, "is_buyer": is_buyer})
    return ticks


def _coinbase_path(asset: str, date: str) -> Path:
    # Naming: ETHUSDT-aggTrades-YYYY-MM-DD.csv under data/aggtrades_coinbase/{ASSET}/
    sym = asset.upper() + "USDT"
    return PROJECT_ROOT / "data" / "aggtrades_coinbase" / asset.upper() / f"{sym}-aggTrades-{date}.csv"


def _kraken_path(asset: str, date: str) -> Path:
    sym = asset.upper() + "USD"
    return PROJECT_ROOT / "data" / "aggtrades_kraken" / asset.upper() / f"{sym}-aggTrades-{date}.csv"


def load_tick_buffer(asset: str, end_ts: datetime, lookback_seconds: int = 300):
    """Return (coinbase_ticks, kraken_ticks) lists for the last N seconds."""
    start_ts = end_ts - timedelta(seconds=lookback_seconds)
    # Could span 2 days if lookback crosses midnight
    dates = set()
    t = start_ts
    while t <= end_ts:
        dates.add(t.strftime("%Y-%m-%d"))
        t += timedelta(minutes=15)
    dates.add(end_ts.strftime("%Y-%m-%d"))

    cb = []
    kr = []
    for d in sorted(dates):
        cb.extend(load_aggtrade_csv(_coinbase_path(asset, d), start_ts, end_ts))
        kr.extend(load_aggtrade_csv(_kraken_path(asset, d), start_ts, end_ts))
    return cb, kr


def run_lstm_once(asset: str, tick_buffer: list[dict], ts: datetime, dm: int,
                  window_open_price: float) -> float | None:
    """Run the live LSTM on a given buffer, return p_bullish."""
    import torch
    model, meta = load_model(str(PROJECT_ROOT / "models" / f"{asset}_lstm.pt"))
    model.eval()
    seq = extract_lstm_sequence(tick_buffer, ts, decision_minute=dm,
                                window_open_price=window_open_price)
    if seq is None:
        return None
    sm = np.array(meta["scaler_mean"], dtype=np.float32)
    ss = np.array(meta["scaler_std"], dtype=np.float32)
    ss[ss == 0] = 1.0
    seq_norm = (seq - sm) / ss
    seq_norm = np.nan_to_num(seq_norm, nan=0.0, posinf=1e6, neginf=-1e6)
    with torch.no_grad():
        x = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)
        return float(model(x).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="SOL",
                    help="Asset to test (pick one where live showed LSTM/XGB disagreement)")
    ap.add_argument("--timestamp", default="",
                    help="ISO timestamp to evaluate at (defaults to 5 min ago)")
    ap.add_argument("--dm", type=int, default=7)
    ap.add_argument("--open-price", type=float, default=0.0,
                    help="Window open price (0 = use first tick in buffer)")
    ap.add_argument("--lookback", type=int, default=300,
                    help="Seconds of tick history to load")
    args = ap.parse_args()

    asset = args.asset.upper()
    if args.timestamp:
        ts = datetime.fromisoformat(args.timestamp.replace("Z", "+00:00"))
    else:
        ts = datetime.now(timezone.utc) - timedelta(minutes=5)

    print(f"Asset: {asset}")
    print(f"Timestamp: {ts.isoformat()}")
    print(f"dm: {args.dm}")
    print(f"Lookback: {args.lookback}s")

    cb, kr = load_tick_buffer(asset, ts, lookback_seconds=args.lookback)
    print(f"Coinbase ticks loaded: {len(cb)}")
    print(f"Kraken ticks loaded: {len(kr)}")
    if not cb and not kr:
        print("No tick data — aborting")
        return

    # BUFFER A (training-style): Coinbase then Kraken, no sort
    buf_training = cb + kr

    # BUFFER B (live-style): merged and sorted
    buf_live = sorted(cb + kr, key=lambda t: t["ts"])

    # Detect if there's actually any ordering difference
    ordering_same = len(buf_training) == len(buf_live) and all(
        buf_training[i]["ts"] == buf_live[i]["ts"] for i in range(len(buf_training))
    )
    if ordering_same:
        print("\nNOTE: training-style and live-style buffers have identical tick order.")
        print("(This happens if one exchange has no ticks in this range, or Kraken ts all > Coinbase ts.)")
    else:
        # Count how many positions differ
        diffs = sum(1 for i in range(len(buf_training)) if buf_training[i]["ts"] != buf_live[i]["ts"])
        pct = 100 * diffs / len(buf_training) if buf_training else 0
        print(f"\nDifferent order at {diffs}/{len(buf_training)} positions ({pct:.1f}%)")

    # Infer a sensible open price if not given
    open_price = args.open_price if args.open_price > 0 else (
        buf_live[0]["price"] if buf_live else 0.0
    )
    print(f"Using window_open_price: {open_price:.4f}")

    # Run LSTM both ways
    p_training = run_lstm_once(asset, buf_training, ts, args.dm, open_price)
    p_live = run_lstm_once(asset, buf_live, ts, args.dm, open_price)

    print(f"\n=== LSTM p_bullish comparison ===")
    print(f"  TRAINING-STYLE buffer (Coinbase then Kraken): {p_training}")
    print(f"  LIVE-STYLE buffer (sorted):                    {p_live}")
    if p_training is not None and p_live is not None:
        delta = p_live - p_training
        print(f"  Delta (live - training):                      {delta:+.4f}")
        if abs(delta) > 0.05:
            print(f"  VERDICT: meaningful difference ({abs(delta):.3f}) — ordering bug is real")
        elif abs(delta) > 0.01:
            print(f"  VERDICT: small difference ({abs(delta):.3f}) — bug is minor")
        else:
            print(f"  VERDICT: negligible difference — ordering bug not the cause (or no cross-exchange timestamps)")


if __name__ == "__main__":
    main()
