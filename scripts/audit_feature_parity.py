"""Feature parity audit.

For a chosen window+dm:
  1. Load features from the training CSV (reference)
  2. Reconstruct the tick buffer at that timestamp from aggtrades archives
  3. Run extract_features() on the reconstructed buffer
  4. Diff the two feature vectors
  5. Run XGB on both → compare ml_p predictions

Purpose:
  - Verify features.py is deterministic (offline recompute == training CSV)
  - If live ever diverges from training, the bug is in tick-buffer state
    (not feature computation math)

Usage:
  python scripts/audit_feature_parity.py --asset BTC --window 2026-04-21T20:00:00 --dm 7
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import FEATURE_NAMES, extract_features
from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker


def load_tick_buffer(asset, end_ts, lookback_s):
    """Load aggtrade ticks from Coinbase + Kraken CSVs covering
    [end_ts - lookback_s, end_ts]. Returns sorted by ts."""
    start_ts = end_ts - timedelta(seconds=lookback_s)

    def _parse(path, suffix, invert_is_buyer):
        ticks = []
        if not path.exists():
            return ticks
        with open(path) as f:
            for row in csv.reader(f):
                if len(row) < 7: continue
                try:
                    ts_raw = int(row[5])
                except ValueError:
                    continue
                # Auto-detect ms vs us
                if ts_raw > 10**14:
                    ts = datetime.fromtimestamp(ts_raw/1_000_000, tz=timezone.utc)
                else:
                    ts = datetime.fromtimestamp(ts_raw/1_000, tz=timezone.utc)
                if ts < start_ts or ts > end_ts:
                    continue
                is_buyer_raw = row[6].strip().lower() in ("true","1")
                is_buyer = (not is_buyer_raw) if invert_is_buyer else is_buyer_raw
                ticks.append({
                    "ts": ts, "price": float(row[1]),
                    "qty": float(row[2]), "is_buyer": is_buyer,
                })
        return ticks

    ticks = []
    # Coinbase and Kraken CSVs both use is_buyer_maker in column 6 → invert for is_buyer
    for offset_days in (1, 0):
        d = (end_ts - timedelta(days=offset_days)).strftime("%Y-%m-%d")
        cb_path = PROJECT_ROOT / "data" / "aggtrades_coinbase" / asset / f"{asset}USDT-aggTrades-{d}.csv"
        kr_path = PROJECT_ROOT / "data" / "aggtrades_kraken" / asset / f"{asset}USD-aggTrades-{d}.csv"
        ticks.extend(_parse(cb_path, "USDT", invert_is_buyer=True))
        kr_ticks = _parse(kr_path, "USD", invert_is_buyer=True)
    ticks.sort(key=lambda t: t["ts"])
    # For kraken tick buffer separately (for xgb v10+ features)
    kr_only = []
    for offset_days in (1, 0):
        d = (end_ts - timedelta(days=offset_days)).strftime("%Y-%m-%d")
        kr_path = PROJECT_ROOT / "data" / "aggtrades_kraken" / asset / f"{asset}USD-aggTrades-{d}.csv"
        kr_only.extend(_parse(kr_path, "USD", invert_is_buyer=True))
    kr_only.sort(key=lambda t: t["ts"])
    return ticks, kr_only


def load_training_row(asset, ws_iso, dm):
    path = PROJECT_ROOT / "ml" / "training_data" / f"{asset}_features.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("window_start") != ws_iso:
                continue
            try:
                row_dm = int(float(row.get("minute_in_window", -1)))
            except ValueError:
                continue
            if row_dm == dm:
                return row
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--window", default="",
                    help="Window start ISO (e.g. 2026-04-21T20:00:00+00:00). "
                         "Defaults to the most recent window in training CSV.")
    ap.add_argument("--dm", type=int, default=7)
    ap.add_argument("--offset-seconds", type=int, default=0,
                    help="Seconds past the top of dm minute to evaluate at "
                         "(0..59). Use this to match live's actual fire time.")
    ap.add_argument("--model-suffix", default="",
                    help="Optional model suffix (e.g. _align2s) to audit a "
                         "staging model instead of the primary one.")
    args = ap.parse_args()

    asset = args.asset.upper()

    # Find a recent window to audit if not specified
    if args.window:
        ws_iso = args.window
        if "+" not in ws_iso:
            ws_iso = ws_iso.replace("Z", "+00:00")
            if "+" not in ws_iso:
                ws_iso = ws_iso + "+00:00"
    else:
        # Find latest window with dm == args.dm in training CSV
        path = PROJECT_ROOT / "ml" / "training_data" / f"{asset}_features.csv"
        latest = None
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    rd = int(float(row.get("minute_in_window", -1)))
                except ValueError:
                    continue
                if rd != args.dm:
                    continue
                ws = row.get("window_start", "")
                if not latest or ws > latest:
                    latest = ws
        if not latest:
            print("No matching window in training CSV")
            return
        ws_iso = latest

    print(f"Auditing: asset={asset}, window={ws_iso}, dm={args.dm}")

    training_row = load_training_row(asset, ws_iso, args.dm)
    if not training_row:
        print(f"Training row missing for {ws_iso} dm={args.dm}")
        return

    window_dt = datetime.fromisoformat(ws_iso)
    eval_ts = window_dt + timedelta(minutes=5 + args.dm, seconds=args.offset_seconds)
    if args.offset_seconds:
        print(f"  eval_ts offset +{args.offset_seconds}s -> {eval_ts.isoformat()}")

    # Load tick buffer from aggtrades (matches training-time reconstruction)
    print(f"Loading ticks covering last 900s ending at {eval_ts.isoformat()}")
    tick_buffer, kraken_ticks = load_tick_buffer(asset, eval_ts, lookback_s=900)
    print(f"  Coinbase+Kraken combined: {len(tick_buffer)} ticks")
    print(f"  Kraken only: {len(kraken_ticks)} ticks")

    if not tick_buffer:
        print("No ticks — cannot recompute features")
        return

    # Build price_history from recent prices
    current_price = tick_buffer[-1]["price"]
    price_history = [t["price"] for t in tick_buffer[-2000:]]

    # Fetch window_open_price from tick data (price at window_dt + 0)
    wop_ts = window_dt
    window_open_price = None
    for t in tick_buffer:
        if t["ts"] >= wop_ts:
            window_open_price = t["price"]
            break
    if window_open_price is None:
        window_open_price = tick_buffer[0]["price"]

    # Fetch Kalshi poll data at eval_ts (same as training does)
    kalshi_idx = KalshiPollIndex(PROJECT_ROOT / "data" / "kalshi_polls", asset)
    event_ticker = window_start_to_event_ticker(asset, window_dt + timedelta(minutes=15))
    poll = kalshi_idx.find_poll(event_ticker, eval_ts)
    k_yes_ask = poll["yes_ask"] if poll else None
    k_yes_bid = poll.get("yes_bid") if poll else None
    k_no_ask = poll.get("no_ask", 100 - (k_yes_bid or 50)) if poll else None
    k_mtc = poll.get("mins_to_close") if poll else None

    # Reconstruct features offline
    offline_feats = extract_features(
        tick_buffer=tick_buffer,
        price_history=price_history,
        current_price=current_price,
        timestamp=eval_ts,
        decision_minute=args.dm,
        window_open_price=window_open_price,
        kalshi_yes_ask=k_yes_ask,
        kalshi_yes_bid=k_yes_bid,
        kalshi_no_ask=k_no_ask,
        kalshi_mins_to_close=k_mtc,
        kraken_tick_buffer=kraken_ticks,
        kraken_current_price=(kraken_ticks[-1]["price"] if kraken_ticks else None),
    )

    # Compare to training CSV row
    print(f"\n{'Feature':<35} {'training_csv':>12} {'offline_recomp':>15} {'delta':>10} {'pct':>8}")
    print("-" * 85)
    mismatches = 0
    for fn in FEATURE_NAMES:
        try:
            csv_val = float(training_row.get(fn, 0.0))
        except ValueError:
            csv_val = 0.0
        off_val = offline_feats.get(fn, None)
        if off_val is None:
            print(f"{fn:<35} {csv_val:>12.4f} {'(missing)':>15}")
            mismatches += 1
            continue
        delta = off_val - csv_val
        denom = max(abs(csv_val), abs(off_val), 1e-9)
        pct = 100 * delta / denom
        flag = ""
        if abs(delta) > max(1e-6, 0.001 * max(abs(csv_val), abs(off_val))):
            flag = " !!"
            mismatches += 1
        print(f"{fn:<35} {csv_val:>12.4f} {off_val:>15.4f} {delta:>+10.4f} {pct:>+7.2f}%{flag}")

    print("-" * 85)
    print(f"Mismatches (|delta| > 0.1% relative): {mismatches}/{len(FEATURE_NAMES)}")

    # Also run XGB on both and compare ml_p
    import xgboost as xgb
    booster = xgb.Booster()
    model_suffix = args.model_suffix
    booster.load_model(str(PROJECT_ROOT / "models" / f"{asset}{model_suffix}_xgb.json"))
    fn_path = PROJECT_ROOT / "models" / f"{asset}{model_suffix}_xgb_features.json"
    if fn_path.exists():
        with open(fn_path) as f:
            xgb_feats = [x for x in json.load(f).get("features", list(FEATURE_NAMES)) if x != "lstm_p"]
    else:
        xgb_feats = list(FEATURE_NAMES)

    csv_vec = np.array([[float(training_row.get(fn, 0.0)) for fn in xgb_feats]], dtype=np.float32)
    off_vec = np.array([[offline_feats.get(fn, 0.0) for fn in xgb_feats]], dtype=np.float32)
    dcsv = xgb.DMatrix(csv_vec, feature_names=xgb_feats)
    doff = xgb.DMatrix(off_vec, feature_names=xgb_feats)
    p_csv = float(booster.predict(dcsv)[0])
    p_off = float(booster.predict(doff)[0])
    print(f"\nXGB prediction:")
    print(f"  from training_csv features: {p_csv:.4f}")
    print(f"  from offline recompute:     {p_off:.4f}")
    print(f"  delta:                      {p_off - p_csv:+.4f}")


if __name__ == "__main__":
    main()
