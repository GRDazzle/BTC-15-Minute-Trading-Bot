"""End-to-end alignment audit.

For every LSTM dump record in today's session, reconstruct the 900s tick
window from the on-disk aggtrade CSVs (the same files replay / sweep /
training read) and compare to what live actually saw.

Three measures per record, roughly the three alignment axes:

  1. **Tick-set match**: count + set-of-ticks identical between live's
     dumped `ticks` field and the CSV reconstruction. Catches WS-arrival
     ordering drift / timing gaps between what live buffered and what
     eventually landed on disk.

  2. **Pre-scale md5 match**: run `extract_lstm_sequence` on the CSV
     reconstruction at the same `inference_ts`; md5 the result; compare to
     live's dumped `pre_scale_md5`. Catches extractor semantic drift — if
     any step from tick list → 180s sequence differs between live and
     backtest paths, this fails.

  3. **Post-scale md5 match**: scale the sequence via the LSTM model's
     scaler; md5; compare to live's dumped `sequence_md5`. Catches scaler
     init / mutation edge cases.

A PASS on all three means live ↔ replay / sweep / training are aligned
down to the byte for at least this slice of time. Multiple PASSes across
assets and dms are stronger evidence than any code inspection.

Usage:
    python scripts/audit_alignment.py
    python scripts/audit_alignment.py --max-per-asset 20 --asset BTC,SOL
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger as _l
_l.remove()
_l.add(sys.stderr, level="WARNING")

from core.tick_window_slicer import TickWindowSlicer
from ml.lstm_features import extract_lstm_sequence, LSTM_SEQ_LEN
from ml.lstm_model import load_model

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
KRAKEN_DIR = PROJECT_ROOT / "data" / "aggtrades_kraken"
DUMP_DIR = PROJECT_ROOT / "logs" / "lstm_tick_dumps"
MODEL_DIR = PROJECT_ROOT / "models"


def _load_csv_ticks(base: Path, asset: str, sym: str, dates: list[str]) -> list[dict]:
    """Load ticks from aggtrade CSV files for given date strings."""
    out = []
    for d in dates:
        path = base / asset / f"{asset}{sym}-aggTrades-{d}.csv"
        if not path.exists():
            continue
        try:
            with open(path, newline="") as f:
                for row in csv.reader(f):
                    if len(row) < 7:
                        continue
                    try:
                        ts_raw = int(row[5])
                    except (ValueError, IndexError):
                        continue
                    # Microsecond since Jan 2025 data; ms for older data
                    if ts_raw > 1_000_000_000_000_000:
                        ts = datetime.fromtimestamp(ts_raw / 1_000_000, tz=timezone.utc)
                    else:
                        ts = datetime.fromtimestamp(ts_raw / 1_000, tz=timezone.utc)
                    price = float(row[1])
                    qty = float(row[2])
                    is_buyer_maker = row[6].strip().lower() in ("true", "1")
                    out.append({
                        "ts": ts,
                        "price": price,
                        "qty": qty,
                        "is_buyer": not is_buyer_maker,  # invert maker flag
                    })
        except Exception as e:
            print(f"  [WARN] error reading {path}: {e}")
    out.sort(key=lambda t: t["ts"])
    return out


def _tick_signature(tick: dict) -> tuple:
    """Hashable tick identity for set-based comparison."""
    ts = tick["ts"]
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (
        ts.isoformat(),
        round(float(tick["price"]), 10),
        round(float(tick["qty"]), 10),
        bool(tick["is_buyer"]),
    )


def _parse_dumped_tick(t: dict) -> dict:
    return {
        "ts": datetime.fromisoformat(t["ts"]),
        "price": float(t["price"]),
        "qty": float(t["qty"]),
        "is_buyer": bool(t["is_buyer"]),
    }


def audit_record(rec: dict, csv_ticks_cb: list[dict], csv_ticks_kr: list[dict],
                 scaler_mean, scaler_std) -> dict:
    """Run the three audit checks for one dump record."""
    asset = rec["asset"]
    anchor = datetime.fromisoformat(rec["inference_ts"])
    dm = rec["dm"]
    wop = rec.get("window_open_price")
    live_pre = rec.get("pre_scale_md5")
    live_post = rec.get("sequence_md5")

    # Live ticks as dumped (the exact 180s slice live fed to extract_lstm_sequence)
    live_ticks = [_parse_dumped_tick(t) for t in rec["ticks"]]
    live_ts_set = {_tick_signature(t) for t in live_ticks}

    # CSV-reconstructed 180s window from the same sources live ingests
    slicer = TickWindowSlicer()
    if csv_ticks_cb:
        slicer.extend("coinbase", csv_ticks_cb)
    if csv_ticks_kr:
        slicer.extend("kraken", csv_ticks_kr)
    csv_merged = slicer.get_merged_window(anchor, LSTM_SEQ_LEN, ("coinbase", "kraken"))
    csv_ts_set = {_tick_signature(t) for t in csv_merged}

    # 1. Tick-set match
    only_in_live = live_ts_set - csv_ts_set
    only_in_csv = csv_ts_set - live_ts_set
    tick_match = (not only_in_live) and (not only_in_csv)

    # 2. Pre-scale md5 match (run extract_lstm_sequence on CSV reconstruction)
    seq_csv = extract_lstm_sequence(
        csv_merged, anchor, decision_minute=dm,
        seq_len=LSTM_SEQ_LEN, window_open_price=wop,
    )
    pre_csv_md5 = None
    post_csv_md5 = None
    if seq_csv is not None:
        pre_csv_md5 = hashlib.md5(seq_csv.tobytes()).hexdigest()
        # 3. Post-scale md5 (apply scaler)
        seq_scaled = (seq_csv - scaler_mean) / scaler_std
        seq_scaled = np.nan_to_num(seq_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        post_csv_md5 = hashlib.md5(seq_scaled.tobytes()).hexdigest()

    pre_match = (live_pre is not None) and (pre_csv_md5 == live_pre)
    post_match = (live_post is not None) and (post_csv_md5 == live_post)

    return {
        "asset": asset,
        "inference_ts": rec["inference_ts"],
        "dm": dm,
        "live_tick_count": len(live_ticks),
        "csv_tick_count": len(csv_merged),
        "only_in_live": len(only_in_live),
        "only_in_csv": len(only_in_csv),
        "tick_match": tick_match,
        "pre_match": pre_match,
        "post_match": post_match,
        "live_pre_md5": (live_pre or "")[:12],
        "csv_pre_md5": (pre_csv_md5 or "")[:12],
        "live_post_md5": (live_post or "")[:12],
        "csv_post_md5": (post_csv_md5 or "")[:12],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC,ETH,SOL,XRP,HYPE,BNB,DOGE",
                    help="Comma-separated asset list")
    ap.add_argument("--max-per-asset", type=int, default=10,
                    help="Audit up to N dump records per asset (sampled across the day)")
    args = ap.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",") if a.strip()]
    all_results = []

    for asset in assets:
        dump_paths = sorted(DUMP_DIR.glob(f"{asset}_*.jsonl"))
        if not dump_paths:
            print(f"[{asset}] no dump files — skipping")
            continue

        # Load all records from all dump files
        records = []
        for dp in dump_paths:
            with open(dp) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("pre_scale_md5") is None:
                        continue  # old-format record, skip
                    records.append(r)
        if not records:
            print(f"[{asset}] no records with pre_scale_md5 — skipping")
            continue

        # Sample N records evenly across the day
        if len(records) > args.max_per_asset:
            step = len(records) // args.max_per_asset
            records = records[::step][:args.max_per_asset]
        print(f"[{asset}] auditing {len(records)} records")

        # Collect dates spanned by the anchor timestamps (+1 day for buffer)
        dates: set[str] = set()
        for r in records:
            anchor = datetime.fromisoformat(r["inference_ts"])
            for delta in (-1, 0):
                dates.add((anchor + timedelta(days=delta)).strftime("%Y-%m-%d"))
        date_list = sorted(dates)

        # Load CSVs once for this asset's date range
        cb_all = _load_csv_ticks(DATA_DIR, asset, "USDT", date_list)
        kr_all = _load_csv_ticks(KRAKEN_DIR, asset, "USD", date_list)
        print(f"  loaded {len(cb_all)} CB + {len(kr_all)} KR ticks from CSVs")

        # Load scaler from the model live is using (standard suffix)
        model_path = MODEL_DIR / f"{asset}_lstm.pt"
        if not model_path.exists():
            print(f"  [{asset}] model file missing; skipping")
            continue
        _, meta = load_model(str(model_path))
        scaler_mean = np.array(meta["scaler_mean"], dtype=np.float32)
        scaler_std = np.array(meta["scaler_std"], dtype=np.float32)
        scaler_std[scaler_std == 0] = 1.0

        # Slice CSV ticks per record to match live's 180s window
        for rec in records:
            anchor = datetime.fromisoformat(rec["inference_ts"])
            lo = anchor - timedelta(seconds=LSTM_SEQ_LEN)
            # Crude linear filter — CSVs are sorted so this is O(n)
            cb_slice = [t for t in cb_all if lo <= t["ts"] <= anchor]
            kr_slice = [t for t in kr_all if lo <= t["ts"] <= anchor]
            result = audit_record(rec, cb_slice, kr_slice, scaler_mean, scaler_std)
            all_results.append(result)

    # Summary
    print()
    print("=" * 96)
    print(f"{'asset':<6} {'dm':>3} {'ts':<27} {'live#':>6} {'csv#':>6} "
          f"{'+live':>5} {'+csv':>5} {'tick':>5} {'pre':>4} {'post':>5}  live_pre->csv_pre")
    print("-" * 96)
    pass_tick = pass_pre = pass_post = 0
    for r in all_results:
        print(f"{r['asset']:<6} {r['dm']:>3} {r['inference_ts'][:26]:<27} "
              f"{r['live_tick_count']:>6} {r['csv_tick_count']:>6} "
              f"{r['only_in_live']:>5} {r['only_in_csv']:>5} "
              f"{'PASS' if r['tick_match'] else 'FAIL':>5} "
              f"{'OK' if r['pre_match'] else 'NO':>4} "
              f"{'OK' if r['post_match'] else 'NO':>5}  "
              f"{r['live_pre_md5']} -> {r['csv_pre_md5']}")
        if r["tick_match"]:
            pass_tick += 1
        if r["pre_match"]:
            pass_pre += 1
        if r["post_match"]:
            pass_post += 1
    total = len(all_results) or 1
    print("-" * 96)
    print(f"TOTAL: {total} records audited")
    print(f"  tick-set match      : {pass_tick}/{total}  ({100.0*pass_tick/total:.1f}%)")
    print(f"  pre-scale md5 match : {pass_pre}/{total}  ({100.0*pass_pre/total:.1f}%)")
    print(f"  post-scale md5 match: {pass_post}/{total}  ({100.0*pass_post/total:.1f}%)")


if __name__ == "__main__":
    main()
