"""Load a live LSTM tick dump and replay LSTM inference on the exact tick data.

If result matches live's dumped p_bullish, the model + extract_sequence pipeline
is deterministic. Any discrepancy in earlier replay tests must be due to different
tick buffer reconstruction, not the LSTM model or inference code.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger as _l
_l.remove()
_l.add(sys.stderr, level="WARNING")

from core.strategy_brain.signal_processors.lstm_processor import LSTMProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-file", required=True)
    ap.add_argument("--inference-ts", required=True,
                    help="ISO ts matching an entry's inference_ts")
    ap.add_argument("--model-suffix", default="")
    args = ap.parse_args()

    # Find the matching dump entry
    rec = None
    with open(args.dump_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("inference_ts", "").startswith(args.inference_ts):
                rec = r
                break
    if not rec:
        print(f"No entry matching {args.inference_ts}")
        return
    print(f"Found entry: inference_ts={rec['inference_ts']}")
    print(f"  live dumped p_bullish = {rec['p_bullish']:.4f}")
    print(f"  asset={rec['asset']} dm={rec['dm']} WOP={rec.get('window_open_price')}")
    print(f"  ticks={len(rec['ticks'])}")

    # Reconstruct the exact tick list live fed to LSTM
    ticks = []
    for t in rec["ticks"]:
        ticks.append({
            "ts": datetime.fromisoformat(t["ts"]),
            "price": float(t["price"]),
            "qty": float(t["qty"]),
            "is_buyer": bool(t["is_buyer"]),
        })

    # Load LSTM with same suffix
    lstm_proc = LSTMProcessor(
        asset=rec["asset"],
        model_dir=PROJECT_ROOT / "models",
        model_suffix=args.model_suffix,
    )

    # Build metadata that matches live's call
    metadata = {
        "raw_tick_buffer": ticks,
        "decision_minute": rec["dm"],
        "window_open_price": rec.get("window_open_price"),
    }

    # Also directly extract the sequence to compare what the LSTM sees
    from ml.lstm_features import extract_lstm_sequence
    ts_anchor = ticks[-1]["ts"]
    seq = extract_lstm_sequence(ticks, ts_anchor,
                                 decision_minute=rec["dm"],
                                 window_open_price=rec.get("window_open_price"))
    if seq is not None:
        import hashlib
        seq_hash = hashlib.md5(seq.tobytes()).hexdigest()
        print(f"\n=== Extracted LSTM sequence ===")
        print(f"  shape={seq.shape}, dtype={seq.dtype}")
        print(f"  md5={seq_hash}")
        print(f"  first bar (t-179s): {seq[0].tolist()[:5]} ...")
        print(f"  last bar  (t=0):    {seq[-1].tolist()[:5]} ...")
        print(f"  nonzero bars (bar.price != 0): {int((seq[:,0] != 0).sum())} of {len(seq)}")

    # Call predict_proba — should match live's dumped p_bullish if deterministic
    current_price = ticks[-1]["price"]
    p = lstm_proc.predict_proba(
        Decimal(str(current_price)),
        [t["price"] for t in ticks],
        metadata,
    )
    print(f"\n=== Result from reloading live's exact tick input ===")
    print(f"  replay lstm p_bullish = {p:.4f}")
    print(f"  LIVE   lstm p_bullish = {rec['p_bullish']:.4f}")
    print(f"  delta = {abs(p - rec['p_bullish']):.4f}")
    if abs(p - rec["p_bullish"]) < 0.001:
        print("  -> IDENTICAL (model is deterministic given same input)")
    else:
        print("  -> MISMATCH (model outputs differ on same input — investigate)")


if __name__ == "__main__":
    main()
