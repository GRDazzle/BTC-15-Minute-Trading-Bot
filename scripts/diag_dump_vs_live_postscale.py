"""Compare live's dumped sequence_md5 (post-scale) to standalone's recomputed
post-scale md5 on the exact same dumped ticks.

This isolates whether the live-vs-standalone gap is in:
  - tick reconstruction from dump (pre-scale md5 mismatch)
  - scaling (pre-scale matches, post-scale doesn't)
  - LSTM forward pass (both md5s match, p differs -- pure torch state bug)

Usage:
    python scripts/diag_dump_vs_live_postscale.py \
        --dump-file logs/lstm_tick_dumps/SOL_2026-04-24.jsonl \
        --inference-ts "2026-04-24T15:57:00" \
        --model-suffix _align2s
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger as _l
_l.remove()
_l.add(sys.stderr, level="WARNING")

from ml.lstm_features import extract_lstm_sequence
from ml.lstm_model import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-file", required=True)
    ap.add_argument("--inference-ts", required=True)
    ap.add_argument("--model-suffix", default="")
    args = ap.parse_args()

    # Load the dump record
    rec = None
    with open(args.dump_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("inference_ts", "").startswith(args.inference_ts):
                rec = r
                break
    if rec is None:
        print(f"No entry matching {args.inference_ts}")
        return 1

    asset = rec["asset"]
    anchor = datetime.fromisoformat(rec["inference_ts"])
    dm = rec["dm"]
    wop = rec.get("window_open_price")
    live_seq_md5 = rec.get("sequence_md5", "MISSING")
    live_p = rec["p_bullish"]

    print(f"[dump] asset={asset} anchor={anchor}")
    print(f"[dump] dm={dm} wop={wop} ticks={len(rec['ticks'])}")
    print(f"[dump] live sequence_md5 (post-scale) = {live_seq_md5}")
    print(f"[dump] live p_bullish                 = {live_p:.6f}")

    # Reconstruct ticks
    ticks = [{
        "ts": datetime.fromisoformat(t["ts"]),
        "price": float(t["price"]),
        "qty": float(t["qty"]),
        "is_buyer": bool(t["is_buyer"]),
    } for t in rec["ticks"]]

    # Extract sequence (pre-scale)
    seq_pre = extract_lstm_sequence(ticks, anchor, decision_minute=dm, window_open_price=wop)
    if seq_pre is None:
        print("extract returned None")
        return 1
    pre_md5 = hashlib.md5(seq_pre.tobytes()).hexdigest()
    print(f"[standalone] pre-scale md5             = {pre_md5}")

    # Load model + scaler from the same file live uses
    model_path = PROJECT_ROOT / "models" / f"{asset}{args.model_suffix}_lstm.pt"
    print(f"[standalone] loading model {model_path}")
    model, meta = load_model(str(model_path))
    model.eval()

    scaler_mean = np.array(meta["scaler_mean"], dtype=np.float32)
    scaler_std = np.array(meta["scaler_std"], dtype=np.float32)
    scaler_std[scaler_std == 0] = 1.0

    # Apply scaler (same as LSTMProcessor.predict_proba)
    seq_scaled = (seq_pre - scaler_mean) / scaler_std
    seq_scaled = np.nan_to_num(seq_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
    post_md5 = hashlib.md5(seq_scaled.tobytes()).hexdigest()
    print(f"[standalone] post-scale md5            = {post_md5}")

    # Run forward pass
    with torch.no_grad():
        x = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)
        standalone_p = model(x).item()
    print(f"[standalone] p_bullish                 = {standalone_p:.6f}")

    # Verdict
    print()
    print("=" * 60)
    if post_md5 == live_seq_md5 and abs(standalone_p - live_p) < 1e-4:
        print("FULL MATCH: sequences and p_bullish identical. Live ==  standalone.")
        return 0
    elif post_md5 == live_seq_md5:
        print("SEQUENCE MATCH but p_bullish DIFFERS.")
        print("  -> Model forward pass is non-reproducible between processes.")
        print("  -> Hunt for torch / cuDNN / model state differences between live's")
        print("     process and standalone's.")
        return 2
    else:
        print("SEQUENCE MISMATCH.")
        print(f"  live     post-scale md5 = {live_seq_md5}")
        print(f"  standalone post-scale md5 = {post_md5}")
        print("  -> Sequences differ — the dumped ticks do not fully reconstruct")
        print("     live's LSTM input. Look for:")
        print("     (a) scaler values differ (numpy array copy semantics)")
        print("     (b) tick-field precision loss in JSON roundtrip")
        print("     (c) extract_lstm_sequence invariant broken (e.g., ts tzinfo,")
        print("         sort order, or mutation between extract and dump)")
        return 3


if __name__ == "__main__":
    sys.exit(main())
