"""Parity test: TickWindowSlicer must reproduce extract_lstm_sequence input
byte-for-byte vs a direct call with the same tick list.

This is the Phase 2 gate before migrating any caller to the slicer. If the
slicer produces a different sequence than the current direct code path, the
redesign silently changes feature distributions and we would have no idea
downstream.

Usage:
    python scripts/diag_slicer_parity.py \
        --dump-file archive/state_reset_20260424_083945/logs/lstm_tick_dumps/XRP_2026-04-24.jsonl \
        --inference-ts 2026-04-24T14:12:06
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.tick_window_slicer import TickWindowSlicer
from ml.lstm_features import extract_lstm_sequence, LSTM_SEQ_LEN


def load_dump_entry(path: Path, ts_prefix: str) -> dict:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("inference_ts", "").startswith(ts_prefix):
                return r
    raise ValueError(f"No entry matching {ts_prefix} in {path}")


def reconstruct_ticks(rec: dict) -> list[dict]:
    out = []
    for t in rec["ticks"]:
        out.append({
            "ts": datetime.fromisoformat(t["ts"]),
            "price": float(t["price"]),
            "qty": float(t["qty"]),
            "is_buyer": bool(t["is_buyer"]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-file", required=True)
    ap.add_argument("--inference-ts", required=True)
    args = ap.parse_args()

    rec = load_dump_entry(Path(args.dump_file), args.inference_ts)
    ticks = reconstruct_ticks(rec)
    anchor = datetime.fromisoformat(rec["inference_ts"])
    dm = rec["dm"]
    wop = rec.get("window_open_price")

    print(f"[dump] asset={rec['asset']} inference_ts={rec['inference_ts']}")
    print(f"[dump] dm={dm} wop={wop} ticks={len(ticks)}")
    print(f"[dump] live_p_bullish={rec['p_bullish']:.6f}")

    # Path A: current direct-call — feed tick list to extract_lstm_sequence
    seq_direct = extract_lstm_sequence(
        ticks, anchor, decision_minute=dm, window_open_price=wop,
    )
    assert seq_direct is not None, "direct extract returned None"
    md5_direct = hashlib.md5(seq_direct.tobytes()).hexdigest()
    print(f"[direct] shape={seq_direct.shape} dtype={seq_direct.dtype}")
    print(f"[direct] md5={md5_direct}")

    # Path B: new TickWindowSlicer — put ticks in under a source, extract
    # time-window, then pass to extract_lstm_sequence. Dump file is already
    # the merged+sliced buffer live fed to LSTM, so feeding all of them into
    # one source is equivalent to "live's merged source".
    slicer = TickWindowSlicer()
    slicer.extend("merged", ticks)

    # Lookback must be >= LSTM_SEQ_LEN. Dumped ticks already cover
    # [anchor - 180s, anchor], so any lookback >= 180s retrieves them all.
    # Use 180 to minimize risk of an off-by-one boundary divergence.
    window = slicer.get_merged_window(anchor, LSTM_SEQ_LEN, sources=("merged",))
    print(f"[slicer] window={len(window)} ticks (vs {len(ticks)} in dump)")

    seq_slicer = extract_lstm_sequence(
        window, anchor, decision_minute=dm, window_open_price=wop,
    )
    assert seq_slicer is not None, "slicer extract returned None"
    md5_slicer = hashlib.md5(seq_slicer.tobytes()).hexdigest()
    print(f"[slicer] shape={seq_slicer.shape} dtype={seq_slicer.dtype}")
    print(f"[slicer] md5={md5_slicer}")

    # Also test with longer lookback — 900s. Slicer should still return only
    # what it has (180s from dump) and sequence should match.
    window_900 = slicer.get_merged_window(anchor, 900, sources=("merged",))
    seq_slicer_900 = extract_lstm_sequence(
        window_900, anchor, decision_minute=dm, window_open_price=wop,
    )
    md5_slicer_900 = hashlib.md5(seq_slicer_900.tobytes()).hexdigest()
    print(f"[slicer-900] md5={md5_slicer_900}")

    # Verdict
    print()
    print("=" * 60)
    if md5_direct == md5_slicer == md5_slicer_900:
        print("PARITY: OK — slicer produces identical sequence")
        return 0
    else:
        print("PARITY: DRIFT — slicer changes the LSTM input!")
        print(f"  direct   : {md5_direct}")
        print(f"  slicer   : {md5_slicer}")
        print(f"  slicer-900: {md5_slicer_900}")
        if len(window) != len(ticks):
            print(f"  tick-count delta: direct={len(ticks)} slicer={len(window)}")
            # Dump first few differing ts to show where
            ts_direct = [t["ts"] for t in ticks]
            ts_slicer = [t["ts"] for t in window]
            only_in_direct = set(ts_direct) - set(ts_slicer)
            only_in_slicer = set(ts_slicer) - set(ts_direct)
            print(f"  only in direct: {len(only_in_direct)}")
            print(f"  only in slicer: {len(only_in_slicer)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
