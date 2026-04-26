#!/bin/bash
# Overnight retrain orchestrator — wait for regens, then train, then sweep.
#
# Step order:
#   1. Wait until all 5 regens (BTC/ETH/SOL/XRP/DOGE) produce {ASSET}_lstm_X.npy
#      + {ASSET}_lstm_meta.npz, with NO {ASSET}_lstm_X.tmp.npy left over.
#      HYPE/BNB are already done with legacy npz format — not waited for.
#   2. Sanity-check that BTC/ETH/SOL/XRP have their _features.csv
#      (DOGE used --outputs lstm so its CSV is from the earlier successful run).
#   3. Run scripts/train_all_5fold.py — 35 models (7 assets × 5: 4 fold-exclusion + 1 all-data).
#   4. Run scripts/retrain_4fold.py — sweep folds 1..N-1 + OOS on fold N, write report.
#
# On any failure, the chain exits with the failed step's exit code. Sweep won't
# run if training failed; training won't run if regens are missing.
#
# Logs:
#   output/retrain_20260424/overnight_chain.log   (this script's output)
#   output/retrain_20260424/train_all_5fold.log    (training stage)
#   output/retrain_20260424/retrain_4fold.log      (sweep stage)
#   output/retrain_4fold/report_v14.txt            (final OOS report)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG="output/retrain_20260424/overnight_chain.log"
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

echo ""
echo "================================================================"
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Overnight chain started"
echo "================================================================"

ANCHOR="2026-04-24"
ASSETS="BTC,ETH,SOL,XRP,HYPE,BNB,DOGE"

# Step 1: wait for regen completion.
NEED_ASSETS=("BTC" "ETH" "SOL" "XRP" "DOGE")
MAX_WAIT_MIN=600  # 10 hours
WAIT_INTERVAL=300  # 5 min between checks
elapsed=0

while true; do
  done=0
  status=""
  for a in "${NEED_ASSETS[@]}"; do
    x="ml/training_data/${a}_lstm_X.npy"
    m="ml/training_data/${a}_lstm_meta.npz"
    tmp="ml/training_data/${a}_lstm_X.tmp.npy"
    if [ -f "$x" ] && [ -f "$m" ] && [ ! -f "$tmp" ]; then
      done=$((done + 1))
      status="$status $a:OK"
    elif [ -f "$tmp" ]; then
      status="$status $a:RUNNING"
    else
      status="$status $a:WAITING"
    fi
  done
  echo "[$(date -u '+%H:%M:%S')] Regen progress: $done/5 done.$status"
  if [ "$done" -ge 5 ]; then
    echo "[$(date -u '+%H:%M:%S')] All 5 regens done — proceeding."
    break
  fi
  if [ "$elapsed" -ge "$MAX_WAIT_MIN" ]; then
    echo "ERROR: regens did not finish within ${MAX_WAIT_MIN} min — aborting."
    exit 2
  fi
  sleep "$WAIT_INTERVAL"
  elapsed=$((elapsed + WAIT_INTERVAL / 60))
done

# Step 2: verify required input files
echo ""
echo "[$(date -u '+%H:%M:%S')] Verifying input files..."
for a in BTC ETH SOL XRP; do
  csv="ml/training_data/${a}_features.csv"
  if [ ! -f "$csv" ]; then
    echo "ERROR: missing $csv (XGB CSV) — aborting."
    exit 3
  fi
  size=$(stat -c%s "$csv" 2>/dev/null || stat -f%z "$csv" 2>/dev/null || echo 0)
  if [ "$size" -lt 1000000 ]; then
    echo "ERROR: $csv is suspiciously small ($size bytes) — aborting."
    exit 3
  fi
done
for a in BTC ETH SOL XRP HYPE BNB DOGE; do
  # LSTM data: either new split format or legacy npz
  new_x="ml/training_data/${a}_lstm_X.npy"
  legacy="ml/training_data/${a}_lstm_sequences.npz"
  if [ ! -f "$new_x" ] && [ ! -f "$legacy" ]; then
    echo "ERROR: missing LSTM data for $a (neither $new_x nor $legacy) — aborting."
    exit 3
  fi
done
echo "[$(date -u '+%H:%M:%S')] All input files verified."

# Step 3: train 35 models
echo ""
echo "================================================================"
echo "[$(date -u '+%H:%M:%S')] STAGE 2: train_all_5fold.py starting"
echo "================================================================"
TRAIN_LOG="output/retrain_20260424/train_all_5fold.log"
python -u scripts/train_all_5fold.py \
  --assets "$ASSETS" \
  --end-date "$ANCHOR" \
  --model-suffix _v14 \
  --xgb-parallel 2 > "$TRAIN_LOG" 2>&1

if [ $? -ne 0 ]; then
  echo "ERROR: train_all_5fold.py failed — see $TRAIN_LOG"
  exit 4
fi
echo "[$(date -u '+%H:%M:%S')] Training complete."

# Step 4: sweep + OOS
echo ""
echo "================================================================"
echo "[$(date -u '+%H:%M:%S')] STAGE 3: retrain_4fold.py starting"
echo "================================================================"
SWEEP_LOG="output/retrain_20260424/retrain_4fold.log"
python -u scripts/retrain_4fold.py \
  --assets "$ASSETS" \
  --end-date "$ANCHOR" \
  --model-suffix _v14 > "$SWEEP_LOG" 2>&1

if [ $? -ne 0 ]; then
  echo "ERROR: retrain_4fold.py failed — see $SWEEP_LOG"
  exit 5
fi

echo ""
echo "================================================================"
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] OVERNIGHT CHAIN COMPLETE"
echo "================================================================"
echo "Final report: output/retrain_4fold/report_v14.txt"
