#!/bin/bash
# Full retrain pipeline with per-asset parallel data generation.

set -e
cd "$(dirname "$0")/.."

DATE_TAG=$(date -u +%Y%m%d)
DAYS=60
ASSETS=(BTC ETH SOL XRP)

log() {
  echo "[$(date -u +%H:%M:%S)] $*"
}

log "=== FULL RETRAIN START (tag=${DATE_TAG}, days=${DAYS}) ==="

log "[1/6] Regenerating XGB training data (4 parallel)..."
pids=()
for asset in "${ASSETS[@]}"; do
  python scripts/generate_training_data.py --asset "$asset" --days "$DAYS" \
    > "output/retrain_gen_xgb_${asset}.log" 2>&1 &
  pids+=($!)
  log "  spawned XGB-gen $asset (pid $!)"
done
for pid in "${pids[@]}"; do
  wait "$pid" || log "  WARNING: XGB data gen pid $pid failed"
done
log "[1/6] Done"

log "[2/6] Regenerating LSTM sequences (4 parallel)..."
pids=()
for asset in "${ASSETS[@]}"; do
  python scripts/generate_lstm_training_data.py --asset "$asset" --days "$DAYS" \
    > "output/retrain_gen_lstm_${asset}.log" 2>&1 &
  pids+=($!)
  log "  spawned LSTM-gen $asset (pid $!)"
done
for pid in "${pids[@]}"; do
  wait "$pid" || log "  WARNING: LSTM data gen pid $pid failed"
done
log "[2/6] Done"

log "[3/6] Retraining LSTM per asset (sequential, GPU shared)..."
for asset in "${ASSETS[@]}"; do
  log "  training LSTM $asset..."
  python scripts/train_lstm.py --asset "$asset" --min-dm 2 \
    --model-suffix "_retrain_${DATE_TAG}" \
    > "output/retrain_lstm_${asset}.log" 2>&1 \
    || log "  WARNING: LSTM train $asset failed"
done
log "[3/6] Done"

# Walk-forward steps — GPU is shared, can't parallelize GPU jobs safely.
# Run sequentially but each uses GPU fully.

log "[4/6] Walk-forward XGB: BTC @ dm=7 (GPU)..."
python scripts/walk_forward_full.py --asset BTC --min-dm 7 --n-folds 3 \
  --holdout-days 10 --staging "retrain_btc_dm7_${DATE_TAG}" \
  > output/retrain_wf_btc.log 2>&1
log "[4/6] Done"

log "[5/6] Walk-forward XGB: ETH,SOL,XRP @ dm=4 (GPU)..."
python scripts/walk_forward_full.py --asset ETH,SOL,XRP --min-dm 4 --n-folds 3 \
  --holdout-days 10 --staging "retrain_esx_dm4_${DATE_TAG}" \
  > output/retrain_wf_esx.log 2>&1
log "[5/6] Done"

log "[6/6] SAFE MODE: NOT promoting to live. Models stay in backup paths."
log ""
log "  New LSTM models (backup):"
for asset in "${ASSETS[@]}"; do
  path="models/${asset}_retrain_${DATE_TAG}_lstm.pt"
  if [ -f "$path" ]; then
    log "    - $path"
  else
    log "    MISSING - $path"
  fi
done
log ""
log "  New XGB models (staging):"
for variant in "" "_weekday" "_weekend"; do
  path="models/staging_retrain_btc_dm7_${DATE_TAG}/BTC${variant}_xgb.json"
  [ -f "$path" ] && log "    - $path"
done
for asset in ETH SOL XRP; do
  for variant in "" "_weekday" "_weekend"; do
    path="models/staging_retrain_esx_dm4_${DATE_TAG}/${asset}${variant}_xgb.json"
    [ -f "$path" ] && log "    - $path"
  done
done
log ""
log "=== FULL RETRAIN COMPLETE (NOT PROMOTED) ==="
log "Live session UNAFFECTED. Evaluate live performance first, then manually promote if desired:"
log "  bash scripts/promote_retrain_models.sh ${DATE_TAG}"
