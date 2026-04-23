#!/bin/bash
# End-to-end pipeline: get ETH/SOL/XRP retrained at 2s-grain 60d to match BTC.
#
# Assumes these are already running / done:
#   - BTC XGB CSV at 2s grain -> DONE (ml/training_data/BTC_features.csv)
#   - BTC XGB _align2s model -> DONE (models/BTC_align2s_xgb.json)
#   - BTC LSTM 2s regen -> IN PROGRESS in background (writes BTC_lstm_sequences.npz)
#   - ETH/SOL/XRP XGB 2s regen -> IN PROGRESS in background
#
# Pipeline (all side-effects staged with _align2s suffix; production untouched):
#   1. Wait for ETH/SOL/XRP XGB CSVs to complete
#   2. Train XGB _align2s for ETH/SOL/XRP (sequential, ~5 min each)
#   3. Wait for BTC LSTM regen to complete
#   4. Train BTC LSTM _align2s
#   5. Kick off ETH/SOL/XRP LSTM 2s regens in parallel
#   6. Wait for all LSTM regens
#   7. Train LSTM _align2s for ETH/SOL/XRP (sequential — GPU OOM guard)
#   8. Walk-forward validation per asset (no config side-effects)
#   9. Write report for user review
#
# After this: user reviews walk-forward PnL. If good, user runs ensemble_sweep
# + pnl_sweep manually (those write to config/trading.json). Then promote.

set -e
cd "$(dirname "$0")/.."

ASSETS_OTHER=(ETH SOL XRP)
ASSETS_ALL=(BTC ETH SOL XRP)
REPORT=output/align2s_pipeline_report.txt
STATE=output/align2s_pipeline_state.txt
MIN_XGB_CSV_SIZE=500000000  # 500MB — conservatively under expected ~950MB
MIN_LSTM_NPZ_SIZE=100000000  # 100MB

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() {
  local msg="[$(ts)] $*"
  echo "$msg"
  echo "$msg" >> "$STATE"
}

wait_for_file() {
  # wait_for_file <path> <min_bytes> <human label>
  local path="$1" min="$2" label="$3"
  local last=0
  while true; do
    local sz=0
    if [ -f "$path" ]; then
      sz=$(stat -c%s "$path" 2>/dev/null || echo 0)
    fi
    if [ "$sz" -ge "$min" ]; then
      local m1=$(stat -c%Y "$path" 2>/dev/null || echo 0)
      sleep 15
      local m2=$(stat -c%Y "$path" 2>/dev/null || echo 0)
      if [ "$m1" = "$m2" ]; then
        log "  $label ready ($((sz/1000000))MB)"
        return 0
      fi
    fi
    if [ "$sz" -ne "$last" ]; then
      log "  $label still growing: $((sz/1000000))MB"
      last=$sz
    fi
    sleep 60
  done
}

log "=== ALIGN2S PIPELINE START ==="
: > "$REPORT"
echo "Pipeline started: $(ts)" >> "$REPORT"

# 1. Wait for ETH/SOL/XRP XGB CSVs
log "[1/8] Waiting for ETH/SOL/XRP XGB CSVs..."
for a in "${ASSETS_OTHER[@]}"; do
  wait_for_file "ml/training_data/${a}_features.csv" "$MIN_XGB_CSV_SIZE" "${a}_features.csv"
done
log "[1/8] Done"

# 2. Train XGB _align2s for ETH/SOL/XRP
log "[2/8] Training XGB _align2s for ETH/SOL/XRP..."
for a in "${ASSETS_OTHER[@]}"; do
  log "  training $a XGB..."
  if [ -f "models/${a}_v10_trimmed_features.json" ]; then
    mv "models/${a}_v10_trimmed_features.json" "models/${a}_v10_trimmed_features.json.prev_v10"
  fi
  python scripts/train_xgb.py --asset "$a" --model-suffix _align2s \
    > "output/train_${a}_align2s_xgb.log" 2>&1 || {
    log "  FAILED: $a XGB training"
    echo "FAILED at step 2: $a XGB" >> "$REPORT"
    exit 1
  }
  log "  $a XGB trained"
done
log "[2/8] Done"

# 3. Wait for BTC LSTM regen
log "[3/8] Waiting for BTC LSTM regen..."
wait_for_file "ml/training_data/BTC_lstm_sequences.npz" "$MIN_LSTM_NPZ_SIZE" "BTC_lstm_sequences.npz"
log "[3/8] Done"

# 4. Train BTC LSTM
log "[4/8] Training BTC LSTM _align2s..."
python scripts/train_lstm.py --asset BTC --model-suffix _align2s \
  > output/train_BTC_align2s_lstm.log 2>&1 || {
  log "  FAILED: BTC LSTM training"
  echo "FAILED at step 4: BTC LSTM" >> "$REPORT"
  exit 1
}
log "[4/8] Done"

# 5. Kick off ETH/SOL/XRP LSTM regens in parallel
log "[5/8] Starting ETH/SOL/XRP LSTM regens (parallel)..."
pids=()
for a in "${ASSETS_OTHER[@]}"; do
  python scripts/generate_lstm_training_data.py --asset "$a" --days 60 \
    --check-interval-seconds 2 --min-move 0.0001 \
    > "output/regen_${a}_lstm_2s.log" 2>&1 &
  pids+=($!)
  log "  spawned LSTM-gen $a (pid $!)"
done
for pid in "${pids[@]}"; do
  wait "$pid" || log "  WARNING: LSTM gen pid $pid exited non-zero"
done
log "[5/8] Done"

# 6. Wait for LSTM npzs to be on disk (in case writer is flushing)
log "[6/8] Verifying ETH/SOL/XRP LSTM .npz files..."
for a in "${ASSETS_OTHER[@]}"; do
  wait_for_file "ml/training_data/${a}_lstm_sequences.npz" "$MIN_LSTM_NPZ_SIZE" "${a}_lstm_sequences.npz"
done
log "[6/8] Done"

# 7. Train LSTM _align2s for ETH/SOL/XRP (sequential — GPU OOM guard)
log "[7/8] Training LSTM _align2s for ETH/SOL/XRP (sequential)..."
for a in "${ASSETS_OTHER[@]}"; do
  log "  training $a LSTM..."
  python scripts/train_lstm.py --asset "$a" --model-suffix _align2s \
    > "output/train_${a}_align2s_lstm.log" 2>&1 || {
    log "  FAILED: $a LSTM training"
    echo "FAILED at step 7: $a LSTM" >> "$REPORT"
    exit 1
  }
  log "  $a LSTM trained"
done
log "[7/8] Done"

# 8. Walk-forward validation per asset (no config side-effects)
log "[8/8] Walk-forward validation for all 4 assets..."
for a in "${ASSETS_ALL[@]}"; do
  log "  walk-forward $a..."
  python scripts/walk_forward_full.py --asset "$a" --model-suffix _align2s \
    --holdout-days 7 --min-dm 2 \
    > "output/walkfwd_${a}_align2s.log" 2>&1 || {
    log "  WARNING: $a walk-forward failed — continuing"
  }
done
log "[8/8] Done"

# Final report
log "Writing final report..."
{
  echo ""
  echo "=== ALIGN2S PIPELINE COMPLETE ==="
  echo "Finished: $(ts)"
  echo ""
  echo "--- Staging models (safe to review; production untouched) ---"
  ls -la models/*_align2s_xgb.json models/*_align2s_lstm.pt 2>/dev/null
  echo ""
  echo "--- Training data row counts ---"
  for a in BTC ETH SOL XRP; do
    if [ -f "ml/training_data/${a}_features.csv" ]; then
      echo "$a XGB CSV: $(wc -l < ml/training_data/${a}_features.csv) rows"
    fi
    if [ -f "ml/training_data/${a}_lstm_sequences.npz" ]; then
      echo "$a LSTM npz: $(stat -c%s ml/training_data/${a}_lstm_sequences.npz) bytes"
    fi
  done
  echo ""
  echo "--- Walk-forward summaries ---"
  for a in "${ASSETS_ALL[@]}"; do
    if [ -f "output/walkfwd_${a}_align2s.log" ]; then
      echo ""
      echo ">>> $a"
      grep -E "^WINNER|holdout|fold [0-9]|PnL|wr" "output/walkfwd_${a}_align2s.log" 2>/dev/null | head -20
    fi
  done
  echo ""
  echo "--- Next steps (user review) ---"
  echo "  1. Review walk-forward PnL per asset above"
  echo "  2. If PnL holds / improves vs current production:"
  echo "     a. Run ensemble_sweep.py and pnl_sweep.py to find new dynamic_k / threshold"
  echo "        (these WILL overwrite config/trading.json ensemble block — backup first)"
  echo "     b. Run scripts/promote_retrain_models.sh align2s to rotate staging -> prod"
  echo "     c. Restart live bot to hot-reload new models"
  echo "  3. If PnL regresses, stay on current production and investigate"
} >> "$REPORT"

log "=== PIPELINE DONE — see $REPORT ==="
