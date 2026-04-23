#!/bin/bash
# Stage B: resumes the align2s pipeline after stage A's failures.
#
# Precondition (all already running / done):
#   - XGB _align2s models for BTC, ETH, SOL, XRP -> DONE
#   - BTC LSTM regen DONE
#   - BTC LSTM training RUNNING (background) — we'll wait for it
#   - ETH/SOL/XRP LSTM regens RUNNING (background) — we'll wait for them
#
# This script:
#   1. Waits for BTC _align2s LSTM model to appear on disk
#   2. Waits for ETH/SOL/XRP LSTM .npz files
#   3. Trains ETH/SOL/XRP LSTM _align2s sequentially (GPU OOM guard)
#   4. Walk-forward all 4 assets with _align2s tag
#   5. Writes final report to output/align2s_pipeline_report.txt

set -e
cd "$(dirname "$0")/.."

ASSETS_OTHER=(ETH SOL XRP)
ASSETS_ALL=(BTC ETH SOL XRP)
REPORT=output/align2s_pipeline_report.txt
STATE=output/align2s_stage_b_state.txt
MIN_LSTM_NPZ_SIZE=100000000
MIN_LSTM_PT_SIZE=100000  # LSTM .pt is small (~1-2MB)

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() {
  local msg="[$(ts)] $*"
  echo "$msg"
  echo "$msg" >> "$STATE"
}

wait_for_file() {
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
    sleep 120
  done
}

log "=== STAGE B START ==="

# 1. Wait for BTC LSTM training to finish
log "[1/4] Waiting for BTC _align2s LSTM model..."
wait_for_file "models/BTC_align2s_lstm.pt" "$MIN_LSTM_PT_SIZE" "BTC_align2s_lstm.pt"
log "[1/4] Done"

# 2. Wait for ETH/SOL/XRP LSTM regens
log "[2/4] Waiting for ETH/SOL/XRP LSTM sequences..."
for a in "${ASSETS_OTHER[@]}"; do
  wait_for_file "ml/training_data/${a}_lstm_sequences.npz" "$MIN_LSTM_NPZ_SIZE" "${a}_lstm_sequences.npz"
done
log "[2/4] Done"

# 3. Train LSTM for ETH/SOL/XRP (sequential — GPU OOM guard)
log "[3/4] Training ETH/SOL/XRP LSTM _align2s (sequential)..."
for a in "${ASSETS_OTHER[@]}"; do
  log "  training $a LSTM..."
  python scripts/train_lstm.py --asset "$a" --model-suffix _align2s \
    > "output/train_${a}_align2s_lstm.log" 2>&1 || {
    log "  FAILED: $a LSTM training — see output/train_${a}_align2s_lstm.log"
    echo "STAGE B FAILED: $a LSTM" >> "$REPORT"
    # Continue to next asset rather than abort whole pipeline
  }
  if [ -f "models/${a}_align2s_lstm.pt" ]; then
    log "  $a LSTM trained"
  fi
done
log "[3/4] Done"

# 4. Walk-forward validation
log "[4/4] Walk-forward validation for all 4 assets..."
for a in "${ASSETS_ALL[@]}"; do
  log "  walk-forward $a..."
  python scripts/walk_forward_full.py --asset "$a" --model-suffix _align2s \
    --holdout-days 7 --min-dm 2 \
    > "output/walkfwd_${a}_align2s.log" 2>&1 || {
    log "  WARNING: $a walk-forward failed"
  }
done
log "[4/4] Done"

# Final report
log "Writing final report..."
{
  echo ""
  echo "=== ALIGN2S PIPELINE COMPLETE (stage B) ==="
  echo "Finished: $(ts)"
  echo ""
  echo "--- Staging models ---"
  ls -la models/*_align2s_xgb.json models/*_align2s_lstm.pt 2>/dev/null
  echo ""
  echo "--- Training data row counts ---"
  for a in "${ASSETS_ALL[@]}"; do
    if [ -f "ml/training_data/${a}_features.csv" ]; then
      echo "$a XGB CSV: $(wc -l < ml/training_data/${a}_features.csv) rows"
    fi
    if [ -f "ml/training_data/${a}_lstm_sequences.npz" ]; then
      echo "$a LSTM npz: $(($(stat -c%s ml/training_data/${a}_lstm_sequences.npz)/1000000))MB"
    fi
  done
  echo ""
  echo "--- Walk-forward summaries ---"
  for a in "${ASSETS_ALL[@]}"; do
    if [ -f "output/walkfwd_${a}_align2s.log" ]; then
      echo ""
      echo ">>> $a"
      grep -E "WINNER|holdout|fold [0-9]|PnL|wr|HOLDOUT|Best:" "output/walkfwd_${a}_align2s.log" 2>/dev/null | tail -30
    fi
  done
  echo ""
  echo "--- Next steps (user review) ---"
  echo "  1. Review walk-forward PnL per asset above"
  echo "  2. If PnL holds / improves:"
  echo "     a. Run ensemble_sweep.py then pnl_sweep.py to recalibrate dynamic_k / threshold"
  echo "        (these OVERWRITE config ensemble block — back up config first)"
  echo "     b. Run scripts/promote_retrain_models.sh align2s"
  echo "     c. Restart live bot"
} >> "$REPORT"

log "=== STAGE B DONE — see $REPORT ==="
