#!/bin/bash
# Stage C: runs after ETH/SOL/XRP LSTM regens produce NEW npz files.
#
# Precondition (in progress in background):
#   - ETH/SOL/XRP LSTM 2s-grain regens (writing new .npz files at
#     ml/training_data/{ASSET}_lstm_sequences.npz)
#   - Old files already renamed to *_10s_old.npz
#   - BTC _align2s XGB and LSTM: already trained
#   - ETH/SOL/XRP _align2s XGB: already trained
#
# This script:
#   1. Waits for ETH/SOL/XRP new LSTM .npz files (file existence at original path,
#      since we renamed the old ones, any file there is new)
#   2. Trains ETH/SOL/XRP LSTM _align2s sequentially (GPU OOM guard)
#   3. Runs ensemble_sweep for all 4 with --model-suffix _align2s
#      -> writes candidates to output/ensemble_sweep/{ASSET}_align2s_candidates.json
#   4. Runs pnl_sweep for all 4 with --model-suffix _align2s
#      -> writes winning params to config/trading.json under "ensemble_align2s" key
#      (does NOT touch production "ensemble" key)
#   5. Writes final report

set -e
cd "$(dirname "$0")/.."

ASSETS_OTHER=(ETH SOL XRP)
ASSETS_ALL=(BTC ETH SOL XRP)
REPORT=output/align2s_pipeline_report.txt
STATE=output/align2s_stage_c_state.txt
MIN_LSTM_NPZ_SIZE=100000000
MIN_LSTM_PT_SIZE=100000

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" | tee -a "$STATE"; }

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

log "=== STAGE C START ==="

# 1. Wait for new ETH/SOL/XRP LSTM npz files
log "[1/4] Waiting for ETH/SOL/XRP new LSTM sequences..."
for a in "${ASSETS_OTHER[@]}"; do
  wait_for_file "ml/training_data/${a}_lstm_sequences.npz" "$MIN_LSTM_NPZ_SIZE" "${a}_lstm_sequences.npz"
done
log "[1/4] Done"

# 2. Train ETH/SOL/XRP LSTM _align2s sequentially
log "[2/4] Training ETH/SOL/XRP LSTM _align2s..."
for a in "${ASSETS_OTHER[@]}"; do
  log "  training $a LSTM..."
  python scripts/train_lstm.py --asset "$a" --model-suffix _align2s \
    > "output/train_${a}_align2s_lstm.log" 2>&1 || {
    log "  FAILED: $a LSTM training"
    echo "STAGE C: $a LSTM training FAILED" >> "$REPORT"
    # continue to next asset
  }
  if [ -f "models/${a}_align2s_lstm.pt" ]; then
    log "  $a LSTM saved"
  fi
done
log "[2/4] Done"

# 3. Ensemble sweep (XGB+LSTM accuracy + candidates)
log "[3/4] Running ensemble_sweep for all 4 assets..."
for a in "${ASSETS_ALL[@]}"; do
  log "  sweeping $a..."
  python scripts/ensemble_sweep.py --asset "$a" --model-suffix _align2s \
    > "output/sweep_${a}_align2s.log" 2>&1 || {
    log "  WARNING: $a ensemble sweep failed"
  }
done
log "[3/4] Done"

# 4. PnL sweep (uses candidates + writes ensemble_align2s config block)
log "[4/4] Running pnl_sweep for all 4 assets..."
python scripts/pnl_sweep.py --asset "BTC,ETH,SOL,XRP" --model-suffix _align2s \
  > output/pnl_sweep_align2s.log 2>&1 || {
  log "  WARNING: pnl_sweep failed"
}
log "[4/4] Done"

# Final report
{
  echo ""
  echo "=== ALIGN2S PIPELINE COMPLETE ==="
  echo "Finished: $(ts)"
  echo ""
  echo "--- Staging models ---"
  ls -la models/*_align2s_xgb.json models/*_align2s_lstm.pt 2>/dev/null
  echo ""
  echo "--- Walk-forward (XGB-only sanity) ---"
  for a in "${ASSETS_ALL[@]}"; do
    if [ -f "output/walkfwd_${a}_align2s.log" ]; then
      echo ""
      echo ">>> $a"
      grep -E "WINNER|HOLDOUT|fold [0-9]|Best:|PnL|holdout" "output/walkfwd_${a}_align2s.log" 2>/dev/null | tail -15
    fi
  done
  echo ""
  echo "--- Ensemble sweep candidates ---"
  for a in "${ASSETS_ALL[@]}"; do
    cand="output/ensemble_sweep/${a}_align2s_candidates.json"
    if [ -f "$cand" ]; then
      echo "$a top 5:"
      python -c "
import json
try:
    data = json.load(open('$cand'))
    for c in data[:5]:
        print(f'  ml_w={c[\"ml_weight\"]:.2f} thr={c[\"threshold\"]:.2f} acc={c[\"accuracy\"]*100:.1f}% net={c[\"net_correct\"]:.0f} n={c[\"traded_count\"]}')
except Exception as e:
    print(f'  could not read: {e}')
"
    fi
  done
  echo ""
  echo "--- PnL sweep results (written to config ensemble_align2s key) ---"
  python -c "
import json
try:
    cfg = json.load(open('config/trading.json'))
    for a in ['BTC','ETH','SOL','XRP']:
        ens = cfg['assets'].get(a, {}).get('ensemble_align2s', {})
        if ens:
            print(f'{a}: ml_w={ens.get(\"ml_weight\"):.2f} thr={ens.get(\"threshold\"):.2f} maxP={ens.get(\"max_price_cents\")}c k_xgb={ens.get(\"dynamic_k_xgb\", 0)} k_lstm={ens.get(\"dynamic_k_lstm\", 0)} pnl=\$' + str(ens.get('pnl_sweep_total_pnl', 'n/a')))
        else:
            print(f'{a}: no ensemble_align2s block in config')
except Exception as e:
    print(f'Could not read config: {e}')
"
  echo ""
  echo "--- Next steps (user manual) ---"
  echo "  1. Review per-asset PnL + threshold params above"
  echo "  2. If params look good, apply via promote_retrain_models.sh align2s"
  echo "     (current production still intact at models/{ASSET}_xgb.json / _lstm.pt)"
  echo "  3. Before restarting live bot, also copy ensemble_align2s -> ensemble in config"
  echo "     (or manually adjust config to use new dynamic_k / threshold values)"
  echo "  4. Restart bot to hot-reload new models + config"
} >> "$REPORT"

log "=== STAGE C DONE — see $REPORT ==="
