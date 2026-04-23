#!/bin/bash
# Serial walk-forward for all 4 assets on the 2s-grain align2s training CSVs.
# Walk-forward internally trains XGB per fold — uses the training CSV only,
# not any pretrained model. Model suffix is via --staging.
set -e
cd "$(dirname "$0")/.."

ASSETS=(BTC ETH SOL XRP)
STATE=output/align2s_walkfwd_state.txt

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" | tee -a "$STATE"; }

log "=== WALK-FORWARD ALIGN2S START ==="
for a in "${ASSETS[@]}"; do
  # Skip BTC if already done (started separately)
  if [ -f "output/walkfwd_${a}_align2s.log" ] && grep -q "HOLDOUT\|WINNER" "output/walkfwd_${a}_align2s.log" 2>/dev/null; then
    log "  $a already has walk-forward results; skipping"
    continue
  fi
  log "  running walk-forward for $a..."
  python scripts/walk_forward_full.py --asset "$a" --staging align2s --holdout-days 7 \
    > "output/walkfwd_${a}_align2s.log" 2>&1 || {
    log "  $a walk-forward FAILED — see log"
  }
  log "  $a walk-forward done"
done
log "=== WALK-FORWARD ALIGN2S DONE ==="

# Summary
{
  echo ""
  echo "=== WALK-FORWARD SUMMARY ==="
  for a in "${ASSETS[@]}"; do
    if [ -f "output/walkfwd_${a}_align2s.log" ]; then
      echo ""
      echo ">>> $a"
      grep -E "WINNER|HOLDOUT|holdout|fold [0-9]|Best:|PnL|wr=" "output/walkfwd_${a}_align2s.log" 2>/dev/null | tail -20
    fi
  done
} | tee -a "$STATE"
