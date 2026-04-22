#!/bin/bash
# Manually promote retrained models to live. Run AFTER evaluating live session.
# Usage: bash scripts/promote_retrain_models.sh <DATE_TAG>
#   e.g. bash scripts/promote_retrain_models.sh 20260422

set -e
cd "$(dirname "$0")/.."

DATE_TAG="${1:?Usage: $0 <DATE_TAG>   (e.g. 20260422)}"
ASSETS=(BTC ETH SOL XRP)

log() {
  echo "[$(date -u +%H:%M:%S)] $*"
}

log "=== PROMOTING RETRAIN MODELS (tag=${DATE_TAG}) ==="

# Backup current live models with timestamp so we can rollback
BACKUP_DIR="models/backup_pre_promote_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
log "Backing up current live models to $BACKUP_DIR"
for asset in "${ASSETS[@]}"; do
  for f in "models/${asset}_lstm.pt" \
           "models/${asset}_xgb.json" "models/${asset}_xgb_features.json" \
           "models/${asset}_weekday_xgb.json" "models/${asset}_weekday_xgb_features.json" \
           "models/${asset}_weekend_xgb.json" "models/${asset}_weekend_xgb_features.json"; do
    [ -f "$f" ] && cp "$f" "$BACKUP_DIR/"
  done
done

# Promote LSTM
for asset in "${ASSETS[@]}"; do
  src="models/${asset}_retrain_${DATE_TAG}_lstm.pt"
  dst="models/${asset}_lstm.pt"
  if [ -f "$src" ]; then
    cp "$src" "$dst"
    log "  promoted LSTM: $dst"
  else
    log "  WARNING missing $src"
  fi
done

# Promote BTC XGB
for variant in "" "_weekday" "_weekend"; do
  for ext in "xgb.json" "xgb_features.json"; do
    src="models/staging_retrain_btc_dm7_${DATE_TAG}/BTC${variant}_${ext}"
    dst="models/BTC${variant}_${ext}"
    [ -f "$src" ] && cp "$src" "$dst" && log "  promoted XGB: $dst"
  done
done

# Promote ETH/SOL/XRP XGB
for asset in ETH SOL XRP; do
  for variant in "" "_weekday" "_weekend"; do
    for ext in "xgb.json" "xgb_features.json"; do
      src="models/staging_retrain_esx_dm4_${DATE_TAG}/${asset}${variant}_${ext}"
      dst="models/${asset}${variant}_${ext}"
      [ -f "$src" ] && cp "$src" "$dst" && log "  promoted XGB: $dst"
    done
  done
done

log "=== PROMOTE COMPLETE ==="
log "Hot-reload will pick up new models at next 15-min window boundary."
log "To rollback: cp $BACKUP_DIR/* models/"
