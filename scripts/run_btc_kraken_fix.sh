#!/bin/bash
# BTC Kraken backfill + BTC-only retrain (safe mode, no auto-promote).
# Fixes the XBT/USD -> BTC/USD pair name issue and re-aligns BTC training data.

set -e
cd "$(dirname "$0")/.."

DATE_TAG=$(date -u +%Y%m%d)

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "=== BTC KRAKEN FIX PIPELINE START (tag=${DATE_TAG}) ==="

# 1. Backfill BTC Kraken data via REST (last 10 days covers the gap Apr 14+)
log "[1/4] Backfilling BTC Kraken trades (REST fetch, ~10 days)..."
python scripts/fetch_kraken_trades.py --assets BTC --days 10 \
  > output/btcfix_fetch_kraken.log 2>&1 \
  || log "  WARNING: Kraken fetch had issues (may have been rate limited)"
log "[1/4] Done"

# 2. Regen BTC XGB training features (picks up newly-filled Kraken CSVs)
log "[2/4] Regenerating BTC XGB training data..."
python scripts/generate_training_data.py --asset BTC --days 60 \
  > output/btcfix_gen_xgb.log 2>&1
log "[2/4] Done"

# 3. Regen BTC LSTM sequences (with P3 sort fix applied)
log "[3/4] Regenerating BTC LSTM sequences..."
python scripts/generate_lstm_training_data.py --asset BTC --days 60 \
  > output/btcfix_gen_lstm.log 2>&1
log "[3/4] Done"

# 4. Retrain BTC LSTM and walk-forward XGB (staging, no auto-promote)
log "[4a/4] Retraining BTC LSTM (GPU)..."
python scripts/train_lstm.py --asset BTC --min-dm 2 \
  --model-suffix "_btcfix_${DATE_TAG}" \
  > output/btcfix_lstm.log 2>&1 \
  || log "  WARNING: BTC LSTM train returned non-zero (model may still have saved)"

log "[4b/4] Walk-forward BTC XGB @ dm=7 (GPU, staging)..."
python scripts/walk_forward_full.py --asset BTC --min-dm 7 --n-folds 3 \
  --holdout-days 10 --staging "btcfix_${DATE_TAG}" \
  > output/btcfix_wf.log 2>&1

log "[4/4] Done"

log ""
log "=== BTC KRAKEN FIX PIPELINE COMPLETE ==="
log "Backup paths (NOT promoted, live unaffected):"
log "  LSTM:  models/BTC_btcfix_${DATE_TAG}_lstm.pt"
log "  XGB:   models/staging_btcfix_${DATE_TAG}/BTC_xgb.json"
log ""
log "To promote after evaluation:"
log "  cp models/BTC_btcfix_${DATE_TAG}_lstm.pt models/BTC_lstm.pt"
log "  cp models/staging_btcfix_${DATE_TAG}/BTC_xgb.json models/BTC_xgb.json"
log "  cp models/staging_btcfix_${DATE_TAG}/BTC_xgb_features.json models/BTC_xgb_features.json"
