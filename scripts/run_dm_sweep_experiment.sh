#!/bin/bash
# Per-dm walk-forward + OOS k-sweep experiment.
# For each dm in 2..7:
#   1. Walk-forward train XGB at min_dm=dm (staging)
#   2. OOS 2D k-sweep at min_dm=dm
# Writes logs and results under output/.

set -e
cd "$(dirname "$0")/.."

XGB_DKS="1,2,3,4.5,6,8"
LSTM_DKS="1,2,3,4.5,6,8"
HOLDOUT=7

SUMMARY="output/dm_sweep_summary.txt"
echo "=== DM sweep experiment started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$SUMMARY"

for DM in 2 3 4 5 6 7; do
  echo ""
  echo "=================================================================="
  echo "  DM = $DM :: starting at $(date -u +%H:%M:%S)"
  echo "=================================================================="

  STAGE="unstacked_dm${DM}"
  WF_LOG="output/walk_forward_dm${DM}_log.txt"
  OOS_LOG="output/oos_dm${DM}_log.txt"
  OOS_OUT="output/oos_dm${DM}.json"

  echo "  [1/2] Walk-forward train (staging/${STAGE}) -> ${WF_LOG}"
  python scripts/walk_forward_full.py \
    --asset BTC,ETH,SOL,XRP \
    --min-dm "$DM" \
    --n-folds 3 \
    --holdout-days 10 \
    --staging "$STAGE" \
    --trade-log-suffix "_wf_dm${DM}" \
    > "$WF_LOG" 2>&1
  echo "  [1/2] Walk-forward done at $(date -u +%H:%M:%S)"

  echo "  [2/2] OOS 2D k-sweep (min_dm=$DM) -> ${OOS_LOG}"
  python scripts/oos_ensemble_test.py \
    --asset BTC,ETH,SOL,XRP \
    --holdout-days "$HOLDOUT" \
    --mode ensemble \
    --xgb-dk-sweep "$XGB_DKS" \
    --lstm-dk-sweep "$LSTM_DKS" \
    --xgb-model-dir "models/staging_${STAGE}" \
    --min-dm "$DM" \
    --out "$OOS_OUT" \
    > "$OOS_LOG" 2>&1
  echo "  [2/2] OOS done at $(date -u +%H:%M:%S)"

  echo "DM=$DM complete; files: $WF_LOG $OOS_OUT" >> "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== DM sweep experiment finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$SUMMARY"
