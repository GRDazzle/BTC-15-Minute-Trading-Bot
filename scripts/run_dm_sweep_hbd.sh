#!/bin/bash
# dm sweep for HYPE, BNB, DOGE: full walk-forward + OOS k-sweep per dm.
# Stages models alongside existing BTC/ETH/SOL/XRP staging (different asset names).
# OOS outputs land in separate oos_hbd_dm{2..7}.json files.

set -e
cd "$(dirname "$0")/.."

XGB_DKS="1,2,3,4.5,6,8"
LSTM_DKS="1,2,3,4.5,6,8"
HOLDOUT=7
ASSETS="HYPE,BNB,DOGE"

echo "=== HBD dm sweep start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> output/dm_sweep_summary.txt

for DM in 2 3 4 5 6 7; do
  echo ""
  echo "=================================================================="
  echo "  DM = $DM :: starting at $(date -u +%H:%M:%S)"
  echo "=================================================================="

  STAGE="hbd_unstacked_dm${DM}"
  WF_LOG="output/walk_forward_hbd_dm${DM}_log.txt"
  OOS_LOG="output/oos_hbd_dm${DM}_log.txt"
  OOS_OUT="output/oos_hbd_dm${DM}.json"

  echo "  [1/2] Walk-forward train (staging/${STAGE}) -> ${WF_LOG}"
  python scripts/walk_forward_full.py \
    --asset "$ASSETS" \
    --min-dm "$DM" \
    --n-folds 3 \
    --holdout-days 10 \
    --staging "$STAGE" \
    --trade-log-suffix "_hbd_wf_dm${DM}" \
    > "$WF_LOG" 2>&1
  echo "  [1/2] Walk-forward done at $(date -u +%H:%M:%S)"

  echo "  [2/2] OOS 2D k-sweep (min_dm=$DM) -> ${OOS_LOG}"
  python scripts/oos_ensemble_test.py \
    --asset "$ASSETS" \
    --holdout-days "$HOLDOUT" \
    --mode ensemble \
    --xgb-dk-sweep "$XGB_DKS" \
    --lstm-dk-sweep "$LSTM_DKS" \
    --xgb-model-dir "models/staging_${STAGE}" \
    --min-dm "$DM" \
    --out "$OOS_OUT" \
    > "$OOS_LOG" 2>&1
  echo "  [2/2] OOS done at $(date -u +%H:%M:%S)"
done

echo "=== HBD sweep done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> output/dm_sweep_summary.txt
