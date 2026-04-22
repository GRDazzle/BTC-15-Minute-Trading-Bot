#!/bin/bash
# Resume the dm sweep from where the previous run stopped.
# dm=2 already fully complete; dm=7 XGB already trained (from prior standalone run).
# Re-do dm=3 (was partial), do dm=4/5/6 full, and OOS-only for dm=7.

set -e
cd "$(dirname "$0")/.."

XGB_DKS="1,2,3,4.5,6,8"
LSTM_DKS="1,2,3,4.5,6,8"
HOLDOUT=7

echo "=== resume dm sweep at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> output/dm_sweep_summary.txt

for DM in 3 4 5 6; do
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
done

# dm=7 — XGB already trained; just run OOS
DM=7
echo ""
echo "  DM = 7 :: OOS only (skipping retrain)"
python scripts/oos_ensemble_test.py \
  --asset BTC,ETH,SOL,XRP \
  --holdout-days "$HOLDOUT" \
  --mode ensemble \
  --xgb-dk-sweep "$XGB_DKS" \
  --lstm-dk-sweep "$LSTM_DKS" \
  --xgb-model-dir "models/staging_unstacked_dm7" \
  --min-dm "$DM" \
  --out "output/oos_dm7.json" \
  > "output/oos_dm7_log.txt" 2>&1
echo "  DM = 7 OOS done at $(date -u +%H:%M:%S)"

echo "" >> output/dm_sweep_summary.txt
echo "=== resume finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> output/dm_sweep_summary.txt
