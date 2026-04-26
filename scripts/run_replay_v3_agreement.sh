#!/bin/bash
# Run replay per-asset with the ensemble_align2s config from pnl_sweep v3.
# Compare replay's actual trade outcomes to sweep's simulated numbers.
set -e
cd "$(dirname "$0")/.."

REPORT=output/align2s_agreement_report.txt

# Sweep's time range (last 24h from when sweep was kicked off)
AFTER=2026-04-23T00:00:00
BEFORE=2026-04-24T00:00:00

# Per-asset configs from ensemble_align2s (config/trading.json)
#   BTC: thr=0.62 k_xgb=4.5 k_lstm=3.0 min_dm=2 maxP=80 (sweep: -$20.87, 76% WR, 50 trades)
#   ETH: thr=0.75 k_xgb=8.0 k_lstm=3.0 min_dm=7 maxP=85 (sweep: +$494.65, 100% WR, 15 trades)
#   SOL: thr=0.75 k_xgb=6.0 k_lstm=6.0 min_dm=7 maxP=90 (sweep: +$64.82, 95.2% WR, 21 trades)
#   XRP: thr=0.75 k_xgb=3.0 k_lstm=8.0 min_dm=7 maxP=90 (sweep: +$34.35, 88.5% WR, 26 trades)

declare -A THRESH=( ["BTC"]=0.62 ["ETH"]=0.75 ["SOL"]=0.75 ["XRP"]=0.75 )
declare -A KXGB=( ["BTC"]=4.5 ["ETH"]=8.0 ["SOL"]=6.0 ["XRP"]=3.0 )
declare -A KLSTM=( ["BTC"]=3.0 ["ETH"]=3.0 ["SOL"]=6.0 ["XRP"]=8.0 )
declare -A MDM=( ["BTC"]=2 ["ETH"]=7 ["SOL"]=7 ["XRP"]=7 )
declare -A MAXP=( ["BTC"]=80 ["ETH"]=85 ["SOL"]=90 ["XRP"]=90 )
declare -A SWEEP_TRADES=( ["BTC"]=50 ["ETH"]=15 ["SOL"]=21 ["XRP"]=26 )
declare -A SWEEP_WR=( ["BTC"]=76.0 ["ETH"]=100.0 ["SOL"]=95.2 ["XRP"]=88.5 )
declare -A SWEEP_PNL=( ["BTC"]=-20.87 ["ETH"]=494.65 ["SOL"]=64.82 ["XRP"]=34.35 )

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"; }

log "=== REPLAY AGREEMENT TEST START ==="

for asset in BTC ETH SOL XRP; do
  log "replaying $asset with thr=${THRESH[$asset]} k_xgb=${KXGB[$asset]} k_lstm=${KLSTM[$asset]} min_dm=${MDM[$asset]} maxP=${MAXP[$asset]}..."
  python scripts/replay_live_inference.py \
    --asset "$asset" \
    --from 2026-04-22 --to 2026-04-24 \
    --after-ts "$AFTER" --before-ts "$BEFORE" \
    --xgb-suffix _align2s --lstm-suffix _align2s \
    --override-threshold "${THRESH[$asset]}" \
    --override-k-xgb "${KXGB[$asset]}" \
    --override-k-lstm "${KLSTM[$asset]}" \
    --override-min-dm "${MDM[$asset]}" \
    --override-max-price "${MAXP[$asset]}" \
    --out "output/replay_${asset,,}_v3.csv" \
    > "output/replay_${asset,,}_v3.log" 2>&1 || log "  $asset replay FAILED"
done

# Build agreement report
log "writing agreement report to $REPORT"
{
  echo "=== ALIGN2S SWEEP vs REPLAY AGREEMENT REPORT ==="
  echo "Generated: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "Time range: $AFTER .. $BEFORE"
  echo ""
  echo "Formula C (2-way confidence-weighted XGB+LSTM, no fusion) in both sweep and replay."
  echo ""
  printf "%-5s | %-50s | %-30s | %-30s\n" "Asset" "Config (thr/k_xgb/k_lstm/min_dm/maxP)" "Sweep (trades/WR/PnL)" "Replay (trades/WR/PnL)"
  printf "%-5s | %-50s | %-30s | %-30s\n" "-----" "--------------------------------------------------" "------------------------------" "------------------------------"
  for asset in BTC ETH SOL XRP; do
    cfg="thr=${THRESH[$asset]} k_xgb=${KXGB[$asset]} k_lstm=${KLSTM[$asset]} min_dm=${MDM[$asset]} maxP=${MAXP[$asset]}c"
    sweep="n=${SWEEP_TRADES[$asset]} WR=${SWEEP_WR[$asset]}% PnL=\$${SWEEP_PNL[$asset]}"
    # Extract replay summary from its log
    rep=$(grep -oP 'trades=\K[0-9]+' "output/replay_${asset,,}_v3.log" 2>/dev/null | tail -1 || echo "?")
    wr=$(grep -oP 'WR=\K[0-9.]+' "output/replay_${asset,,}_v3.log" 2>/dev/null | tail -1 || echo "?")
    pnl=$(grep -oP 'PnL=\K[+-]?[0-9.]+' "output/replay_${asset,,}_v3.log" 2>/dev/null | tail -1 || echo "?")
    replay="n=${rep} WR=${wr}% PnL=\$${pnl}"
    printf "%-5s | %-50s | %-30s | %-30s\n" "$asset" "$cfg" "$sweep" "$replay"
  done
  echo ""
  echo "--- Interpretation guide ---"
  echo " * Trade counts match closely: execution path is faithful to sweep simulation."
  echo " * Significant gap (>20% diff in counts, or WR off by >5pp): hunt for state drift"
  echo "   (tick buffer reconstruction, poll snapshot timing, feature computation)."
  echo ""
  echo "--- Per-asset replay logs ---"
  for asset in BTC ETH SOL XRP; do
    echo ""
    echo ">>> $asset"
    tail -15 "output/replay_${asset,,}_v3.log" 2>/dev/null
  done
} > "$REPORT"

log "=== AGREEMENT TEST DONE — see $REPORT ==="
