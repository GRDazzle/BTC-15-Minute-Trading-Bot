#!/bin/bash
set -e
cd "$(dirname "$0")/.."
AFTER="2026-04-24T20:52:00"
BEFORE="2026-04-24T22:00:00"
OUT_DIR=output/replay_compare_v2
mkdir -p "$OUT_DIR"
ASSETS="BTC ETH SOL XRP HYPE BNB DOGE"
pids=""
for asset in $ASSETS; do
    python -u scripts/replay_live_inference.py \
        --asset "$asset" --from 2026-04-24 --to 2026-04-24 \
        --after-ts "$AFTER" --before-ts "$BEFORE" \
        --balance 10000 \
        --out "$OUT_DIR/replay_${asset,,}.csv" > "$OUT_DIR/replay_${asset,,}.log" 2>&1 &
    pids="$pids $!"
done
for pid in $pids; do wait "$pid" || true; done
echo ""; echo "=== LIVE vs REPLAY (high balance) ==="
printf "%-6s %8s %8s  %s\n" "asset" "live" "replay" "notes"
printf "%-6s %8s %8s\n" "-----" "----" "------"
TL=0; TR=0
for asset in $ASSETS; do
    live_n=$(awk -F',' -v a="$asset" 'NR>1 && $2==a && $1>="'"$AFTER"'" && $1<="'"$BEFORE"'"' output/trades.csv 2>/dev/null | wc -l | tr -d ' ')
    replay_n=$(tail -n +2 "$OUT_DIR/replay_${asset,,}.csv" 2>/dev/null | wc -l | tr -d ' ')
    TL=$((TL + live_n)); TR=$((TR + replay_n))
    notes=""
    [ "$live_n" != "$replay_n" ] && notes="MISMATCH"
    printf "%-6s %8s %8s  %s\n" "$asset" "$live_n" "$replay_n" "$notes"
done
printf "%-6s %8s %8s\n" "TOTAL" "$TL" "$TR"
