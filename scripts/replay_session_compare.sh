#!/bin/bash
# Replay today's live session per-asset in parallel, then diff against live trades.
# Output a concise report: does replay reproduce the trades live placed?
set -e
cd "$(dirname "$0")/.."

AFTER="2026-04-24T20:52:00"
BEFORE="2026-04-24T22:00:00"
DATE_FROM="2026-04-24"
DATE_TO="2026-04-24"

LIVE_TRADES=output/trades.csv
OUT_DIR=output/replay_compare
mkdir -p "$OUT_DIR"

ASSETS="BTC ETH SOL XRP HYPE BNB DOGE"

echo "=== Launching 7 parallel replays for $AFTER -> $BEFORE ==="
pids=""
for asset in $ASSETS; do
    OUT="$OUT_DIR/replay_${asset,,}_session.csv"
    LOG="$OUT_DIR/replay_${asset,,}_session.log"
    python -u scripts/replay_live_inference.py \
        --asset "$asset" \
        --from "$DATE_FROM" --to "$DATE_TO" \
        --after-ts "$AFTER" --before-ts "$BEFORE" \
        --out "$OUT" \
        > "$LOG" 2>&1 &
    pids="$pids $!"
    echo "  $asset: pid $!"
done

echo "=== Waiting for replays... ==="
for pid in $pids; do
    wait "$pid" || echo "  pid $pid exited non-zero"
done
echo "=== All replays done ==="

# Compare: for each asset, how many trades live vs replay?
echo ""
echo "=============================================="
echo "  LIVE vs REPLAY COMPARISON"
echo "=============================================="
printf "%-6s %8s %8s  %s\n" "asset" "live" "replay" "notes"
printf "%-6s %8s %8s\n" "-----" "----" "------"

TOTAL_LIVE=0
TOTAL_REPLAY=0
for asset in $ASSETS; do
    live_n=$(awk -F',' -v a="$asset" 'NR>1 && $2==a && $1>="'"$AFTER"'" && $1<="'"$BEFORE"'"' "$LIVE_TRADES" 2>/dev/null | wc -l | tr -d ' ')
    replay_csv="$OUT_DIR/replay_${asset,,}_session.csv"
    replay_n=$(tail -n +2 "$replay_csv" 2>/dev/null | wc -l | tr -d ' ')
    TOTAL_LIVE=$((TOTAL_LIVE + live_n))
    TOTAL_REPLAY=$((TOTAL_REPLAY + replay_n))
    notes=""
    if [ "$live_n" != "$replay_n" ]; then
        notes="MISMATCH"
    fi
    printf "%-6s %8s %8s  %s\n" "$asset" "$live_n" "$replay_n" "$notes"
done
echo ""
printf "%-6s %8s %8s\n" "TOTAL" "$TOTAL_LIVE" "$TOTAL_REPLAY"
