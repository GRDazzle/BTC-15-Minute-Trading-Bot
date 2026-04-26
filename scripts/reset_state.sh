#!/bin/bash
# Reset bot state before a restart.
#
# Usage:
#   ./scripts/reset_state.sh              # fast wipe — no archive, just reset
#   ./scripts/reset_state.sh --archive    # archive trade history first, then wipe
#
# What gets reset in every run:
#   - bot + manager processes are killed (Python main.py / manager.py matches)
#   - data/account_state.json removed -> regens to $25/asset on next startup
#   - output/balance.csv, output/trades.csv, output/signal_log.csv truncated
#   - logs/lstm_tick_dumps/ removed (recreated on first LSTM dump)
#   - logs/trading.log removed (recreated by loguru on startup)
#
# With --archive: all of the above files are copied to
# archive/state_reset_<YYYYMMDD_HHMMSS>/ first, so you can inspect or repair
# trades later. Without the flag, they're just wiped — faster, loses history.

set -e

cd "$(dirname "$0")/.."

# -- parse args ---------------------------------------------------------------
ARCHIVE=false
for arg in "$@"; do
    case "$arg" in
        --archive|-a)
            ARCHIVE=true
            ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $arg"
            echo "Usage: $0 [--archive]"
            exit 2
            ;;
    esac
done

# -- 1. kill bot + manager ---------------------------------------------------
# Wait loop — don't proceed to file-wipe until the processes are actually gone.
# Python bots can take a few seconds to exit cleanly and may write to
# account_state.json during shutdown, racing with rm and silently undoing
# the wipe. We kill, then poll-until-gone.
echo "[1/3] Stopping bot + manager (if running)..."
_kill_matching() {
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -NonInteractive -Command \
            "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'main\.py|manager\.py' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }" \
            >/dev/null 2>&1 || true
    else
        pids=$(ps -eo pid,args 2>/dev/null | grep -E "python.*(main\.py|manager\.py)" | grep -v grep | awk '{print $1}')
        if [ -n "$pids" ]; then
            echo "$pids" | xargs -r kill -9 2>/dev/null || true
        fi
    fi
}
_count_matching() {
    if command -v powershell.exe >/dev/null 2>&1; then
        n=$(powershell.exe -NoProfile -NonInteractive -Command \
            "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'main\.py|manager\.py' }).Count" 2>/dev/null || echo 0)
        echo "$n" | tr -d '\r\n '
    else
        ps -eo pid,args 2>/dev/null | grep -cE "python.*(main\.py|manager\.py)" || echo 0
    fi
}
_kill_matching
# Poll until gone (max 15 attempts x 1s = 15s)
attempts=0
while true; do
    n=$(_count_matching)
    n=${n:-0}
    if [ "$n" = "0" ]; then
        echo "  bot + manager stopped"
        break
    fi
    attempts=$((attempts + 1))
    if [ "$attempts" -ge 15 ]; then
        echo "  WARN: $n bot/manager process(es) still running after 15s; retrying kill once"
        _kill_matching
        sleep 2
        break
    fi
    sleep 1
    # Re-issue kill each loop in case a watchdog resurrected
    _kill_matching
done
# Give the filesystem an extra beat for any in-flight writes to flush/close
sleep 1

# -- 2. archive (optional) ---------------------------------------------------
if [ "$ARCHIVE" = true ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    ARCH="archive/state_reset_${TS}"
    mkdir -p "${ARCH}/data" "${ARCH}/logs" "${ARCH}/output"
    echo "[2/3] Archiving to ${ARCH}..."
    [ -f data/account_state.json ] && cp data/account_state.json "${ARCH}/data/" && echo "  + data/account_state.json"
    for f in output/balance.csv output/trades.csv output/signal_log.csv; do
        [ -f "$f" ] && cp "$f" "${ARCH}/output/" && echo "  + $f"
    done
    [ -d logs/lstm_tick_dumps ] && cp -r logs/lstm_tick_dumps "${ARCH}/logs/" && echo "  + logs/lstm_tick_dumps/"
    [ -f logs/trading.log ] && cp logs/trading.log "${ARCH}/logs/" && echo "  + logs/trading.log"
    echo "  saved to ${ARCH}"
else
    echo "[2/3] Skipping archive (no --archive flag)"
fi

# -- 3. wipe -----------------------------------------------------------------
echo "[3/3] Wiping state..."
if [ -f data/account_state.json ]; then
    rm data/account_state.json
    echo '  removed data/account_state.json (regens to $25/asset on startup)'
fi
for f in output/balance.csv output/trades.csv output/signal_log.csv; do
    if [ -f "$f" ]; then
        : > "$f"
        echo "  truncated $f"
    fi
done
[ -d logs/lstm_tick_dumps ] && rm -rf logs/lstm_tick_dumps && echo "  removed logs/lstm_tick_dumps/"
[ -f logs/trading.log ] && rm logs/trading.log && echo "  removed logs/trading.log"

cat <<'EOF'

Done. Restart with:
  $env:LSTM_SELFTEST = "1"
  $env:LSTM_DUMP_TICKS = "1"
  python main.py --assets BTC,ETH,SOL,XRP,HYPE,BNB,DOGE
  # (open manager.py in a separate window)

EOF
