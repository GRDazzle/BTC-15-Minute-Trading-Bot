"""Simulate what dynamic blend would have done on today's actual trades.

For each settled trade since Apr 15 22:00 UTC:
  1. Find the matching log line to get xgb_p, lstm_p, fus_p
  2. Compute what the dynamic-blend ensemble_p would be
  3. Compare: kept (same direction), filtered (no signal), or flipped (opposite)
  4. Tally hypothetical PnL
"""
import re
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone

CUTOFF = datetime(2026, 4, 15, 22, 0, tzinfo=timezone.utc)

# Dynamic blend params (matches strategies/kalshi_strategy.py:1791-1807)
DYNAMIC_K = 4.5
LSTM_MIN_W, LSTM_MAX_W = 0.10, 0.40
XGB_MAX_W = 0.60


def dyn_blend(xgb, lstm, fus, ml_w):
    xgb_conf = abs(xgb - 0.5) * 2
    dyn_xgb_w = ml_w + (XGB_MAX_W - ml_w) * (xgb_conf ** DYNAMIC_K)
    lstm_conf = abs(lstm - 0.5) * 2
    dyn_lstm_w = LSTM_MIN_W + (LSTM_MAX_W - LSTM_MIN_W) * (lstm_conf ** DYNAMIC_K)
    dyn_fus_w = max(0.0, 1.0 - dyn_xgb_w - dyn_lstm_w)
    return dyn_xgb_w * xgb + dyn_lstm_w * lstm + dyn_fus_w * fus


# Load trades
trades = []
with open('output/trades.csv') as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        if not row.get('outcome') or not row.get('pnl'):
            continue
        ts = datetime.fromisoformat(row['timestamp'])
        if ts < CUTOFF:
            continue
        trades.append({
            'ts': ts,
            'asset': row['asset'],
            'window_id': row['window_id'],
            'direction': row['direction'],
            'price': int(row['price_cents']),
            'contracts': int(row['contracts']),
            'cost': float(row['cost']),
            'outcome': row['outcome'],
            'pnl': float(row['pnl']),
            'confidence': float(row['confidence']),
        })

# Parse log signals
log_re = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[(\w+)\] Ensemble (BULLISH|BEARISH).*p=([\d.]+) \(xgb=([\d.]+) lstm=([\d.]+) fus=([\d.]+)\) price=(\d+)c'
)
signals = []
import glob
# Loguru timestamps are LOCAL time (PT = UTC-7)
LOCAL_OFFSET_HOURS = 7
log_files = ['logs/trading.log'] + sorted(glob.glob('logs/trading.2026-04-1*.log'))
for path in log_files:
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = log_re.search(line)
                if m:
                    ts_str, asset, direction, p, xgb, lstm, fus, price = m.groups()
                    ts_local = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                    ts = ts_local.replace(tzinfo=timezone.utc)
                    # Convert local PT to UTC by adding 7h
                    from datetime import timedelta
                    ts = ts + timedelta(hours=LOCAL_OFFSET_HOURS)
                    signals.append({
                        'ts': ts, 'asset': asset, 'direction': direction,
                        'p': float(p), 'xgb': float(xgb), 'lstm': float(lstm),
                        'fus': float(fus), 'price': int(price),
                    })
    except FileNotFoundError:
        pass
print(f"Parsed {len(signals)} signals from {len(log_files)} log files")

# Match each trade to its triggering signal (closest preceding within 30s)
matched = []
unmatched = 0
for t in trades:
    candidates = [
        s for s in signals
        if s['asset'] == t['asset']
        and abs((s['ts'] - t['ts']).total_seconds()) < 30
    ]
    if not candidates:
        unmatched += 1
        continue
    best = min(candidates, key=lambda s: abs((s['ts'] - t['ts']).total_seconds()))
    matched.append({**t, 'xgb': best['xgb'], 'lstm': best['lstm'], 'fus': best['fus']})

print(f"Settled trades: {len(trades)}, matched to log: {len(matched)}, unmatched: {unmatched}")
print()

# Per-asset config
with open('config/trading.json') as f:
    cfg = json.load(f)
asset_thresh = {}
asset_mlw = {}
for a in cfg['assets']:
    # Pick weekday/weekend per actual trade time? For simplicity use ensemble base.
    ens = cfg['assets'][a]['ensemble']
    asset_thresh[a] = ens['threshold']
    asset_mlw[a] = ens.get('ml_weight', 0.1)

# Simulate
sim = defaultdict(lambda: {
    'count': 0, 'real_pnl': 0.0, 'real_wins': 0,
    'kept': 0, 'kept_pnl': 0.0, 'kept_wins': 0,
    'filtered': 0, 'filt_pnl_avoided': 0.0,
    'flipped': 0, 'flip_new_pnl': 0.0,
})

for t in matched:
    asset = t['asset']
    thresh = asset_thresh[asset]
    ml_w = asset_mlw[asset]
    dyn_p = dyn_blend(t['xgb'], t['lstm'], t['fus'], ml_w)

    s = sim[asset]
    s['count'] += 1
    s['real_pnl'] += t['pnl']
    if t['pnl'] > 0:
        s['real_wins'] += 1

    same_dir = (
        (t['direction'] == 'BULLISH' and dyn_p >= thresh) or
        (t['direction'] == 'BEARISH' and dyn_p <= 1 - thresh)
    )
    opp_dir = (
        (t['direction'] == 'BULLISH' and dyn_p <= 1 - thresh) or
        (t['direction'] == 'BEARISH' and dyn_p >= thresh)
    )

    if same_dir:
        s['kept'] += 1
        s['kept_pnl'] += t['pnl']
        if t['pnl'] > 0:
            s['kept_wins'] += 1
    elif opp_dir:
        s['flipped'] += 1
        # Flipped: take opposite side at price (100 - original_price)
        new_cost = t['contracts'] * (100 - t['price']) / 100.0
        if t['pnl'] > 0:
            # Original won -> opposite side loses its cost
            new_pnl = -new_cost
        else:
            # Original lost -> opposite side wins
            new_pnl = t['contracts'] * 1.0 - new_cost
        s['flip_new_pnl'] += new_pnl
    else:
        s['filtered'] += 1
        s['filt_pnl_avoided'] += -t['pnl']

# Print per-asset table
print(f"{'Asset':<6} {'N':>4} {'Real PnL':>10} {'WR%':>6} | {'Kept':>4} {'Filt':>4} {'Flip':>4} | {'Sim PnL':>10} {'Delta':>9}")
print('-' * 80)
total_real = 0.0
total_sim = 0.0
for asset in sorted(sim):
    s = sim[asset]
    sim_pnl = s['kept_pnl'] + s['flip_new_pnl']  # filtered = 0
    delta = sim_pnl - s['real_pnl']
    wr = 100 * s['real_wins'] / s['count'] if s['count'] else 0
    total_real += s['real_pnl']
    total_sim += sim_pnl
    print(f"{asset:<6} {s['count']:>4} ${s['real_pnl']:>+9.2f} {wr:>5.1f}% | {s['kept']:>4} {s['filtered']:>4} {s['flipped']:>4} | ${sim_pnl:>+9.2f} ${delta:>+8.2f}")
print('-' * 80)
n = sum(s['count'] for s in sim.values())
kept = sum(s['kept'] for s in sim.values())
filt = sum(s['filtered'] for s in sim.values())
flip = sum(s['flipped'] for s in sim.values())
print(f"{'TOTAL':<6} {n:>4} ${total_real:>+9.2f}        | {kept:>4} {filt:>4} {flip:>4} | ${total_sim:>+9.2f} ${total_sim-total_real:>+8.2f}")

# Diagnostic: how often does dyn_blend agree with xgb in direction
agree = sum(1 for t in matched if (t['direction'] == 'BULLISH' and dyn_blend(t['xgb'], t['lstm'], t['fus'], asset_mlw[t['asset']]) >= 0.5) or (t['direction'] == 'BEARISH' and dyn_blend(t['xgb'], t['lstm'], t['fus'], asset_mlw[t['asset']]) < 0.5))
print(f"\nDirectional agreement with XGB-only: {agree}/{len(matched)} = {100*agree/len(matched):.1f}%")
