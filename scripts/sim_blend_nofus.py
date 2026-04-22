"""Simulate dynamic blend variants with NO fusion."""
import re, csv, json, glob
from collections import defaultdict
from datetime import datetime, timezone, timedelta

CUTOFF = datetime(2026, 4, 15, 22, 0, tzinfo=timezone.utc)
DYNAMIC_K = 4.5
LSTM_MIN_W, LSTM_MAX_W = 0.10, 0.40
XGB_MAX_W = 0.60
LOCAL_OFFSET_HOURS = 7


def blend_v1_renorm(xgb, lstm, ml_w):
    """Drop fusion, renormalize XGB+LSTM weights so they sum to 1."""
    xgb_conf = abs(xgb - 0.5) * 2
    w_x = ml_w + (XGB_MAX_W - ml_w) * (xgb_conf ** DYNAMIC_K)
    lstm_conf = abs(lstm - 0.5) * 2
    w_l = LSTM_MIN_W + (LSTM_MAX_W - LSTM_MIN_W) * (lstm_conf ** DYNAMIC_K)
    total = w_x + w_l
    return (w_x * xgb + w_l * lstm) / total


def blend_v2_complement(xgb, lstm, ml_w):
    """XGB-dominant: dyn_xgb_w as before, LSTM gets the rest (1 - dyn_xgb_w)."""
    xgb_conf = abs(xgb - 0.5) * 2
    w_x = ml_w + (XGB_MAX_W - ml_w) * (xgb_conf ** DYNAMIC_K)
    return w_x * xgb + (1 - w_x) * lstm


def blend_v3_equal(xgb, lstm, ml_w):
    """Simple 50/50."""
    return 0.5 * xgb + 0.5 * lstm


def blend_v4_xgb_heavy(xgb, lstm, ml_w):
    """70/30 XGB-favored."""
    return 0.7 * xgb + 0.3 * lstm


def blend_v5_max_conf(xgb, lstm, ml_w):
    """Pick whichever model is more confident (further from 0.5)."""
    if abs(xgb - 0.5) >= abs(lstm - 0.5):
        return xgb
    return lstm


VARIANTS = [
    ("Renormalize XGB+LSTM", blend_v1_renorm),
    ("XGB + (1-w)*LSTM", blend_v2_complement),
    ("50/50 XGB+LSTM", blend_v3_equal),
    ("70/30 XGB+LSTM", blend_v4_xgb_heavy),
    ("Max-confidence wins", blend_v5_max_conf),
]


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
            'ts': ts, 'asset': row['asset'], 'window_id': row['window_id'],
            'direction': row['direction'], 'price': int(row['price_cents']),
            'contracts': int(row['contracts']),
            'pnl': float(row['pnl']),
        })

# Load signals from all logs
log_re = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[(\w+)\] Ensemble (BULLISH|BEARISH).*p=([\d.]+) \(xgb=([\d.]+) lstm=([\d.]+) fus=([\d.]+)\)'
)
signals = []
for path in ['logs/trading.log'] + sorted(glob.glob('logs/trading.2026-04-1*.log')):
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = log_re.search(line)
                if m:
                    ts_str, asset, direction, p, xgb, lstm, fus = m.groups()
                    ts_local = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    ts = ts_local + timedelta(hours=LOCAL_OFFSET_HOURS)
                    signals.append({
                        'ts': ts, 'asset': asset, 'direction': direction,
                        'xgb': float(xgb), 'lstm': float(lstm),
                    })
    except FileNotFoundError:
        pass

# Match
matched = []
for t in trades:
    cands = [s for s in signals if s['asset'] == t['asset']
             and abs((s['ts'] - t['ts']).total_seconds()) < 30]
    if cands:
        best = min(cands, key=lambda s: abs((s['ts'] - t['ts']).total_seconds()))
        matched.append({**t, 'xgb': best['xgb'], 'lstm': best['lstm']})

# Per-asset config
with open('config/trading.json') as f:
    cfg = json.load(f)
asset_thresh = {a: cfg['assets'][a]['ensemble']['threshold'] for a in cfg['assets']}
asset_mlw = {a: cfg['assets'][a]['ensemble'].get('ml_weight', 0.1) for a in cfg['assets']}

real_total = sum(t['pnl'] for t in matched)
real_count = len(matched)
real_wins = sum(1 for t in matched if t['pnl'] > 0)
print(f"REAL (XGB-only): {real_count} trades, {real_wins} wins ({100*real_wins/real_count:.1f}% WR), ${real_total:+.2f}")
print()

# Run each variant
print(f"{'Variant':<25} {'Trades':>7} {'Kept':>5} {'Filt':>5} {'Flip':>5} {'WR%':>6} {'Sim PnL':>10} {'D':>9}")
print('-' * 90)
for name, blend_fn in VARIANTS:
    sim_pnl = 0.0
    kept = filt = flip = 0
    sim_wins = sim_count = 0
    for t in matched:
        thresh = asset_thresh[t['asset']]
        ml_w = asset_mlw[t['asset']]
        dyn_p = blend_fn(t['xgb'], t['lstm'], ml_w)
        same = (t['direction'] == 'BULLISH' and dyn_p >= thresh) or \
               (t['direction'] == 'BEARISH' and dyn_p <= 1 - thresh)
        opp = (t['direction'] == 'BULLISH' and dyn_p <= 1 - thresh) or \
              (t['direction'] == 'BEARISH' and dyn_p >= thresh)
        if same:
            kept += 1
            sim_pnl += t['pnl']
            sim_count += 1
            if t['pnl'] > 0:
                sim_wins += 1
        elif opp:
            flip += 1
            new_cost = t['contracts'] * (100 - t['price']) / 100.0
            if t['pnl'] > 0:
                new_pnl = -new_cost
            else:
                new_pnl = t['contracts'] * 1.0 - new_cost
            sim_pnl += new_pnl
            sim_count += 1
            if new_pnl > 0:
                sim_wins += 1
        else:
            filt += 1
    wr = 100 * sim_wins / sim_count if sim_count else 0
    delta = sim_pnl - real_total
    print(f"{name:<25} {sim_count:>7} {kept:>5} {filt:>5} {flip:>5} {wr:>5.1f}% ${sim_pnl:>+9.2f} ${delta:>+8.2f}")

# Also break down the BEST variant per asset
print()
print("=== Per-asset breakdown for each variant ===")
for name, blend_fn in VARIANTS:
    print(f"\n[{name}]")
    per_asset = defaultdict(lambda: {'real': 0.0, 'sim': 0.0, 'count': 0, 'kept': 0, 'wins': 0, 'sim_count': 0})
    for t in matched:
        a = t['asset']
        per_asset[a]['real'] += t['pnl']
        per_asset[a]['count'] += 1
        thresh = asset_thresh[a]
        ml_w = asset_mlw[a]
        dyn_p = blend_fn(t['xgb'], t['lstm'], ml_w)
        same = (t['direction'] == 'BULLISH' and dyn_p >= thresh) or \
               (t['direction'] == 'BEARISH' and dyn_p <= 1 - thresh)
        opp = (t['direction'] == 'BULLISH' and dyn_p <= 1 - thresh) or \
              (t['direction'] == 'BEARISH' and dyn_p >= thresh)
        if same:
            per_asset[a]['sim'] += t['pnl']
            per_asset[a]['kept'] += 1
            per_asset[a]['sim_count'] += 1
            if t['pnl'] > 0:
                per_asset[a]['wins'] += 1
        elif opp:
            new_cost = t['contracts'] * (100 - t['price']) / 100.0
            new_pnl = -new_cost if t['pnl'] > 0 else t['contracts'] - new_cost
            per_asset[a]['sim'] += new_pnl
            per_asset[a]['sim_count'] += 1
            if new_pnl > 0:
                per_asset[a]['wins'] += 1
    for a in sorted(per_asset):
        d = per_asset[a]
        wr = 100 * d['wins'] / d['sim_count'] if d['sim_count'] else 0
        delta = d['sim'] - d['real']
        print(f"  {a:<5} N={d['count']:>3} Real ${d['real']:>+7.2f}  Sim ${d['sim']:>+7.2f} (n={d['sim_count']}, WR {wr:.0f}%)  D ${delta:>+7.2f}")
