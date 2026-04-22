"""Find time-of-day patterns in PnL across all available trade history."""
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

archives = [
    ("output/trades.csv", "Today"),
    ("output/v9_archive_2026-04-10/trades.csv", "v9"),
    ("output/v10_archive_2026-04-12/trades.csv", "v10"),
    ("output/v10_stacked_archive_2026-04-15/trades.csv", "v10s"),
]

per_hour = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0, "by_archive": defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})})
per_window_min = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})

for path, name in archives:
    p = PROJECT_ROOT / path
    if not p.exists():
        continue
    with open(p) as f:
        for row in csv.DictReader(f):
            if not row.get("outcome") or not row.get("pnl"):
                continue
            try:
                ts = datetime.fromisoformat(row["timestamp"])
                we = datetime.fromisoformat(row["window_id"].replace("Z", "+00:00"))
                pnl = float(row["pnl"])
            except (ValueError, KeyError):
                continue
            hr = ts.hour
            wm = we.minute
            ph = per_hour[hr]
            ph["n"] += 1
            ph["pnl"] += pnl
            if pnl > 0:
                ph["wins"] += 1
            ph["by_archive"][name]["n"] += 1
            ph["by_archive"][name]["pnl"] += pnl
            if pnl > 0:
                ph["by_archive"][name]["wins"] += 1
            pwm = per_window_min[wm]
            pwm["n"] += 1
            pwm["pnl"] += pnl
            if pnl > 0:
                pwm["wins"] += 1

print("=" * 90)
print("PER UTC HOUR \u2014 combined across all 4 archives".replace("\u2014", "-"))
print("=" * 90)
print(f"{'Hour':>4} {'N':>5} {'WR%':>6} {'PnL':>10} {'PnL/trade':>10}  archive split")
print("-" * 90)
for hr in sorted(per_hour):
    s = per_hour[hr]
    wr = 100 * s["wins"] / s["n"] if s["n"] else 0
    avg = s["pnl"] / s["n"] if s["n"] else 0
    arch_str = " ".join(
        f"{n}:{sa['pnl']:+.0f}/{sa['n']}" for n, sa in sorted(s["by_archive"].items()) if sa["n"] > 0
    )
    print(f"{hr:>4} {s['n']:>5} {wr:>5.1f}% ${s['pnl']:>+9.2f} ${avg:>+9.2f}  {arch_str}")

print()
print("=" * 90)
print("PER WINDOW-CLOSE MINUTE")
print("=" * 90)
print(f"{'Min':>4} {'N':>5} {'WR%':>6} {'PnL':>10} {'PnL/trade':>10}")
for wm in sorted(per_window_min):
    s = per_window_min[wm]
    wr = 100 * s["wins"] / s["n"] if s["n"] else 0
    avg = s["pnl"] / s["n"] if s["n"] else 0
    print(f":{wm:02d}  {s['n']:>5} {wr:>5.1f}% ${s['pnl']:>+9.2f} ${avg:>+9.2f}")

print()
print("=" * 90)
print("CONSISTENTLY BAD HOURS - negative PnL in EVERY archive that traded that hour (n>=5)")
print("=" * 90)
for hr in sorted(per_hour):
    s = per_hour[hr]
    archs_with_data = [(n, sa) for n, sa in s["by_archive"].items() if sa["n"] >= 5]
    if len(archs_with_data) < 2:
        continue
    if all(sa["pnl"] < 0 for _, sa in archs_with_data):
        wr = 100 * s["wins"] / s["n"]
        arch_str = " | ".join(f"{n}:{sa['pnl']:+.0f}/{sa['n']}" for n, sa in sorted(archs_with_data))
        print(f"  Hour {hr:02d}: total ${s['pnl']:+.2f} ({wr:.1f}% WR over {s['n']} trades) [{arch_str}]")

print()
print("=" * 90)
print("CONSISTENTLY GOOD HOURS - positive PnL in EVERY archive (n>=5)")
print("=" * 90)
for hr in sorted(per_hour):
    s = per_hour[hr]
    archs_with_data = [(n, sa) for n, sa in s["by_archive"].items() if sa["n"] >= 5]
    if len(archs_with_data) < 2:
        continue
    if all(sa["pnl"] > 0 for _, sa in archs_with_data):
        wr = 100 * s["wins"] / s["n"]
        arch_str = " | ".join(f"{n}:{sa['pnl']:+.0f}/{sa['n']}" for n, sa in sorted(archs_with_data))
        print(f"  Hour {hr:02d}: total ${s['pnl']:+.2f} ({wr:.1f}% WR over {s['n']} trades) [{arch_str}]")

print()
print("=" * 90)
print("WORST 5 HOURS / BEST 5 HOURS (by total PnL, n>=30)")
print("=" * 90)
ranked = sorted([(h, s) for h, s in per_hour.items() if s["n"] >= 30], key=lambda x: x[1]["pnl"])
print("\nWORST 5:")
for hr, s in ranked[:5]:
    wr = 100 * s["wins"] / s["n"]
    print(f"  Hour {hr:02d}: ${s['pnl']:+.2f} over {s['n']} trades ({wr:.1f}% WR)")
print("\nBEST 5:")
for hr, s in ranked[-5:]:
    wr = 100 * s["wins"] / s["n"]
    print(f"  Hour {hr:02d}: ${s['pnl']:+.2f} over {s['n']} trades ({wr:.1f}% WR)")
