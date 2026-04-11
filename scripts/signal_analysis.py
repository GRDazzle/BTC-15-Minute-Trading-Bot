"""Analyze signal_log.csv to find exploitable patterns.

Slices the signal data by confidence, direction consistency, ML/Fusion
agreement, price band, hour, DM, and various combos. Reports WR and
simulated PnL (10 contracts normalized) for each slice.
"""
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone, date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET = {"BTC", "ETH", "SOL", "XRP"}
ASSET_SERIES = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M", "XRP": "KXXRP15M"}
POLLS_DIR = PROJECT_ROOT / "data" / "kalshi_polls"

monday_pt = datetime(2026, 4, 7, 7, 0, 0, tzinfo=timezone.utc)
monday_date = date(2026, 4, 7)


def window_to_event(asset, window_id):
    dt = datetime.strptime(window_id, "%Y%m%d_%H%M")
    month_abbr = dt.strftime("%b").upper()
    series = ASSET_SERIES[asset]
    return f"{series}-{dt.strftime('%y')}{month_abbr}{dt.strftime('%d%H%M')}"


def load_outcomes():
    outcomes = {}
    for asset, series in ASSET_SERIES.items():
        series_dir = POLLS_DIR / series
        if not series_dir.exists():
            continue
        for jsonl_file in sorted(series_dir.glob("*.jsonl")):
            try:
                file_date = datetime.strptime(jsonl_file.name[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            if file_date < monday_date:
                continue
            with open(jsonl_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if entry.get("type") == "outcome":
                        outcomes[entry.get("event_ticker", "")] = entry.get("outcome", "")
    return outcomes


def load_signals():
    windows = defaultdict(list)
    with open(PROJECT_ROOT / "output" / "signal_log.csv") as f:
        for row in csv.DictReader(f):
            asset = row.get("asset", "").strip()
            if asset not in TARGET:
                continue
            try:
                ts = datetime.fromisoformat(row["timestamp"])
            except Exception:
                continue
            if ts < monday_pt:
                continue
            key = (asset, row["window_id"])
            windows[key].append(row)
    return windows


def build_features(windows, outcomes):
    features = []
    for (asset, wid), sigs in windows.items():
        event_ticker = window_to_event(asset, wid)
        outcome = outcomes.get(event_ticker)
        if outcome is None:
            continue

        all_ensemble_p = []
        all_ml_p = []
        all_fusion_p = []
        directions = []
        prices = []
        dms = []

        for s in sigs:
            try:
                ep = float(s.get("ensemble_p", 0))
                mp = float(s.get("ml_p", 0))
                fp = float(s.get("fusion_p", 0))
                dm = int(s.get("dm", 0))
                price = int(s.get("entry_price", 0) or 0)
                direction = s.get("direction", "NONE")
            except Exception:
                continue
            all_ensemble_p.append(ep)
            all_ml_p.append(mp)
            all_fusion_p.append(fp)
            if direction not in ("NONE", ""):
                directions.append(direction)
                prices.append(price)
                dms.append(dm)

        if not directions:
            continue

        bull_count = sum(1 for d in directions if d == "BULLISH")
        bear_count = len(directions) - bull_count
        majority_dir = "BULLISH" if bull_count >= bear_count else "BEARISH"
        majority_pct = max(bull_count, bear_count) / len(directions)
        flips = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i - 1])

        if majority_dir == "BULLISH":
            relevant_p = [p for p in all_ensemble_p if p > 0.5]
        else:
            relevant_p = [1 - p for p in all_ensemble_p if p < 0.5]
        avg_confidence = sum(relevant_p) / len(relevant_p) if relevant_p else 0.5
        max_confidence = max(relevant_p) if relevant_p else 0.5

        avg_ml_p = sum(all_ml_p) / len(all_ml_p)
        avg_fusion_p = sum(all_fusion_p) / len(all_fusion_p)
        ml_dir = "BULLISH" if avg_ml_p > 0.5 else "BEARISH"
        fusion_dir = "BULLISH" if avg_fusion_p > 0.5 else "BEARISH"
        ml_fusion_agree = ml_dir == fusion_dir
        ml_strength = abs(avg_ml_p - 0.5) * 2
        fusion_strength = abs(avg_fusion_p - 0.5) * 2

        try:
            wdt = datetime.strptime(wid, "%Y%m%d_%H%M")
            hour_utc = wdt.hour
        except Exception:
            hour_utc = -1

        side = "yes" if majority_dir == "BULLISH" else "no"
        won = side == outcome

        features.append({
            "asset": asset,
            "window_id": wid,
            "outcome": outcome,
            "majority_dir": majority_dir,
            "won": won,
            "n_actionable": len(directions),
            "n_total": len(sigs),
            "pct_actionable": len(directions) / len(sigs) if sigs else 0,
            "majority_pct": majority_pct,
            "flips": flips,
            "avg_confidence": avg_confidence,
            "max_confidence": max_confidence,
            "ml_fusion_agree": ml_fusion_agree,
            "ml_strength": ml_strength,
            "fusion_strength": fusion_strength,
            "first_price": prices[0] if prices else 0,
            "first_dm": dms[0] if dms else 0,
            "min_price": min(prices) if prices else 0,
            "hour_utc": hour_utc,
        })
    return features


def report(label, subset):
    if not subset:
        print(f"  {label:<55} (no data)")
        return
    w = sum(1 for x in subset if x["won"])
    n = len(subset)
    wr = w / n * 100
    pnl = 0
    for x in subset:
        p = x["first_price"]
        if p < 55 or p > 85:
            p = 70
        cost = (p / 100 + 0.02) * 10
        if x["won"]:
            pnl += 10 - cost
        else:
            pnl -= cost
    print(f"  {label:<55} n={n:>4}  WR={wr:>5.1f}%  PnL=${pnl:>+8.2f}")


def main():
    print("Loading data...")
    outcomes = load_outcomes()
    windows = load_signals()
    features = build_features(windows, outcomes)
    print(f"Outcomes: {len(outcomes)}, Signal windows: {len(windows)}, Features built: {len(features)}")
    print()

    print("=" * 80)
    print("SIGNAL ANALYSIS: Monday PT to now (BTC ETH SOL XRP)")
    print("Normalized: 10 contracts, first_price at entry, 2c fee")
    print("=" * 80)

    print("\n=== 1. CONFIDENCE LEVEL (avg ensemble distance from 0.5) ===")
    for lo, hi, label in [(0.0, 0.55, "Low (<55%)"), (0.55, 0.65, "Medium (55-65%)"),
                          (0.65, 0.75, "High (65-75%)"), (0.75, 1.0, "Very high (75%+)")]:
        report(f"Confidence {label}", [w for w in features if lo <= w["avg_confidence"] < hi])

    print("\n=== 2. MAX CONFIDENCE (peak signal in window) ===")
    for lo, hi, label in [(0.5, 0.6, "50-60%"), (0.6, 0.7, "60-70%"), (0.7, 0.8, "70-80%"),
                          (0.8, 0.9, "80-90%"), (0.9, 1.0, "90%+")]:
        report(f"Max confidence {label}", [w for w in features if lo <= w["max_confidence"] < hi])

    print("\n=== 3. DIRECTION CONSISTENCY (% of checkpoints agreeing) ===")
    for lo, hi, label in [(0.0, 0.6, "<60%"), (0.6, 0.8, "60-80%"), (0.8, 0.95, "80-95%"),
                          (0.95, 1.01, "95-100%")]:
        report(f"Consistency {label}", [w for w in features if lo <= w["majority_pct"] < hi])

    print("\n=== 4. SIGNAL FLIPS (direction changed N times in window) ===")
    for flips in [0, 1, 2, 3]:
        label = f"{flips} flips" if flips < 3 else "3+ flips"
        if flips < 3:
            subset = [w for w in features if w["flips"] == flips]
        else:
            subset = [w for w in features if w["flips"] >= 3]
        report(label, subset)

    print("\n=== 5. ML vs FUSION AGREEMENT ===")
    report("ML + Fusion AGREE", [w for w in features if w["ml_fusion_agree"]])
    report("ML + Fusion DISAGREE", [w for w in features if not w["ml_fusion_agree"]])

    print("\n=== 6. ML STRENGTH (how far ML avg is from 0.5) ===")
    for lo, hi, label in [(0.0, 0.1, "Weak (<10%)"), (0.1, 0.2, "Moderate (10-20%)"),
                          (0.2, 0.4, "Strong (20-40%)"), (0.4, 1.0, "Very strong (40%+)")]:
        report(f"ML strength {label}", [w for w in features if lo <= w["ml_strength"] < hi])

    print("\n=== 7. FIRST DM (when signal first appeared) ===")
    for dm in range(2, 9):
        report(f"First actionable at dm {dm}", [w for w in features if w["first_dm"] == dm])

    print("\n=== 8. PRICE BAND (first actionable price) ===")
    for lo, hi in [(0, 55), (55, 65), (65, 75), (75, 85), (85, 100)]:
        report(f"Price {lo}-{hi}c", [w for w in features if lo <= w["first_price"] < hi])

    print("\n=== 9. HOUR (UTC) ===")
    for h in range(0, 24):
        subset = [w for w in features if w["hour_utc"] == h]
        if len(subset) >= 5:
            report(f"{h:02d}:00 UTC ({(h-7)%24:02d}:00 PT)", subset)

    print("\n=== 10. PER ASSET ===")
    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        report(asset, [w for w in features if w["asset"] == asset])

    print("\n=== 11. COMBO: High confidence + ML/Fusion agree ===")
    for thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
        report(
            f"Confidence >= {thresh:.0%} AND agree",
            [w for w in features if w["avg_confidence"] >= thresh and w["ml_fusion_agree"]],
        )

    print("\n=== 12. COMBO: Zero flips + high confidence ===")
    for thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
        report(
            f"0 flips + confidence >= {thresh:.0%}",
            [w for w in features if w["flips"] == 0 and w["avg_confidence"] >= thresh],
        )

    print("\n=== 13. ACTIONABLE DENSITY (what % of checkpoints are actionable) ===")
    for lo, hi, label in [(0.0, 0.3, "<30%"), (0.3, 0.5, "30-50%"), (0.5, 0.7, "50-70%"),
                          (0.7, 0.9, "70-90%"), (0.9, 1.01, "90%+")]:
        report(f"Density {label}", [w for w in features if lo <= w["pct_actionable"] < hi])

    print("\n=== 14. ML STRONG + FUSION direction ===")
    report("ML strong (>0.3) + Fusion weak (<0.1)", [w for w in features if w["ml_strength"] > 0.3 and w["fusion_strength"] < 0.1])
    report("ML strong (>0.3) + Fusion agrees", [w for w in features if w["ml_strength"] > 0.3 and w["ml_fusion_agree"]])
    report("ML strong (>0.3) + Fusion disagrees", [w for w in features if w["ml_strength"] > 0.3 and not w["ml_fusion_agree"]])

    print("\n=== 15. PRICE BAND + CONFIDENCE combo ===")
    for price_lo, price_hi in [(55, 65), (65, 75), (75, 85)]:
        for conf in [0.60, 0.70, 0.80]:
            report(
                f"Price {price_lo}-{price_hi}c + confidence >= {conf:.0%}",
                [w for w in features if price_lo <= w["first_price"] < price_hi and w["avg_confidence"] >= conf],
            )

    print("\n=== 16. ASSET + CONFIDENCE ===")
    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        for conf in [0.60, 0.65, 0.70, 0.75]:
            report(
                f"{asset} confidence >= {conf:.0%}",
                [w for w in features if w["asset"] == asset and w["avg_confidence"] >= conf],
            )

    print("\n=== 17. WINNING COMBOS (filter for WR > 55% AND n >= 10) ===")
    print("  Scanning all 2-way combos for profitable filters...\n")
    combos_found = []
    # Confidence x price band
    for conf in [0.60, 0.65, 0.70, 0.75, 0.80]:
        for price_lo, price_hi in [(55, 65), (65, 75), (75, 85)]:
            subset = [w for w in features if w["avg_confidence"] >= conf and price_lo <= w["first_price"] < price_hi]
            if len(subset) >= 10:
                wr = sum(1 for x in subset if x["won"]) / len(subset) * 100
                if wr > 55:
                    combos_found.append((f"conf>={conf:.0%} + price {price_lo}-{price_hi}", len(subset), wr))

    # Confidence x ML/Fusion agree x price band
    for conf in [0.60, 0.65, 0.70, 0.75]:
        for price_lo, price_hi in [(55, 65), (65, 75), (75, 85)]:
            subset = [w for w in features if w["avg_confidence"] >= conf and w["ml_fusion_agree"] and price_lo <= w["first_price"] < price_hi]
            if len(subset) >= 10:
                wr = sum(1 for x in subset if x["won"]) / len(subset) * 100
                if wr > 55:
                    combos_found.append((f"conf>={conf:.0%} + agree + price {price_lo}-{price_hi}", len(subset), wr))

    # Asset x confidence
    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        for conf in [0.60, 0.65, 0.70, 0.75, 0.80]:
            subset = [w for w in features if w["asset"] == asset and w["avg_confidence"] >= conf]
            if len(subset) >= 10:
                wr = sum(1 for x in subset if x["won"]) / len(subset) * 100
                if wr > 55:
                    combos_found.append((f"{asset} + conf>={conf:.0%}", len(subset), wr))

    # 0 flips x asset
    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        subset = [w for w in features if w["asset"] == asset and w["flips"] == 0]
        if len(subset) >= 10:
            wr = sum(1 for x in subset if x["won"]) / len(subset) * 100
            if wr > 55:
                combos_found.append((f"{asset} + 0 flips", len(subset), wr))

    # 0 flips x confidence x asset
    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        for conf in [0.65, 0.70, 0.75]:
            subset = [w for w in features if w["asset"] == asset and w["flips"] == 0 and w["avg_confidence"] >= conf]
            if len(subset) >= 10:
                wr = sum(1 for x in subset if x["won"]) / len(subset) * 100
                if wr > 55:
                    combos_found.append((f"{asset} + 0 flips + conf>={conf:.0%}", len(subset), wr))

    # Density x confidence
    for dens_lo in [0.5, 0.7, 0.9]:
        for conf in [0.60, 0.65, 0.70, 0.75]:
            subset = [w for w in features if w["pct_actionable"] >= dens_lo and w["avg_confidence"] >= conf]
            if len(subset) >= 10:
                wr = sum(1 for x in subset if x["won"]) / len(subset) * 100
                if wr > 55:
                    combos_found.append((f"density>={dens_lo:.0%} + conf>={conf:.0%}", len(subset), wr))

    combos_found.sort(key=lambda x: -x[2])
    for label, n, wr in combos_found[:25]:
        pnl_marker = "+" if wr > 58 else "~"
        print(f"  {pnl_marker} {label:<55} n={n:>4}  WR={wr:>5.1f}%")


if __name__ == "__main__":
    main()
