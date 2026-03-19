"""
Backtest reporter.
Prints summary statistics and exports detailed CSV.
"""
import csv
from collections import defaultdict
from pathlib import Path

from backtester.simulator import WindowResult


def print_report(results: list[WindowResult], asset: str) -> None:
    total = len(results)
    if total == 0:
        print(f"=== {asset} Backtest ===")
        print("No windows to report.")
        return

    traded = [r for r in results if r.predicted_direction != "NONE"]
    skipped = total - len(traded)
    correct = [r for r in traded if r.correct]
    accuracy = len(correct) / len(traded) * 100 if traded else 0.0

    avg_minute = (
        sum(r.decision_minute for r in traded) / len(traded)
        if traded else 0.0
    )

    first_date = results[0].window_start.strftime("%Y-%m-%d")
    last_date = results[-1].window_start.strftime("%Y-%m-%d")
    days = len(set(r.window_start.strftime("%Y-%m-%d") for r in results))

    print()
    print(f"=== {asset} Backtest ({days} days: {first_date} -> {last_date}) ===")
    print(f"Windows:      {total}")
    print(f"Traded:       {len(traded)}  ({len(traded)/total*100:.1f}%)")
    print(f"Skipped:      {skipped}")
    if traded:
        print(f"Accuracy:     {accuracy:.1f}%  ({len(correct)}/{len(traded)})")
        print(f"Avg decision: minute {avg_minute:.1f}")
    print()

    # --- By direction ---
    if traded:
        bull_traded = [r for r in traded if r.predicted_direction == "BULLISH"]
        bear_traded = [r for r in traded if r.predicted_direction == "BEARISH"]
        bull_correct = sum(1 for r in bull_traded if r.correct)
        bear_correct = sum(1 for r in bear_traded if r.correct)

        print("By prediction direction:")
        if bull_traded:
            print(f"  BULLISH: {bull_correct}/{len(bull_traded)} = {bull_correct/len(bull_traded)*100:.1f}%")
        if bear_traded:
            print(f"  BEARISH: {bear_correct}/{len(bear_traded)} = {bear_correct/len(bear_traded)*100:.1f}%")
        print()

    # --- By decision minute ---
    if traded:
        minute_counts: dict[int, list[WindowResult]] = defaultdict(list)
        for r in traded:
            minute_counts[r.decision_minute].append(r)

        print("By decision minute:")
        for m in sorted(minute_counts.keys()):
            rs = minute_counts[m]
            mc = sum(1 for r in rs if r.correct)
            print(f"  min {m:2d}: {len(rs):4d} trades, {mc}/{len(rs)} correct ({mc/len(rs)*100:.0f}%)")
        print()

    # --- By confidence bucket ---
    if traded:
        buckets = [(0.6, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.01)]
        print("By confidence bucket:")
        for lo, hi in buckets:
            bucket = [r for r in traded if lo <= r.confidence < hi]
            if bucket:
                bc = sum(1 for r in bucket if r.correct)
                label = f"[{lo:.0%}-{hi:.0%})" if hi < 1.0 else f"[{lo:.0%}-100%]"
                print(f"  {label}: {bc}/{len(bucket)} = {bc/len(bucket)*100:.1f}%")
        print()

    # --- Daily summary ---
    if traded:
        daily: dict[str, list[WindowResult]] = defaultdict(list)
        for r in traded:
            daily[r.window_start.strftime("%Y-%m-%d")].append(r)

        print("Daily accuracy (traded days only):")
        for date in sorted(daily.keys()):
            rs = daily[date]
            dc = sum(1 for r in rs if r.correct)
            print(f"  {date}: {dc}/{len(rs)} = {dc/len(rs)*100:.0f}%")
        print()


def export_csv(results: list[WindowResult], path: Path) -> None:
    """Export per-window detail to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_start", "actual_direction", "predicted_direction",
            "decision_minute", "confidence", "score", "correct",
            "price_open", "price_close",
        ])
        for r in results:
            writer.writerow([
                r.window_start.isoformat(),
                r.actual_direction,
                r.predicted_direction,
                r.decision_minute,
                f"{r.confidence:.4f}",
                f"{r.score:.2f}",
                r.correct,
                f"{r.price_open:.2f}",
                f"{r.price_close:.2f}",
            ])
    print(f"Exported {len(results)} rows to {path}")
