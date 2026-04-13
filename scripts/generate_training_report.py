"""Generate a training report from the latest model metadata files.

Reads models/{ASSET}_xgb_features.json for each asset and produces
a timestamped training report in output/training_reports/.

Can be run after any training pipeline (walk_forward_select,
walk_forward_full, etc.) to persist the results.

Usage:
    python scripts/generate_training_report.py
    python scripts/generate_training_report.py --assets BTC,ETH,SOL,XRP
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "output" / "training_reports"
CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"


def generate_report(assets):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    tag = ts.strftime("%Y-%m-%d_%H%M")

    report = {
        "generated_at": ts.isoformat(),
        "assets": {},
        "summary": {},
    }

    print(f"{'='*70}")
    print(f"  TRAINING REPORT — {tag}")
    print(f"{'='*70}")

    total_pnl = 0
    total_wr = 0
    n_assets = 0

    for asset in assets:
        meta_path = MODELS_DIR / f"{asset}_xgb_features.json"
        if not meta_path.exists():
            print(f"  {asset}: no metadata found, skipping")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Read config for execution params
        config_ens = {}
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            config_ens = cfg.get("assets", {}).get(asset, {}).get("ensemble", {})
        except Exception:
            pass

        asset_report = {
            "n_features": meta.get("n_features"),
            "features": meta.get("features"),
            "training_method": meta.get("training_method", "unknown"),
            "best_params": meta.get("best_params", {}),
            "avg_oos_pnl": meta.get("best_avg_oos_pnl"),
            "avg_oos_wr": meta.get("best_avg_oos_wr"),
            "avg_trades_per_fold": meta.get("avg_trades_per_fold"),
            "n_folds": meta.get("n_folds"),
            "min_dm": meta.get("min_dm", config_ens.get("min_dm", 2)),
            "threshold": meta.get("threshold", config_ens.get("threshold")),
            "max_price": meta.get("max_price", config_ens.get("max_price_cents")),
            "has_lstm_p": "lstm_p" in meta.get("features", []),
        }
        report["assets"][asset] = asset_report

        pnl = asset_report["avg_oos_pnl"] or 0
        wr = asset_report["avg_oos_wr"] or 0
        total_pnl += pnl
        total_wr += wr
        n_assets += 1

        print(f"\n  {asset}:")
        print(f"    Method:     {asset_report['training_method']}")
        print(f"    Features:   {asset_report['n_features']} ({'stacked' if asset_report['has_lstm_p'] else 'XGB only'})")
        print(f"    OOS PnL:    ${pnl:+.2f}")
        print(f"    OOS WR:     {wr:.1f}%")
        if asset_report.get("avg_trades_per_fold"):
            print(f"    Trades/fold: {asset_report['avg_trades_per_fold']:.0f}")
        print(f"    Folds:      {asset_report['n_folds']}")
        print(f"    min_dm:     {asset_report['min_dm']}")
        print(f"    threshold:  {asset_report['threshold']}")
        print(f"    max_price:  {asset_report['max_price']}c")
        if asset_report["best_params"]:
            print(f"    XGB params:")
            for k, v in asset_report["best_params"].items():
                print(f"      {k}: {v}")

    report["summary"] = {
        "total_avg_oos_pnl": round(total_pnl, 2),
        "avg_wr": round(total_wr / n_assets, 1) if n_assets else 0,
        "n_assets": n_assets,
    }

    print(f"\n  {'='*60}")
    print(f"  TOTAL: ${total_pnl:+.2f} avg OOS PnL, {total_wr/n_assets:.1f}% avg WR" if n_assets else "  No assets")
    print(f"  {'='*60}")

    # Save report
    report_path = REPORTS_DIR / f"training_report_{tag}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {report_path}")

    # Also save as latest
    latest_path = REPORTS_DIR / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {latest_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate training report")
    parser.add_argument("--assets", default="BTC,ETH,SOL,XRP,HYPE,BNB,DOGE")
    args = parser.parse_args()
    assets = [a.strip().upper() for a in args.assets.split(",")]
    generate_report(assets)


if __name__ == "__main__":
    main()
