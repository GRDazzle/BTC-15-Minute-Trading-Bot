"""
Tick-level signal accuracy backtester.

Uses real Binance aggTrades data (sub-second resolution) instead of 1m klines,
matching what the live system actually sees.

Usage:
  python scripts/backtest_ticks.py --asset BTC
  python scripts/backtest_ticks.py --asset BTC,ETH,SOL,XRP --days 30
  python scripts/backtest_ticks.py --asset BTC --days 7 --no-blackout
  python scripts/backtest_ticks.py --asset BTC --days 30 --ml
"""
import argparse
import json
import sys
from datetime import time as dt_time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader import load_fear_greed
from backtester.data_loader_ticks import load_aggtrades_multi, generate_tick_windows
from backtester.simulator import BacktestSimulator
from backtester.reporter import print_report, export_csv

from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine

DATA_DIR = PROJECT_ROOT / "data" / "aggtrades"
FG_CSV = PROJECT_ROOT / "data" / "historical" / "fear_greed.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "backtest_ticks"
CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"
MODEL_DIR = PROJECT_ROOT / "models"


def build_processors() -> list:
    """Instantiate signal processors with sweep-validated parameters."""
    return [
        SpikeDetectionProcessor(
            spike_threshold=0.003,
            velocity_threshold=0.0015,
            lookback_periods=20,
            min_confidence=0.55,
        ),
        TickVelocityProcessor(
            velocity_threshold_60s=0.001,
            velocity_threshold_30s=0.0007,
            min_ticks=5,
            min_confidence=0.55,
        ),
    ]


def load_blackout_windows() -> list[dict]:
    """Load blackout windows from config/trading.json."""
    windows = []
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        for bw in config.get("blackout_windows", []):
            h_s, m_s = bw["start_utc"].split(":")
            h_e, m_e = bw["end_utc"].split(":")
            windows.append({
                "start": dt_time(int(h_s), int(m_s)),
                "end": dt_time(int(h_e), int(m_e)),
            })
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return windows


def run_asset(
    asset: str,
    days: int | None,
    use_blackout: bool,
    use_ml: bool = False,
    min_dm: int = 0,
    ensemble: bool = False,
    ml_weight: float = 0.65,
    ens_threshold: float = 0.60,
) -> None:
    """Run tick-level backtest for one asset."""
    # Load tick data
    ticks = load_aggtrades_multi(DATA_DIR, asset, days=days)
    if not ticks:
        logger.error("No aggTrades data found for {}. Run download_binance_aggtrades.py first.", asset)
        return

    # Load Fear & Greed scores
    fg_scores = load_fear_greed(FG_CSV) if FG_CSV.exists() else {}

    # Generate tick windows
    windows = generate_tick_windows(ticks)
    if not windows:
        logger.error("No valid tick windows for {}", asset)
        return

    # Load blackout windows
    blackout = load_blackout_windows() if use_blackout else []
    if blackout:
        print(f"\nBlackout windows enabled: {len(blackout)} configured")
        for bw in blackout:
            print(f"  {bw['start'].strftime('%H:%M')}-{bw['end'].strftime('%H:%M')} UTC")

    # Build processors + fusion engine
    processors = build_processors()
    fusion_engine = SignalFusionEngine()

    # ML processor (needed for --ml or --ensemble)
    ml_processor = None
    if use_ml or ensemble:
        try:
            from core.strategy_brain.signal_processors.ml_processor import MLProcessor
            ml_processor = MLProcessor(
                asset=asset,
                model_dir=MODEL_DIR,
                confidence_threshold=0.60,
            )
            print(f"\nML model loaded for {asset}")
        except FileNotFoundError as e:
            print(f"\nML model not found for {asset}: {e}")
            if ensemble:
                print("Ensemble mode requires an ML model. Aborting.")
                return
            print("Falling back to fusion engine")
        except ImportError:
            print("\nxgboost not installed, falling back to fusion engine")
            if ensemble:
                return

    # Ensemble weights
    ensemble_weights = None
    if ensemble and ml_processor is not None:
        ensemble_weights = (ml_weight, ens_threshold)
        print(f"Ensemble mode: ml_weight={ml_weight:.2f}, threshold={ens_threshold:.2f}")

    # Run backtest
    simulator = BacktestSimulator(
        processors, fusion_engine,
        ml_processor=ml_processor,
        min_dm=min_dm,
        ensemble_weights=ensemble_weights,
    )
    results = simulator.run_ticks(windows, fg_scores, blackout_windows=blackout or None)

    # Report
    suffix = f"{asset.upper()}_ticks"
    if use_blackout:
        suffix += "_blackout"
    if ensemble and ml_processor is not None:
        suffix += f"_ensemble_w{ml_weight:.2f}_t{ens_threshold:.2f}"
    elif use_ml and ml_processor is not None:
        suffix += "_ml"
    label = f"{asset} (tick-level"
    if use_blackout:
        label += " + blackout"
    if ensemble and ml_processor is not None:
        label += f" + ensemble w={ml_weight:.2f} t={ens_threshold:.2f}"
    elif use_ml and ml_processor is not None:
        label += " + ML"
    label += ")"
    print_report(results, label)

    # Export CSV
    out_path = OUTPUT_DIR / f"{suffix}_backtest.csv"
    export_csv(results, out_path)


def main():
    # Configure loguru
    logger.remove()
    log_path = PROJECT_ROOT / "logs" / "backtest_ticks.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Use mode="w" to overwrite each run (avoids rotation rename errors on Windows)
    logger.add(
        str(log_path),
        mode="w",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="Tick-level signal accuracy backtester")
    parser.add_argument(
        "--asset",
        required=True,
        help="Asset(s) to backtest, comma-separated (e.g. BTC or BTC,ETH,SOL,XRP)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Limit to last N days of data (default: all available)",
    )
    parser.add_argument(
        "--no-blackout",
        action="store_true",
        help="Disable blackout window filtering (default: blackout enabled)",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Use trained XGBoost ML model instead of fusion engine",
    )
    parser.add_argument(
        "--min-dm",
        type=int,
        default=0,
        help="Minimum decision minute to consider signals (default: 0). "
             "Set to 2 to match ML training min_dm.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use weighted ensemble of ML + fusion (requires ML model)",
    )
    parser.add_argument(
        "--ml-weight",
        type=float,
        default=0.65,
        help="ML weight in ensemble blend (default: 0.65). Only used with --ensemble.",
    )
    parser.add_argument(
        "--ens-threshold",
        type=float,
        default=0.60,
        help="Ensemble confidence threshold (default: 0.60). Only used with --ensemble.",
    )
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]
    use_blackout = not args.no_blackout

    for asset in assets:
        run_asset(
            asset, args.days, use_blackout,
            use_ml=args.ml, min_dm=args.min_dm,
            ensemble=args.ensemble,
            ml_weight=args.ml_weight,
            ens_threshold=args.ens_threshold,
        )


if __name__ == "__main__":
    main()
