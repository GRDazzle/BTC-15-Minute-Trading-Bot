"""
Signal accuracy backtester.

Usage:
  python scripts/backtest.py --asset BTC
  python scripts/backtest.py --asset BTC,ETH,SOL,XRP
  python scripts/backtest.py --asset BTC --days 30
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader import load_binance_klines, load_fear_greed, generate_windows
from backtester.simulator import BacktestSimulator
from backtester.reporter import print_report, export_csv

from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine

# Map CLI asset names to CSV filenames
ASSET_MAP = {
    "BTC": "BTCUSD_1m.csv",
    "ETH": "ETHUSD_1m.csv",
    "SOL": "SOLUSD_1m.csv",
    "XRP": "XRPUSD_1m.csv",
}

DATA_DIR = PROJECT_ROOT / "data" / "historical"
BINANCE_DIR = DATA_DIR / "binance"
FG_CSV = DATA_DIR / "fear_greed.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "backtest"


def build_processors() -> list:
    """
    Instantiate signal processors calibrated for spot USD prices.

    Thresholds tuned for 1m kline close-to-close moves:
      BTC typical 1m move: 0.03-0.15%
      Meaningful deviation from 20-period MA: ~0.2-0.5%
    """
    return [
        SpikeDetectionProcessor(
            spike_threshold=0.003,      # 0.3% deviation from 20-period MA
            velocity_threshold=0.0015,  # 0.15% move in 3 ticks
            lookback_periods=20,
            min_confidence=0.55,
        ),
        TickVelocityProcessor(
            velocity_threshold_60s=0.001,   # 0.1% in 60s
            velocity_threshold_30s=0.0007,  # 0.07% in 30s
            min_ticks=5,
            min_confidence=0.55,
        ),
        # SentimentProcessor disabled: daily F&G has no predictive power at 15m resolution.
        # PriceDivergenceProcessor disabled: requires Kalshi/Polymarket probability prices.
    ]


def run_asset(asset: str, days: int | None) -> None:
    csv_name = ASSET_MAP.get(asset.upper())
    if csv_name is None:
        logger.error(f"Unknown asset: {asset}. Available: {list(ASSET_MAP.keys())}")
        return

    csv_path = BINANCE_DIR / csv_name
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return

    # Load data
    klines = load_binance_klines(csv_path)

    # Filter by --days if specified
    if days is not None and klines:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        klines = [k for k in klines if k.timestamp >= cutoff]
        logger.info(f"Filtered to last {days} days: {len(klines)} klines")

    fg_scores = load_fear_greed(FG_CSV) if FG_CSV.exists() else {}

    # Generate windows
    windows = generate_windows(klines)
    if not windows:
        logger.error(f"No valid windows for {asset}")
        return

    # Build processors + fusion engine (fresh per asset)
    processors = build_processors()
    fusion_engine = SignalFusionEngine()

    # Run backtest
    simulator = BacktestSimulator(processors, fusion_engine)
    results = simulator.run(windows, fg_scores)

    # Report
    print_report(results, asset)

    # Export CSV
    out_path = OUTPUT_DIR / f"{asset.upper()}_backtest.csv"
    export_csv(results, out_path)


def main():
    # Configure loguru: file only, no stdout spam during backtest
    logger.remove()
    logger.add(
        PROJECT_ROOT / "logs" / "backtest.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )
    # Minimal stderr for errors only
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="Signal accuracy backtester")
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
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]

    for asset in assets:
        run_asset(asset, args.days)


if __name__ == "__main__":
    main()
