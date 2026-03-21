"""
Kalshi PnL backtester.

Joins Binance aggTrade signals with real Kalshi polling data (bid/ask prices +
settlement outcomes) to calculate actual dollar PnL -- not just signal accuracy.

Validates the full pipeline:
  signal -> Kalshi price check -> Kelly band filter -> position size -> settlement

Usage:
  python scripts/backtest_kalshi_pnl.py --asset BTC
  python scripts/backtest_kalshi_pnl.py --asset BTC,ETH,SOL,XRP
  python scripts/backtest_kalshi_pnl.py --asset BTC --min-dm 3
"""
import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from backtester.data_loader import load_fear_greed
from backtester.data_loader_ticks import (
    load_aggtrades_multi,
    generate_tick_windows,
    TickWindow,
    Tick,
    resample_ticks,
)
from backtester.data_loader_kalshi import (
    load_kalshi_windows,
    get_kalshi_prices,
    KalshiWindow,
)
from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine

# -- Paths --
DATA_DIR = PROJECT_ROOT / "data" / "aggtrades"
KALSHI_DATA_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
FG_CSV = PROJECT_ROOT / "data" / "historical" / "fear_greed.csv"
CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output" / "kalshi_pnl"

# Kalshi fee per contract in cents
KALSHI_FEE_CENTS = 2


@dataclass
class PnLResult:
    """Full audit trail for one simulated trade."""
    window_start: datetime
    event_ticker: str
    actual_direction: str          # Binance: BULLISH / BEARISH
    predicted_direction: str       # BULLISH / BEARISH / NONE
    decision_minute: int
    confidence: float

    # Kalshi trade details
    side: str                      # "yes" or "no"
    entry_price_cents: int         # yes_ask or no_ask from Kalshi poll
    contracts: int
    cost_dollars: float            # (price/100) * contracts
    fees_dollars: float            # (fee/100) * contracts

    # Settlement
    outcome: str                   # "yes" or "no" from Kalshi JSONL
    revenue_dollars: float         # contracts * $1.00 if won, else $0
    pnl_dollars: float             # revenue - cost - fees
    balance_after: float

    # Skip reason (if trade was not taken)
    skip_reason: str = ""          # "no_signal", "kelly_band", "no_kalshi_data", "no_ask"


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


def load_config(asset: str) -> dict:
    """Load per-asset config from trading.json."""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning("Config not found at {}, using defaults", CONFIG_PATH)
        return {
            "initial_balance": 25.0,
            "max_contracts_per_trade": 10,
            "max_price_cents": 85,
            "min_price_cents": 15,
            "ensemble": None,
        }

    defaults = config.get("defaults", {})
    asset_cfg = config.get("assets", {}).get(asset.upper(), {})

    return {
        "initial_balance": asset_cfg.get("initial_balance", defaults.get("initial_balance", 25.0)),
        "max_contracts_per_trade": asset_cfg.get("max_contracts_per_trade", defaults.get("max_contracts_per_trade", 10)),
        "max_price_cents": asset_cfg.get("max_price_cents", defaults.get("max_price_cents", 85)),
        "min_price_cents": asset_cfg.get("min_price_cents", defaults.get("min_price_cents", 15)),
        "ensemble": asset_cfg.get("ensemble"),
    }


def calculate_contracts(
    balance: float,
    price_cents: int,
    confidence: float,
    max_contracts: int,
) -> int:
    """Determine how many contracts to buy.

    Scales by confidence, capped by max_contracts and available balance.
    """
    cost_per = (price_cents + KALSHI_FEE_CENTS) / 100.0
    if cost_per <= 0 or balance <= 0:
        return 0

    max_by_balance = int(balance / cost_per)
    scale = min(1.0, confidence)
    desired = max(1, int(max_by_balance * scale))
    return min(desired, max_contracts)


def simulate_pnl(
    asset: str,
    tick_windows: list[TickWindow],
    kalshi_windows: dict[datetime, KalshiWindow],
    processors: list,
    fusion_engine: SignalFusionEngine,
    ml_processor: Optional[object],
    ensemble_weights: Optional[tuple],
    config: dict,
    min_dm: int,
    fg_scores: dict,
) -> list[PnLResult]:
    """Run PnL simulation: replay Binance signals, trade at Kalshi prices.

    For each tick window:
    1. Match to Kalshi window by window_end == close_time
    2. Replay tick data through ensemble (same as simulator._simulate_tick_window)
    3. At first actionable signal, look up Kalshi ask price
    4. Apply Kelly band filter, calculate contracts, deduct cost
    5. Look up Kalshi outcome for settlement
    """
    results: list[PnLResult] = []
    balance = config["initial_balance"]
    max_contracts = config["max_contracts_per_trade"]
    max_price = config["max_price_cents"]
    min_price = config["min_price_cents"]

    # Continuous buffers across windows (like live + simulator)
    price_history_dq: deque[Decimal] = deque(maxlen=200)
    tick_buffer: deque[dict] = deque(maxlen=300)

    total = len(tick_windows)
    matched = 0
    skipped_no_kalshi = 0
    skipped_no_signal = 0
    skipped_kelly = 0
    skipped_no_ask = 0

    log_interval = max(1, total // 20)

    for i, tw in enumerate(tick_windows):
        if i % log_interval == 0:
            logger.info("PnL sim {}/{} ({}) balance=${:.2f}", i + 1, total, tw.window_start, balance)

        # Match Binance window_end to Kalshi close_time
        kw = kalshi_windows.get(tw.window_end)
        if kw is None:
            skipped_no_kalshi += 1
            # Still feed ticks into buffers to maintain continuity for
            # processors (price_history, tick_buffer span across windows)
            _feed_buffers_only(tw, price_history_dq, tick_buffer)
            continue

        matched += 1

        # Replay tick window through signal pipeline
        signal_result = _replay_window_for_signal(
            tw, processors, fusion_engine, ml_processor, ensemble_weights,
            min_dm, fg_scores, price_history_dq, tick_buffer,
        )

        if signal_result is None:
            skipped_no_signal += 1
            results.append(PnLResult(
                window_start=tw.window_start,
                event_ticker=kw.event_ticker,
                actual_direction=tw.actual_direction,
                predicted_direction="NONE",
                decision_minute=-1,
                confidence=0.0,
                side="",
                entry_price_cents=0,
                contracts=0,
                cost_dollars=0.0,
                fees_dollars=0.0,
                outcome=kw.outcome or "",
                revenue_dollars=0.0,
                pnl_dollars=0.0,
                balance_after=balance,
                skip_reason="no_signal",
            ))
            continue

        predicted, dm, confidence, signal_ts = signal_result

        # Look up Kalshi prices at signal timestamp
        kalshi_prices = get_kalshi_prices(kw, signal_ts)
        if kalshi_prices is None:
            skipped_no_ask += 1
            results.append(PnLResult(
                window_start=tw.window_start,
                event_ticker=kw.event_ticker,
                actual_direction=tw.actual_direction,
                predicted_direction=predicted,
                decision_minute=dm,
                confidence=confidence,
                side="",
                entry_price_cents=0,
                contracts=0,
                cost_dollars=0.0,
                fees_dollars=0.0,
                outcome=kw.outcome or "",
                revenue_dollars=0.0,
                pnl_dollars=0.0,
                balance_after=balance,
                skip_reason="no_ask",
            ))
            continue

        # Determine side and ask price
        if predicted == "BULLISH":
            side = "yes"
            entry_price = kalshi_prices["yes_ask"]
        else:  # BEARISH
            side = "no"
            entry_price = kalshi_prices["no_ask"]

        # Kelly band filter (matches live _validate_price: entry must be in [min, max])
        if entry_price > max_price or entry_price < min_price:
            skipped_kelly += 1
            if entry_price > max_price:
                reason = f"kelly_band(ask={entry_price}c>max={max_price}c)"
            else:
                reason = f"kelly_band(ask={entry_price}c<min={min_price}c)"
            results.append(PnLResult(
                window_start=tw.window_start,
                event_ticker=kw.event_ticker,
                actual_direction=tw.actual_direction,
                predicted_direction=predicted,
                decision_minute=dm,
                confidence=confidence,
                side=side,
                entry_price_cents=entry_price,
                contracts=0,
                cost_dollars=0.0,
                fees_dollars=0.0,
                outcome=kw.outcome or "",
                revenue_dollars=0.0,
                pnl_dollars=0.0,
                balance_after=balance,
                skip_reason=reason,
            ))
            continue

        # Validate ask price is reasonable (not 0, not 100)
        if entry_price <= 0 or entry_price >= 100:
            skipped_no_ask += 1
            results.append(PnLResult(
                window_start=tw.window_start,
                event_ticker=kw.event_ticker,
                actual_direction=tw.actual_direction,
                predicted_direction=predicted,
                decision_minute=dm,
                confidence=confidence,
                side=side,
                entry_price_cents=entry_price,
                contracts=0,
                cost_dollars=0.0,
                fees_dollars=0.0,
                outcome=kw.outcome or "",
                revenue_dollars=0.0,
                pnl_dollars=0.0,
                balance_after=balance,
                skip_reason=f"invalid_ask({entry_price}c)",
            ))
            continue

        # Calculate contracts
        contracts = calculate_contracts(balance, entry_price, confidence, max_contracts)
        if contracts < 1:
            results.append(PnLResult(
                window_start=tw.window_start,
                event_ticker=kw.event_ticker,
                actual_direction=tw.actual_direction,
                predicted_direction=predicted,
                decision_minute=dm,
                confidence=confidence,
                side=side,
                entry_price_cents=entry_price,
                contracts=0,
                cost_dollars=0.0,
                fees_dollars=0.0,
                outcome=kw.outcome or "",
                revenue_dollars=0.0,
                pnl_dollars=0.0,
                balance_after=balance,
                skip_reason="insufficient_balance",
            ))
            continue

        # Calculate cost and fees
        cost = round((entry_price / 100.0) * contracts, 4)
        fees = round((KALSHI_FEE_CENTS / 100.0) * contracts, 4)

        # Deduct from balance
        balance -= (cost + fees)

        # Settlement
        outcome = kw.outcome or ""
        won = (side == outcome)
        revenue = contracts * 1.00 if won else 0.0
        pnl = revenue - cost - fees

        # Credit revenue
        balance += revenue

        results.append(PnLResult(
            window_start=tw.window_start,
            event_ticker=kw.event_ticker,
            actual_direction=tw.actual_direction,
            predicted_direction=predicted,
            decision_minute=dm,
            confidence=confidence,
            side=side,
            entry_price_cents=entry_price,
            contracts=contracts,
            cost_dollars=cost,
            fees_dollars=fees,
            outcome=outcome,
            revenue_dollars=revenue,
            pnl_dollars=pnl,
            balance_after=balance,
        ))

    logger.info(
        "PnL sim done: {} matched, {} no Kalshi data, {} no signal, {} Kelly skip, {} no ask",
        matched, skipped_no_kalshi, skipped_no_signal, skipped_kelly, skipped_no_ask,
    )

    return results


def _feed_buffers_only(
    window: TickWindow,
    price_history: deque,
    tick_buffer: deque,
) -> None:
    """Feed tick data into buffers without running signal processors.

    Used for non-matching windows to maintain price_history/tick_buffer
    continuity across the full date range.
    """
    resample_ms = 250

    if window.ticks_before:
        warmup_bars = resample_ticks(
            window.ticks_before,
            window.ticks_before[0].ts,
            window.ticks_before[-1].ts + timedelta(seconds=1),
            interval_ms=resample_ms,
        )
        for bar in warmup_bars:
            price_history.append(Decimal(str(bar["price"])))
            tick_buffer.append(bar)

    if window.ticks_during:
        decision_bars = resample_ticks(
            window.ticks_during,
            window.ticks_during[0].ts,
            window.ticks_during[-1].ts + timedelta(seconds=1),
            interval_ms=resample_ms,
        )
        for bar in decision_bars:
            price_history.append(Decimal(str(bar["price"])))
            tick_buffer.append(bar)


def _replay_window_for_signal(
    window: TickWindow,
    processors: list,
    fusion_engine: SignalFusionEngine,
    ml_processor: Optional[object],
    ensemble_weights: Optional[tuple],
    min_dm: int,
    fg_scores: dict,
    price_history: deque,
    tick_buffer: deque,
) -> Optional[tuple[str, int, float, datetime]]:
    """Replay a tick window through signal pipeline, return first actionable signal.

    Returns (predicted_direction, dm, confidence, signal_ts) or None.

    This mirrors BacktestSimulator._simulate_tick_window exactly.
    """
    resample_ms = 250

    raw_tick_buffer: deque[dict] = deque(maxlen=1200)

    # Feed warmup ticks
    if window.ticks_before:
        warmup_bars = resample_ticks(
            window.ticks_before,
            window.ticks_before[0].ts,
            window.ticks_before[-1].ts + timedelta(seconds=1),
            interval_ms=resample_ms,
        )
        for bar in warmup_bars:
            price_history.append(Decimal(str(bar["price"])))
            tick_buffer.append(bar)
        for tick in window.ticks_before:
            raw_tick_buffer.append({
                "ts": tick.ts, "price": tick.price,
                "qty": tick.qty, "is_buyer": tick.is_buyer,
            })

    # Pre-resample decision zone
    if window.ticks_during:
        decision_bars = resample_ticks(
            window.ticks_during,
            window.ticks_during[0].ts,
            window.ticks_during[-1].ts + timedelta(seconds=1),
            interval_ms=resample_ms,
        )
    else:
        decision_bars = []

    fg_score = fg_scores.get(window.window_start.strftime("%Y-%m-%d"), 50)

    decision_start = window.window_start + timedelta(minutes=5)
    check_interval = timedelta(seconds=10)
    current_check = decision_start + check_interval
    bar_idx = 0
    raw_tick_idx = 0

    while current_check < window.window_end:
        next_check = current_check + check_interval

        # Feed bars up to current_check
        while bar_idx < len(decision_bars) and decision_bars[bar_idx]["ts"] < current_check:
            bar = decision_bars[bar_idx]
            price_history.append(Decimal(str(bar["price"])))
            tick_buffer.append(bar)
            bar_idx += 1

        # Feed raw ticks
        while raw_tick_idx < len(window.ticks_during) and window.ticks_during[raw_tick_idx].ts < current_check:
            tick = window.ticks_during[raw_tick_idx]
            raw_tick_buffer.append({
                "ts": tick.ts, "price": tick.price,
                "qty": tick.qty, "is_buyer": tick.is_buyer,
            })
            raw_tick_idx += 1

        if len(price_history) < 20:
            current_check = next_check
            continue

        current_price = price_history[-1]

        # Decision minute
        elapsed_s = (current_check - window.window_start).total_seconds()
        dm = int((elapsed_s - 300) / 60)

        if dm < min_dm:
            current_check = next_check
            continue

        # Momentum
        if len(price_history) >= 6:
            prev = float(price_history[-6])
            curr = float(current_price)
            momentum = (curr - prev) / prev if prev != 0 else 0.0
        else:
            momentum = 0.0

        metadata = {
            "tick_buffer": list(tick_buffer),
            "raw_tick_buffer": list(raw_tick_buffer),
            "spot_price": float(current_price),
            "momentum": momentum,
            "sentiment_score": fg_score,
            "decision_minute": dm,
            "window_open_price": window.price_open,
        }

        # Ensemble path
        if ensemble_weights is not None and ml_processor is not None:
            ml_w, ens_threshold = ensemble_weights
            fusion_w = 1.0 - ml_w

            ml_p = 0.5
            try:
                raw_p = ml_processor.predict_proba(
                    current_price, list(price_history), metadata,
                )
                if raw_p is not None:
                    ml_p = raw_p
            except Exception as e:
                logger.debug("ML predict_proba error: {}", e)

            fusion_p = _get_fusion_probability(
                processors, fusion_engine, current_price, price_history, metadata,
            )

            ensemble_p = ml_w * ml_p + fusion_w * fusion_p

            if ensemble_p >= ens_threshold:
                return ("BULLISH", dm, ensemble_p, current_check)
            elif ensemble_p <= 1.0 - ens_threshold:
                return ("BEARISH", dm, 1.0 - ensemble_p, current_check)

        # ML-only path
        elif ml_processor is not None:
            try:
                ml_signal = ml_processor.process(
                    current_price, list(price_history), metadata,
                )
                if ml_signal is not None:
                    direction_str = str(ml_signal.direction).upper()
                    if "BULLISH" in direction_str:
                        return ("BULLISH", dm, ml_signal.confidence, current_check)
                    elif "BEARISH" in direction_str:
                        return ("BEARISH", dm, ml_signal.confidence, current_check)
            except Exception as e:
                logger.debug("ML processor error: {}", e)

        # Fusion path
        else:
            signals = []
            for p in processors:
                try:
                    sig = p.process(current_price, list(price_history), metadata)
                    if sig is not None:
                        signals.append(sig)
                except Exception as e:
                    logger.debug("Processor {} error: {}", p.name, e)

            if signals:
                fused = fusion_engine.fuse_signals(signals)
                if fused and fused.is_actionable:
                    direction_str = str(fused.direction).upper()
                    if "BULLISH" in direction_str:
                        return ("BULLISH", dm, fused.confidence, current_check)
                    elif "BEARISH" in direction_str:
                        return ("BEARISH", dm, fused.confidence, current_check)

        current_check = next_check

    return None


def _get_fusion_probability(
    processors: list,
    fusion_engine: SignalFusionEngine,
    current_price: Decimal,
    price_history,
    metadata: dict,
) -> float:
    """Run fusion processors and return P(BULLISH) in [0, 1]."""
    signals = []
    for p in processors:
        try:
            sig = p.process(current_price, list(price_history), metadata)
            if sig is not None:
                signals.append(sig)
        except Exception:
            pass

    if not signals:
        return 0.5

    fused = fusion_engine.fuse_signals(signals)
    if not fused or not fused.is_actionable:
        return 0.5

    direction_str = str(fused.direction).upper()
    if "BULLISH" in direction_str:
        return fused.confidence
    elif "BEARISH" in direction_str:
        return 1.0 - fused.confidence
    return 0.5


# -- Reporting --

def print_pnl_report(results: list[PnLResult], asset: str, initial_balance: float) -> None:
    """Print PnL summary report."""
    total = len(results)
    if total == 0:
        print(f"\n=== {asset} PnL Backtest ===")
        print("No windows to report.")
        return

    traded = [r for r in results if r.contracts > 0]
    skipped_no_signal = sum(1 for r in results if r.skip_reason == "no_signal")
    skipped_kelly = sum(1 for r in results if r.skip_reason.startswith("kelly_band"))
    skipped_other = sum(1 for r in results if r.skip_reason and r.skip_reason not in ("no_signal",) and not r.skip_reason.startswith("kelly_band"))

    wins = [r for r in traded if r.pnl_dollars > 0]
    losses = [r for r in traded if r.pnl_dollars <= 0]

    final_balance = results[-1].balance_after if results else initial_balance
    total_pnl = final_balance - initial_balance

    # Date range
    first_date = results[0].window_start.strftime("%b %d")
    last_date = results[-1].window_start.strftime("%b %d")
    dates = sorted(set(r.window_start.strftime("%Y-%m-%d") for r in results))

    print()
    print(f"=== {asset} PnL Backtest ({first_date}-{last_date}, {len(dates)} days) ===")
    print(f"Total windows:    {total}")
    print(f"Traded:           {len(traded)} ({len(traded)/total*100:.1f}%)")
    print(f"Skipped (no signal): {skipped_no_signal}")
    print(f"Skipped (Kelly band): {skipped_kelly}")
    if skipped_other > 0:
        print(f"Skipped (other):  {skipped_other}")
    print()

    if traded:
        win_rate = len(wins) / len(traded) * 100
        print(f"Wins:  {len(wins)} ({win_rate:.1f}%)     Losses: {len(losses)}")

        # Average entry prices by side
        yes_trades = [r for r in traded if r.side == "yes"]
        no_trades = [r for r in traded if r.side == "no"]
        if yes_trades:
            avg_yes = sum(r.entry_price_cents for r in yes_trades) / len(yes_trades)
            print(f"Avg entry price:  {avg_yes:.1f}c (YES, {len(yes_trades)} trades)", end="")
        if no_trades:
            avg_no = sum(r.entry_price_cents for r in no_trades) / len(no_trades)
            if yes_trades:
                print(f", {avg_no:.1f}c (NO, {len(no_trades)} trades)")
            else:
                print(f"Avg entry price:  {avg_no:.1f}c (NO, {len(no_trades)} trades)")
        else:
            print()

        print()
        print(f"Starting balance: ${initial_balance:.2f}")
        print(f"Final balance:    ${final_balance:.2f}")
        pnl_sign = "+" if total_pnl >= 0 else ""
        print(f"Total PnL:        {pnl_sign}${total_pnl:.2f}")

        if traded:
            avg_pnl = sum(r.pnl_dollars for r in traded) / len(traded)
            best = max(traded, key=lambda r: r.pnl_dollars)
            worst = min(traded, key=lambda r: r.pnl_dollars)
            avg_sign = "+" if avg_pnl >= 0 else ""
            print(f"Avg PnL/trade:    {avg_sign}${avg_pnl:.3f}")
            print(f"Best trade:       +${best.pnl_dollars:.2f}")
            print(f"Worst trade:      -${abs(worst.pnl_dollars):.2f}")

        # Total fees
        total_fees = sum(r.fees_dollars for r in traded)
        total_cost = sum(r.cost_dollars for r in traded)
        total_revenue = sum(r.revenue_dollars for r in traded)
        print(f"\nTotal cost:       ${total_cost:.2f}")
        print(f"Total fees:       ${total_fees:.2f}")
        print(f"Total revenue:    ${total_revenue:.2f}")
        print(f"Avg contracts:    {sum(r.contracts for r in traded)/len(traded):.1f}")

        # Daily PnL
        print("\nDaily PnL:")
        daily: dict[str, list[PnLResult]] = defaultdict(list)
        for r in traded:
            daily[r.window_start.strftime("%Y-%m-%d")].append(r)

        for date in sorted(daily.keys()):
            day_results = daily[date]
            day_pnl = sum(r.pnl_dollars for r in day_results)
            day_wins = sum(1 for r in day_results if r.pnl_dollars > 0)
            sign = "+" if day_pnl >= 0 else ""
            print(f"  {date}: {sign}${day_pnl:.2f}  ({len(day_results)} trades, {day_wins} wins)")
        print()

        # By decision minute
        print("By decision minute:")
        dm_groups: dict[int, list[PnLResult]] = defaultdict(list)
        for r in traded:
            dm_groups[r.decision_minute].append(r)
        for dm in sorted(dm_groups.keys()):
            dm_results = dm_groups[dm]
            dm_pnl = sum(r.pnl_dollars for r in dm_results)
            dm_wins = sum(1 for r in dm_results if r.pnl_dollars > 0)
            sign = "+" if dm_pnl >= 0 else ""
            print(f"  dm {dm:2d}: {len(dm_results):4d} trades, {dm_wins} wins, PnL: {sign}${dm_pnl:.2f}")
        print()


def export_pnl_csv(results: list[PnLResult], path: Path) -> None:
    """Export per-window PnL detail to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_start", "event_ticker", "actual_direction", "predicted_direction",
            "decision_minute", "confidence", "side", "entry_price_cents",
            "contracts", "cost_dollars", "fees_dollars", "outcome",
            "revenue_dollars", "pnl_dollars", "balance_after", "skip_reason",
        ])
        for r in results:
            writer.writerow([
                r.window_start.isoformat(),
                r.event_ticker,
                r.actual_direction,
                r.predicted_direction,
                r.decision_minute,
                f"{r.confidence:.4f}",
                r.side,
                r.entry_price_cents,
                r.contracts,
                f"{r.cost_dollars:.4f}",
                f"{r.fees_dollars:.4f}",
                r.outcome,
                f"{r.revenue_dollars:.2f}",
                f"{r.pnl_dollars:.4f}",
                f"{r.balance_after:.4f}",
                r.skip_reason,
            ])
    print(f"Exported {len(results)} rows to {path}")


def run_asset(asset: str, min_dm: int, threshold_override: float = None) -> None:
    """Run PnL backtest for one asset."""
    config = load_config(asset)
    initial_balance = config["initial_balance"]
    ensemble_cfg = config.get("ensemble")

    print(f"\n{'='*60}")
    print(f"Asset: {asset}")
    print(f"Initial balance: ${initial_balance:.2f}")
    print(f"Max contracts: {config['max_contracts_per_trade']}")
    print(f"Kelly bands: {config['min_price_cents']}c - {config['max_price_cents']}c")

    # Load Binance tick data
    ticks = load_aggtrades_multi(DATA_DIR, asset)
    if not ticks:
        logger.error("No aggTrades data for {}. Run download_binance_aggtrades.py first.", asset)
        return

    # Generate tick windows
    tick_windows = generate_tick_windows(ticks)
    if not tick_windows:
        logger.error("No valid tick windows for {}", asset)
        return

    # Load Kalshi polling data
    kalshi_windows = load_kalshi_windows(KALSHI_DATA_DIR, asset)
    if not kalshi_windows:
        logger.error("No Kalshi data for {}.", asset)
        return

    # Pre-filter tick windows to Kalshi date range (avoid processing weeks
    # of Binance data that have no Kalshi match -- 2877 -> ~600 windows)
    kalshi_close_times = set(kalshi_windows.keys())
    kalshi_min = min(kalshi_close_times)
    kalshi_max = max(kalshi_close_times)
    # Keep a 1-window buffer before first Kalshi window for warmup
    buffer = timedelta(minutes=15)
    original_count = len(tick_windows)
    tick_windows = [
        tw for tw in tick_windows
        if tw.window_end >= kalshi_min - buffer and tw.window_end <= kalshi_max + buffer
    ]

    overlap = sum(1 for tw in tick_windows if tw.window_end in kalshi_close_times)
    print(f"Binance windows: {original_count} (filtered to {len(tick_windows)} in Kalshi date range)")
    print(f"Kalshi windows:  {len(kalshi_windows)}")
    print(f"Matched windows: {overlap}")

    if overlap == 0:
        print("No overlapping windows between Binance and Kalshi data!")
        return

    # Fear & Greed scores
    fg_scores = load_fear_greed(FG_CSV) if FG_CSV.exists() else {}

    # Build signal processors
    processors = build_processors()
    fusion_engine = SignalFusionEngine()

    # ML processor + ensemble
    ml_processor = None
    ensemble_weights = None

    if ensemble_cfg:
        try:
            from core.strategy_brain.signal_processors.ml_processor import MLProcessor
            ml_processor = MLProcessor(
                asset=asset,
                model_dir=MODEL_DIR,
                confidence_threshold=0.60,
            )
            ml_w = ensemble_cfg.get("ml_weight", 0.65)
            threshold = threshold_override if threshold_override is not None else ensemble_cfg.get("threshold", 0.70)
            ensemble_weights = (ml_w, threshold)
            ens_min_dm = ensemble_cfg.get("min_dm", min_dm)
            if ens_min_dm > min_dm:
                min_dm = ens_min_dm
            print(f"Ensemble: ml_weight={ml_w:.2f}, threshold={threshold:.2f}, min_dm={min_dm}")
        except FileNotFoundError as e:
            print(f"ML model not found for {asset}: {e}")
            print("Falling back to fusion engine")
        except ImportError:
            print("xgboost not installed, falling back to fusion engine")

    print(f"Min decision minute: {min_dm}")
    print(f"{'='*60}")

    # Run PnL simulation
    results = simulate_pnl(
        asset=asset,
        tick_windows=tick_windows,
        kalshi_windows=kalshi_windows,
        processors=processors,
        fusion_engine=fusion_engine,
        ml_processor=ml_processor,
        ensemble_weights=ensemble_weights,
        config=config,
        min_dm=min_dm,
        fg_scores=fg_scores,
    )

    # Report
    print_pnl_report(results, asset, initial_balance)

    # Export CSV
    out_path = OUTPUT_DIR / f"{asset.upper()}_kalshi_pnl.csv"
    export_pnl_csv(results, out_path)


def main():
    # Configure loguru
    logger.remove()
    log_path = PROJECT_ROOT / "logs" / "backtest_kalshi_pnl.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        mode="w",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="Kalshi PnL backtester")
    parser.add_argument(
        "--asset",
        required=True,
        help="Asset(s) to backtest, comma-separated (e.g. BTC or BTC,ETH,SOL,XRP)",
    )
    parser.add_argument(
        "--min-dm",
        type=int,
        default=2,
        help="Minimum decision minute (default: 2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override ensemble threshold (e.g. 0.63). Uses config value if not set.",
    )
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",")]

    for asset in assets:
        run_asset(asset, args.min_dm, threshold_override=args.threshold)


if __name__ == "__main__":
    main()
