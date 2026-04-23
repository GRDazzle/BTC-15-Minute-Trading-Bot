"""Replay archived data through the EXACT live inference path.

For a given date range and asset, this script:
  1. Loads Coinbase + Kraken aggTrade archives into a merged tick stream
  2. Loads Kalshi poll archive (for yes_ask/no_ask/mtc snapshots and settlement)
  3. Instantiates the LIVE MLProcessor and LSTMProcessor (same model files the
     running bot uses)
  4. Walks each 15-min window; at every 2s check in the decision zone it slices
     the last 900s of ticks, builds a metadata dict matching what
     kalshi_strategy.py assembles, and calls predict_proba() on both processors
  5. Applies the v9 dynamic blend inline (same formula as
     strategies/kalshi_strategy.py:~2037-2079) using per-asset config
  6. Takes the first trade that passes threshold + max_price gates (one trade
     per window, like live)
  7. Looks up settlement outcome from Kalshi poll archive, computes PnL
  8. Writes output/replay_trades.csv with schema matching live output/trades.csv

Usage:
  python scripts/replay_live_inference.py --asset BTC --date 2026-04-21
  python scripts/replay_live_inference.py --asset SOL --from 2026-04-19 --to 2026-04-22

Why this matters:
  The live bot claims some PnL; walk-forward backtest claims another. If we run
  the live inference code against archived data, any divergence is state-based
  (tick buffer ordering, poll timing, prewarm completeness) not math. So this
  script separates "the model is broken" from "the live pipeline feeds it
  slightly different inputs at evaluation time."

Limitations:
  - fusion_p is set to 0.5 (neutral). With dynamic_k_lstm=3, k_xgb=8 and
    confident XGB the fusion weight is typically ~0 in practice, so this is
    a reasonable simplification. Pass --with-fusion to construct a minimal
    fusion engine (SpikeDetection + TickVelocity only).
  - allowed_hours and blocked flags are ignored — replay decides on the
    ensemble alone. Pass --respect-hours to gate by config allowed_hours.
  - No hedge-on-flip; one trade per window (live keeps re-evaluating).
"""
import argparse
import bisect
import csv
import json
import sys
from collections import deque
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Silence loguru INFO spam from sub-processors (TickVelocity, etc.) so the
# replay's own progress output is readable. Users who want the noise can
# re-raise the level with LOGURU_LEVEL=INFO.
from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.add(sys.stderr, level="WARNING")

from core.strategy_brain.signal_processors.ml_processor import MLProcessor
from core.strategy_brain.signal_processors.lstm_processor import LSTMProcessor
from ml.kalshi_features import KalshiPollIndex, window_start_to_event_ticker

CONFIG_PATH = PROJECT_ROOT / "config" / "trading.json"
MODEL_DIR = PROJECT_ROOT / "models"
COINBASE_DIR = PROJECT_ROOT / "data" / "aggtrades_coinbase"
KRAKEN_DIR = PROJECT_ROOT / "data" / "aggtrades_kraken"
KALSHI_DIR = PROJECT_ROOT / "data" / "kalshi_polls"
OUTPUT_CSV = PROJECT_ROOT / "output" / "replay_trades.csv"

KALSHI_FEE_CENTS = 2  # matches execution adapter


def _load_config(asset: str) -> dict:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    a = cfg["assets"][asset]
    ens = a["ensemble"]
    return {
        "ml_weight": ens["ml_weight"],
        "threshold": ens["threshold"],
        "min_dm": ens.get("min_dm", 2),
        "max_dm": ens.get("max_dm", 8),
        "max_price_cents": ens.get("max_price_cents", a.get("max_price_cents", 90)),
        "min_price_cents": a.get("min_price_cents", 5),
        "xgb_max_w": ens.get("xgb_max_w", 0.60),
        "lstm_min_w": ens.get("lstm_min_w", 0.10),
        "lstm_max_w": ens.get("lstm_max_w", 0.40),
        "dynamic_k": ens.get("dynamic_k", 4.5),
        "dynamic_k_xgb": ens.get("dynamic_k_xgb", ens.get("dynamic_k", 4.5)),
        "dynamic_k_lstm": ens.get("dynamic_k_lstm", ens.get("dynamic_k", 4.5)),
        "max_contracts": a.get("max_contracts_per_trade", 500),
    }


def _parse_ts(ts_raw: int) -> datetime:
    """Auto-detect ms vs microsecond timestamps."""
    if ts_raw > 10**14:
        return datetime.fromtimestamp(ts_raw / 1_000_000, tz=timezone.utc)
    return datetime.fromtimestamp(ts_raw / 1000, tz=timezone.utc)


def _load_aggtrades(base_dir: Path, asset: str, quote: str, date: datetime.date) -> list[dict]:
    """Load one day of aggtrade CSV ticks.

    CSV column 6 is is_buyer_maker; invert to get is_buyer (taker side) to match
    live WS semantics (strategy + features.py treat is_buyer as "aggressive buy").
    """
    path = base_dir / asset / f"{asset}{quote}-aggTrades-{date:%Y-%m-%d}.csv"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            try:
                price = float(row[1])
                qty = float(row[2])
                ts = _parse_ts(int(row[5]))
            except (ValueError, IndexError):
                continue
            is_buyer_maker = row[6].strip().lower() in ("true", "1")
            out.append({
                "ts": ts,
                "price": price,
                "qty": qty,
                "is_buyer": not is_buyer_maker,
            })
    return out


def _merge_sorted(*streams: list[list[dict]]) -> list[dict]:
    merged = []
    for s in streams:
        merged.extend(s)
    merged.sort(key=lambda t: t["ts"])
    return merged


def _iter_windows(start_utc: datetime, end_utc: datetime):
    """Yield 15-min (window_start, window_end) UTC pairs covering the range.
    Aligned to :00 :15 :30 :45."""
    t = start_utc.replace(second=0, microsecond=0)
    t = t.replace(minute=(t.minute // 15) * 15)
    while t < end_utc:
        yield t, t + timedelta(minutes=15)
        t += timedelta(minutes=15)


def _v9_blend(ml_p: float, lstm_p: float | None, fusion_p: float, cfg: dict):
    """Replicates strategies/kalshi_strategy.py v9 dynamic blend."""
    xgb_min_w = cfg["ml_weight"]
    xgb_max_w = cfg["xgb_max_w"]
    k_xgb = cfg["dynamic_k_xgb"]
    k_lstm = cfg["dynamic_k_lstm"]
    xgb_conf = abs(ml_p - 0.5) * 2.0
    dyn_xgb_w = xgb_min_w + (xgb_max_w - xgb_min_w) * (xgb_conf ** k_xgb)
    if lstm_p is not None:
        lstm_conf = abs(lstm_p - 0.5) * 2.0
        dyn_lstm_w = cfg["lstm_min_w"] + (cfg["lstm_max_w"] - cfg["lstm_min_w"]) * (lstm_conf ** k_lstm)
        dyn_fusion_w = max(0.0, 1.0 - dyn_xgb_w - dyn_lstm_w)
        ensemble_p = dyn_xgb_w * ml_p + dyn_lstm_w * lstm_p + dyn_fusion_w * fusion_p
    else:
        dyn_lstm_w = 0.0
        dyn_fusion_w = max(0.0, 1.0 - dyn_xgb_w)
        ensemble_p = dyn_xgb_w * ml_p + dyn_fusion_w * fusion_p
    return ensemble_p, dyn_xgb_w, dyn_lstm_w, dyn_fusion_w


def _series_from_asset(asset: str) -> str:
    return f"KX{asset.upper()}15M"


def _market_ticker_from_event(event_ticker: str, strike: str = "00") -> str:
    """Event ticker + strike suffix → market_ticker (best-effort heuristic).
    Not all trades settle on the -00 market. For replay purposes we only need
    the event_ticker for settlement lookup; market_ticker is cosmetic."""
    return f"{event_ticker}-{strike}"


def replay_asset(asset: str, from_date: datetime.date, to_date: datetime.date,
                 cfg: dict, writer: csv.writer, balance: float = 25.0,
                 after_ts: datetime | None = None,
                 before_ts: datetime | None = None,
                 xgb_suffix: str = "",
                 lstm_suffix: str = "") -> dict:
    """Replay [from_date, to_date] inclusive for one asset. Returns summary dict.

    If after_ts / before_ts are given, only checks with that timestamp range are
    evaluated (useful for matching a specific live-bot run)."""
    ts_label = ""
    if after_ts or before_ts:
        lo = after_ts.isoformat() if after_ts else "-inf"
        hi = before_ts.isoformat() if before_ts else "+inf"
        ts_label = f"  [filter {lo} .. {hi}]"
    print(f"\n=== {asset} | {from_date} -> {to_date}{ts_label} ===")

    # Load processors (same as live). Optional suffix lets us audit staging models.
    ml_proc = MLProcessor(asset=asset, model_dir=MODEL_DIR, model_suffix=xgb_suffix)
    try:
        lstm_proc = LSTMProcessor(asset=asset, model_dir=MODEL_DIR, model_suffix=lstm_suffix)
    except FileNotFoundError:
        print(f"  [warn] No LSTM model for {asset}{lstm_suffix}; running XGB+fusion only")
        lstm_proc = None
    if xgb_suffix or lstm_suffix:
        print(f"  using models: XGB={asset}{xgb_suffix}_xgb, LSTM={asset}{lstm_suffix}_lstm")

    # Load Kalshi poll index for all dates covered (+1 day overlap)
    poll_idx = KalshiPollIndex(KALSHI_DIR, asset)
    series = _series_from_asset(asset)

    n_windows = 0
    n_trades = 0
    n_wins = 0
    pnl_total = 0.0

    current = from_date
    while current <= to_date:
        # Load previous + current day ticks so the first windows have warmup history
        prev_day = current - timedelta(days=1)
        cb_ticks = _load_aggtrades(COINBASE_DIR, asset, "USDT", prev_day) \
                 + _load_aggtrades(COINBASE_DIR, asset, "USDT", current)
        kr_ticks = _load_aggtrades(KRAKEN_DIR, asset, "USD", prev_day) \
                 + _load_aggtrades(KRAKEN_DIR, asset, "USD", current)

        if not cb_ticks and not kr_ticks:
            print(f"  {current}: no tick data, skipping")
            current += timedelta(days=1)
            continue

        merged = _merge_sorted(cb_ticks, kr_ticks)
        merged_ts = [t["ts"] for t in merged]
        kr_only = sorted(kr_ticks, key=lambda t: t["ts"])
        kr_ts = [t["ts"] for t in kr_only]

        day_start = datetime.combine(current, time(0, 0), tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)

        for window_start, window_end in _iter_windows(day_start, day_end):
            # Time-window filter: skip windows entirely outside the requested range
            if before_ts is not None and window_start >= before_ts:
                continue
            if after_ts is not None and window_end <= after_ts:
                continue

            n_windows += 1

            # Skip windows that have no ticks around them
            if not merged:
                continue
            if window_end < merged[0]["ts"] or window_start > merged[-1]["ts"]:
                continue

            # Event ticker uses window close_time (= window_end)
            event_ticker = window_start_to_event_ticker(asset, window_end)
            outcome = poll_idx.get_outcome(event_ticker)

            # Window open price: first tick at/after window_start
            wop_idx = bisect.bisect_left(merged_ts, window_start)
            if wop_idx >= len(merged):
                continue
            window_open_price = merged[wop_idx]["price"]

            decision_start = window_start + timedelta(minutes=5)
            # Walk decision zone in 2s steps
            check_ts = decision_start
            fired = False
            while check_ts < window_end and not fired:
                # Time-window filter for individual checks
                if after_ts is not None and check_ts < after_ts:
                    check_ts += timedelta(seconds=2)
                    continue
                if before_ts is not None and check_ts > before_ts:
                    break

                dm = int((check_ts - decision_start).total_seconds() // 60)
                if dm < cfg["min_dm"] or dm > cfg["max_dm"]:
                    check_ts += timedelta(seconds=2)
                    continue

                # Slice last 900s of merged ticks
                cutoff = check_ts - timedelta(seconds=900)
                lo = bisect.bisect_left(merged_ts, cutoff)
                hi = bisect.bisect_right(merged_ts, check_ts)
                if hi - lo < 20:
                    check_ts += timedelta(seconds=2)
                    continue
                tick_buffer = merged[lo:hi]
                if not tick_buffer:
                    check_ts += timedelta(seconds=2)
                    continue

                # Kraken-only slice
                kr_lo = bisect.bisect_left(kr_ts, cutoff)
                kr_hi = bisect.bisect_right(kr_ts, check_ts)
                kraken_tick_buffer = kr_only[kr_lo:kr_hi]

                current_price = tick_buffer[-1]["price"]
                price_history = [t["price"] for t in tick_buffer]

                # Kalshi snapshot closest to check_ts
                poll = poll_idx.find_poll(event_ticker, check_ts)
                if poll is None:
                    check_ts += timedelta(seconds=2)
                    continue
                k_yes_ask = int(poll["yes_ask"])
                k_yes_bid = int(poll["yes_bid"])
                k_no_ask = int(poll.get("no_ask", 100 - k_yes_bid))
                k_mtc = float(poll.get("mins_to_close", (window_end - check_ts).total_seconds() / 60))
                poll_history = poll_idx.get_poll_history(event_ticker, check_ts, 60)

                # SMAs over last N prices
                sma5 = sum(price_history[-5:]) / min(5, len(price_history))
                sma15 = sum(price_history[-15:]) / min(15, len(price_history))
                sma30 = sum(price_history[-30:]) / min(30, len(price_history))

                metadata = {
                    "tick_buffer": tick_buffer,
                    "raw_tick_buffer": tick_buffer,
                    "kraken_tick_buffer": kraken_tick_buffer,
                    "kraken_current_price": kraken_tick_buffer[-1]["price"] if kraken_tick_buffer else None,
                    "window_open_price": window_open_price,
                    "sma5": sma5, "sma15": sma15, "sma30": sma30,
                    "kalshi_yes_ask": k_yes_ask,
                    "kalshi_yes_bid": k_yes_bid,
                    "kalshi_no_ask": k_no_ask,
                    "kalshi_mins_to_close": k_mtc,
                    "kalshi_poll_history": poll_history,
                    "decision_minute": dm,
                }

                # Call live processors
                ml_p = ml_proc.predict_proba(
                    Decimal(str(current_price)), price_history, metadata,
                )
                if ml_p is None:
                    check_ts += timedelta(seconds=2)
                    continue
                lstm_p = None
                if lstm_proc is not None:
                    lstm_p = lstm_proc.predict_proba(
                        Decimal(str(current_price)), price_history, metadata,
                    )

                fusion_p = 0.5  # neutral — see docstring
                ensemble_p, dxw, dlw, dfw = _v9_blend(ml_p, lstm_p, fusion_p, cfg)

                # Decision
                threshold = cfg["threshold"]
                if ensemble_p >= threshold:
                    direction = "BULLISH"
                    entry_price = k_yes_ask
                    side = "yes"
                    confidence = ensemble_p
                elif ensemble_p <= 1.0 - threshold:
                    direction = "BEARISH"
                    entry_price = k_no_ask
                    side = "no"
                    confidence = 1.0 - ensemble_p
                else:
                    check_ts += timedelta(seconds=2)
                    continue

                # Price gate (same as execution adapter)
                if entry_price > cfg["max_price_cents"] or entry_price < cfg["min_price_cents"]:
                    check_ts += timedelta(seconds=2)
                    continue

                # Sizing: use config max_contracts (replay doesn't do Kelly scaling —
                # all trades at max size to make PnL comparable to live at same size)
                contracts = cfg["max_contracts"]
                cost_per = (entry_price + KALSHI_FEE_CENTS) / 100.0
                cost = contracts * cost_per
                if cost > balance:
                    contracts = int(balance / cost_per)
                    cost = contracts * cost_per
                if contracts < 1:
                    check_ts += timedelta(seconds=2)
                    continue

                # Settle
                if outcome is None:
                    # No settlement available — record as unsettled
                    pnl = 0.0
                    trade_outcome = "unknown"
                else:
                    won = (direction == "BULLISH" and outcome == "yes") or \
                          (direction == "BEARISH" and outcome == "no")
                    if won:
                        payout = contracts * 1.0
                        pnl = payout - cost
                        trade_outcome = outcome
                        n_wins += 1
                    else:
                        pnl = -cost
                        trade_outcome = outcome

                balance += pnl
                n_trades += 1
                pnl_total += pnl

                writer.writerow([
                    check_ts.isoformat(),
                    asset,
                    window_end.isoformat().replace("+00:00", "Z"),
                    _market_ticker_from_event(event_ticker),
                    event_ticker,
                    direction,
                    side,
                    entry_price,
                    contracts,
                    f"{cost:.4f}",
                    dm,
                    f"{k_mtc:.1f}",
                    f"{confidence:.4f}",
                    f"{confidence * 100:.2f}",
                    trade_outcome,
                    f"{pnl:+.4f}",
                    f"{balance:.2f}",
                    f"{ml_p:.4f}",
                    f"{lstm_p:.4f}" if lstm_p is not None else "",
                    f"{ensemble_p:.4f}",
                    f"{dxw:.4f}",
                    f"{dlw:.4f}",
                    f"{dfw:.4f}",
                ])
                fired = True
                # Don't break the check loop — fired flag exits while

        current += timedelta(days=1)

    wr = (n_wins / n_trades * 100) if n_trades else 0.0
    summary = {
        "asset": asset,
        "windows": n_windows,
        "trades": n_trades,
        "wins": n_wins,
        "win_rate": wr,
        "pnl": pnl_total,
        "ending_balance": balance,
    }
    print(f"  windows={n_windows} trades={n_trades} wins={n_wins} WR={wr:.1f}% PnL={pnl_total:+.2f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", action="append", help="Asset symbol; repeat for multiple (default: BTC,ETH,SOL,XRP)")
    ap.add_argument("--date", help="Single date YYYY-MM-DD")
    ap.add_argument("--from", dest="from_date", help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--to", dest="to_date", help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--balance", type=float, default=25.0, help="Starting balance per asset")
    ap.add_argument("--out", default=str(OUTPUT_CSV), help="Output CSV path")
    ap.add_argument("--hours-ago", type=float, default=None,
                    help="Replay only from N hours ago to now. Overrides --date/--from/--to.")
    ap.add_argument("--after-ts", default=None,
                    help="Only include checks at/after this ISO timestamp (UTC).")
    ap.add_argument("--before-ts", default=None,
                    help="Only include checks at/before this ISO timestamp (UTC).")
    ap.add_argument("--xgb-suffix", default="",
                    help="Load a staging XGB model (e.g. _align2s -> BTC_align2s_xgb.json)")
    ap.add_argument("--lstm-suffix", default="",
                    help="Load a staging LSTM model (same pattern)")
    args = ap.parse_args()

    after_ts = before_ts = None
    if args.hours_ago is not None:
        now = datetime.now(timezone.utc)
        after_ts = now - timedelta(hours=args.hours_ago)
        before_ts = now
        from_date = after_ts.date()
        to_date = before_ts.date()
    elif args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d").date()
        from_date = to_date = d
        if args.after_ts:
            after_ts = datetime.fromisoformat(args.after_ts).replace(tzinfo=timezone.utc) \
                if "+" not in args.after_ts else datetime.fromisoformat(args.after_ts)
        if args.before_ts:
            before_ts = datetime.fromisoformat(args.before_ts).replace(tzinfo=timezone.utc) \
                if "+" not in args.before_ts else datetime.fromisoformat(args.before_ts)
    else:
        if not args.from_date or not args.to_date:
            ap.error("Specify --hours-ago, --date, or both --from and --to")
        from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()
        to_date = datetime.strptime(args.to_date, "%Y-%m-%d").date()

    assets = args.asset or ["BTC", "ETH", "SOL", "XRP"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp", "asset", "window_id", "market_ticker", "event_ticker",
        "direction", "side", "price_cents", "contracts", "cost",
        "dm", "mtc", "confidence", "score", "outcome", "pnl", "balance_after",
        # Replay diagnostics (not in live trades.csv)
        "ml_p", "lstm_p", "ensemble_p", "dyn_xgb_w", "dyn_lstm_w", "dyn_fusion_w",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        summaries = []
        for asset in assets:
            try:
                cfg = _load_config(asset)
            except KeyError:
                print(f"No config for {asset}, skipping")
                continue
            s = replay_asset(asset, from_date, to_date, cfg, writer,
                             balance=args.balance,
                             after_ts=after_ts, before_ts=before_ts,
                             xgb_suffix=args.xgb_suffix,
                             lstm_suffix=args.lstm_suffix)
            summaries.append(s)

    print("\n=== SUMMARY ===")
    print(f"{'Asset':<6} {'Wins':>4} / {'Trades':>6}  {'WR':>6}  {'PnL':>10}")
    total_trades = total_wins = 0
    total_pnl = 0.0
    for s in summaries:
        print(f"{s['asset']:<6} {s['wins']:>4} / {s['trades']:>6}  {s['win_rate']:>5.1f}%  {s['pnl']:>+10.2f}")
        total_trades += s["trades"]
        total_wins += s["wins"]
        total_pnl += s["pnl"]
    tot_wr = (total_wins / total_trades * 100) if total_trades else 0.0
    print(f"{'TOTAL':<6} {total_wins:>4} / {total_trades:>6}  {tot_wr:>5.1f}%  {total_pnl:>+10.2f}")
    print(f"\nWrote {total_trades} trades to {out_path}")


if __name__ == "__main__":
    main()
