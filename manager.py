"""
Terminal UI Manager for Kalshi 15-Min Trading Bot.

Launches the live trading bot, displays real-time stats, and manages
weekly model retraining on a schedule.

Usage:
    python manager.py                          # Live trading (dry run)
    python manager.py --real                   # Live trading (real)
    python manager.py --assets BTC,ETH         # Specific assets
    python manager.py --retrain-day sunday     # Retrain on Sundays (default)
    python manager.py --retrain-hour 6         # Retrain at 6am UTC (default)
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parent
TRADES_CSV = PROJECT_ROOT / "output" / "trades.csv"
TRADES_LIVE_CSV = PROJECT_ROOT / "output" / "trades_live.csv"
BALANCE_CSV = PROJECT_ROOT / "output" / "balance.csv"
ACCOUNT_STATE = PROJECT_ROOT / "data" / "account_state.json"
CONFIG_JSON = PROJECT_ROOT / "config" / "trading.json"
DEPOSITS_LOG = PROJECT_ROOT / "data" / "deposits_log.csv"
RUNTIME_STATE = PROJECT_ROOT / "data" / "runtime_state.json"
RETRAIN_LOG = PROJECT_ROOT / "logs" / "weekly_retrain.log"
RETRAIN_STATE = PROJECT_ROOT / "data" / "retrain_state.json"

SERIES_TO_ASSET = {
    "KXBTC15M": "BTC", "KXETH15M": "ETH", "KXSOL15M": "SOL",
    "KXXRP15M": "XRP", "KXHYPE15M": "HYPE", "KXBNB15M": "BNB",
    "KXDOGE15M": "DOGE",
}

DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def load_trades():
    """Load settled trades from CSV."""
    trades = []
    if not TRADES_CSV.exists():
        return trades
    try:
        with open(TRADES_CSV, "r") as f:
            for row in csv.DictReader(f):
                if row.get("outcome") and row["outcome"].strip():
                    trades.append(row)
    except Exception:
        pass
    return trades


def load_account_state():
    """Load account state JSON."""
    if not ACCOUNT_STATE.exists():
        return {}
    try:
        with open(ACCOUNT_STATE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_retrain_state():
    """Load retrain schedule state."""
    if not RETRAIN_STATE.exists():
        return {"last_retrain": None, "last_retrain_status": None}
    try:
        with open(RETRAIN_STATE, "r") as f:
            return json.load(f)
    except Exception:
        return {"last_retrain": None, "last_retrain_status": None}


def load_live_assets():
    """Read which assets have dry_run=False in config/trading.json."""
    if not CONFIG_JSON.exists():
        return set()
    try:
        with open(CONFIG_JSON, "r") as f:
            cfg = json.load(f)
        live = set()
        for asset, asset_cfg in cfg.get("assets", {}).items():
            if asset_cfg.get("dry_run") is False:
                live.add(asset.upper())
        return live
    except Exception:
        return set()


def load_initial_balances():
    """Read initial_balance per asset from config/trading.json."""
    if not CONFIG_JSON.exists():
        return {}
    try:
        with open(CONFIG_JSON, "r") as f:
            cfg = json.load(f)
        defaults = cfg.get("defaults", {})
        default_balance = defaults.get("initial_balance", 25.0)
        out = {}
        for asset, asset_cfg in cfg.get("assets", {}).items():
            out[asset.upper()] = float(asset_cfg.get("initial_balance", default_balance))
        return out
    except Exception:
        return {}


def load_runtime_state():
    """Load the bot-published runtime stats (Kalshi balance, verification counts)."""
    if not RUNTIME_STATE.exists():
        return None
    try:
        with open(RUNTIME_STATE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_live_trades():
    """Load settled trades from output/trades_live.csv (created by go_live.py)."""
    trades = []
    if not TRADES_LIVE_CSV.exists():
        return trades
    try:
        with open(TRADES_LIVE_CSV, "r") as f:
            for row in csv.DictReader(f):
                if row.get("outcome") and row["outcome"].strip():
                    trades.append(row)
    except Exception:
        pass
    return trades


def load_deposits_per_asset():
    """Sum applied deposits per asset from data/deposits_log.csv."""
    out = {}
    if not DEPOSITS_LOG.exists():
        return out
    try:
        with open(DEPOSITS_LOG, "r") as f:
            for row in csv.DictReader(f):
                asset = (row.get("asset") or "").strip()
                amt_str = (row.get("amount_dollars") or "").strip()
                if not asset or not amt_str:
                    continue
                try:
                    out[asset] = out.get(asset, 0.0) + float(amt_str)
                except ValueError:
                    continue
    except Exception:
        pass
    return out


def compute_ledger_per_asset(trades, initial_balances, deposits):
    """Rebuild each asset's balance from history.

    ledger(asset) = initial + sum(pnl) + sum(deposits)
    """
    pnl_sum = {}
    for t in trades:
        asset = (t.get("asset") or "").strip()
        if not asset:
            continue
        pnl_str = (t.get("pnl") or "").strip().replace("+", "")
        if not pnl_str or pnl_str.lower() == "pending":
            continue
        try:
            pnl_sum[asset] = pnl_sum.get(asset, 0.0) + float(pnl_str)
        except ValueError:
            continue
    out = {}
    all_assets = set(initial_balances.keys()) | set(pnl_sum.keys()) | set(deposits.keys())
    for asset in all_assets:
        out[asset] = (
            initial_balances.get(asset, 0.0)
            + pnl_sum.get(asset, 0.0)
            + deposits.get(asset, 0.0)
        )
    return out


def save_retrain_state(state):
    """Save retrain schedule state."""
    RETRAIN_STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_STATE, "w") as f:
        json.dump(state, f, indent=2)


def next_retrain_time(retrain_day: int, retrain_hour: int) -> datetime:
    """Calculate next retrain datetime."""
    now = datetime.now(timezone.utc)
    days_ahead = retrain_day - now.weekday()
    if days_ahead < 0 or (days_ahead == 0 and now.hour >= retrain_hour):
        days_ahead += 7
    next_dt = now.replace(hour=retrain_hour, minute=0, second=0, microsecond=0)
    next_dt += timedelta(days=days_ahead)
    return next_dt


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 0:
        return "overdue"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _build_balance_table(title, account_state, trades, asset_filter, ledger):
    """Build a balance table for either live or dry-run assets.

    `asset_filter` is a set of asset names to include. If empty, returns None
    (caller should skip the table).
    """
    rows = []
    for series_key, acct in account_state.items():
        asset = SERIES_TO_ASSET.get(series_key, series_key.replace("KX", "").replace("15M", ""))
        if asset not in asset_filter:
            continue
        rows.append((asset, acct))
    if not rows:
        return None

    table = Table(title=title, expand=True)
    table.add_column("Asset", style="cyan", width=8)
    table.add_column("Balance", justify="right", width=11)
    table.add_column("PnL", justify="right", width=11)
    table.add_column("Trades", justify="right", width=7)
    table.add_column("WR All", justify="right", width=7)
    table.add_column("WR L15", justify="right", width=7)

    total_bal = 0.0
    total_pnl = 0.0
    total_count = 0
    total_wins = 0
    total_l15: list = []

    for asset, acct in rows:
        bal = acct.get("balance_dollars", 0)
        pnl = acct.get("pnl_dollars", 0)
        total_bal += bal
        total_pnl += pnl

        asset_trades = [t for t in trades if t.get("asset") == asset]
        wins = sum(1 for t in asset_trades if float(t.get("pnl", "0").replace("+", "")) > 0)
        count = len(asset_trades)
        wr = f"{wins/count*100:.0f}%" if count > 0 else "--"
        total_count += count
        total_wins += wins

        l15 = asset_trades[-15:]
        l15_wins = sum(1 for t in l15 if float(t.get("pnl", "0").replace("+", "")) > 0)
        if l15:
            wr_l15_pct = l15_wins / len(l15) * 100
            l15_color = "green" if wr_l15_pct >= 70 else ("yellow" if wr_l15_pct >= 55 else "red")
            wr_l15 = f"[{l15_color}]{wr_l15_pct:.0f}%[/]"
        else:
            wr_l15 = "--"
        total_l15.extend(l15)

        # Per-asset drift glyph: bal vs ledger
        ledger_bal = ledger.get(asset)
        drift_marker = ""
        if ledger_bal is not None and abs(bal - ledger_bal) > 0.10:
            drift_marker = " [yellow]![/]"

        pnl_style = "green" if pnl >= 0 else "red"
        table.add_row(
            asset,
            f"${bal:.2f}",
            f"[{pnl_style}]${pnl:+.2f}[/]{drift_marker}",
            str(count),
            wr,
            wr_l15,
        )

    total_wr = f"{total_wins/total_count*100:.0f}%" if total_count > 0 else "--"
    if total_l15:
        tl15_wins = sum(1 for t in total_l15 if float(t.get("pnl", "0").replace("+", "")) > 0)
        total_l15_wr = f"{tl15_wins/len(total_l15)*100:.0f}%"
    else:
        total_l15_wr = "--"
    total_pnl_style = "green" if total_pnl >= 0 else "red"
    table.add_row(
        "[bold]TOTAL[/]",
        f"[bold]${total_bal:.2f}[/]",
        f"[bold {total_pnl_style}]${total_pnl:+.2f}[/]",
        f"[bold]{total_count}[/]",
        f"[bold]{total_wr}[/]",
        f"[bold]{total_l15_wr}[/]",
        style="bold",
    )
    return table


def _build_reconciliation_panel(
    live_assets,
    in_memory_per_asset,
    ledger_per_asset,
    runtime_state,
):
    """Build the reconciliation panel showing live & dry 3-way checks."""
    live_total_memory = sum(in_memory_per_asset.get(a, 0.0) for a in live_assets)
    live_total_ledger = sum(ledger_per_asset.get(a, 0.0) for a in live_assets)
    dry_assets = set(in_memory_per_asset.keys()) - set(live_assets)
    dry_total_memory = sum(in_memory_per_asset.get(a, 0.0) for a in dry_assets)
    dry_total_ledger = sum(ledger_per_asset.get(a, 0.0) for a in dry_assets)

    # Pull Kalshi balance + verification stats from runtime_state
    kalshi_bal = None
    pending = 0
    stale = 0
    corrections = 0
    drift_total = 0.0
    last_run_at = None
    if runtime_state:
        recon = runtime_state.get("reconciliation", {})
        kalshi_bal = recon.get("kalshi_balance")
        last_run_at = recon.get("last_run_at")
        verif = runtime_state.get("verification", {})
        pending = verif.get("pending_count", 0)
        stale = verif.get("stale_count", 0)
        corrections = verif.get("total_corrections", 0)
        drift_total = verif.get("total_drift_dollars", 0.0)

    # LIVE row
    if live_assets:
        live_diff_mem_ledger = live_total_memory - live_total_ledger
        if kalshi_bal is not None:
            kalshi_diff = kalshi_bal - live_total_memory
            kalshi_color = (
                "green" if abs(kalshi_diff) < 0.10
                else ("yellow" if abs(kalshi_diff) < 1.0 else "red")
            )
            live_line = (
                f"[bold]LIVE[/]  L=${live_total_ledger:.2f}  "
                f"M=${live_total_memory:.2f}  "
                f"K=${kalshi_bal:.2f}  "
                f"[{kalshi_color}]drift ${kalshi_diff:+.2f}[/]"
            )
        else:
            mem_color = (
                "green" if abs(live_diff_mem_ledger) < 0.10
                else ("yellow" if abs(live_diff_mem_ledger) < 1.0 else "red")
            )
            live_line = (
                f"[bold]LIVE[/]  L=${live_total_ledger:.2f}  "
                f"M=${live_total_memory:.2f}  "
                f"K=[dim](no fetch yet)[/]  "
                f"[{mem_color}]M-L ${live_diff_mem_ledger:+.2f}[/]"
            )
    else:
        live_line = "[bold]LIVE[/]  [dim](no live assets)[/]"

    # DRY row -- only ledger vs memory (no Kalshi)
    dry_diff = dry_total_memory - dry_total_ledger
    dry_color = (
        "green" if abs(dry_diff) < 0.50
        else ("yellow" if abs(dry_diff) < 5.0 else "red")
    )
    dry_line = (
        f"[bold]DRY [/]  L=${dry_total_ledger:.2f}  "
        f"M=${dry_total_memory:.2f}  "
        f"[{dry_color}]M-L ${dry_diff:+.2f}[/]"
    )

    # Verification line (only meaningful when live assets exist)
    if live_assets:
        verify_line = (
            f"Pending verify: {pending}   Stale: {stale}   "
            f"Corrections: {corrections} ($ {drift_total:+.2f})"
        )
    else:
        verify_line = "[dim](verification idle -- no live assets)[/]"

    # Last reconcile timestamp
    if last_run_at:
        try:
            last_dt = datetime.fromisoformat(last_run_at)
            ago = format_duration((datetime.now(timezone.utc) - last_dt).total_seconds())
            ts_line = f"Last reconcile: {ago} ago"
        except Exception:
            ts_line = "Last reconcile: unknown"
    else:
        ts_line = "Last reconcile: never"

    body = "\n".join([live_line, "", dry_line, "", verify_line, ts_line])
    return Text.from_markup(body)


def build_dashboard(
    bot_process,
    trades,
    live_trades,
    account_state,
    retrain_state,
    retrain_day,
    retrain_hour,
    retrain_running,
    mode,
    assets,
    start_time,
    log_lines=None,
    runtime_state=None,
):
    """Build the rich dashboard layout."""
    now = datetime.now(timezone.utc)
    uptime = (now - start_time).total_seconds()

    # --- Header ---
    status = "[green]RUNNING[/]" if bot_process and bot_process.poll() is None else "[red]STOPPED[/]"
    mode_str = "[red]REAL[/]" if mode == "real" else "[yellow]DRY RUN[/]"
    header = Text.from_markup(
        f"  Kalshi 15-Min Trading Bot  |  {mode_str}  |  Bot: {status}  |  "
        f"Uptime: {format_duration(uptime)}  |  Assets: {assets}"
    )

    # --- Balance Tables (Live + Dry, split by config) ---
    live_assets = load_live_assets()
    initial_balances = load_initial_balances()
    deposits = load_deposits_per_asset()

    # Ledger uses dry-run trades.csv for dry assets and live trades.csv for live assets
    dry_ledger = compute_ledger_per_asset(trades, initial_balances, deposits)
    live_ledger = compute_ledger_per_asset(live_trades, initial_balances, deposits)
    # Combine: if asset is live, use live_ledger; else dry_ledger
    combined_ledger = {**dry_ledger}
    for a in live_assets:
        combined_ledger[a] = live_ledger.get(a, initial_balances.get(a, 0.0))

    # In-memory balances per asset
    in_memory_per_asset = {}
    for series_key, acct in account_state.items():
        asset = SERIES_TO_ASSET.get(series_key, series_key.replace("KX", "").replace("15M", ""))
        in_memory_per_asset[asset] = acct.get("balance_dollars", 0)

    # All assets that have a sub-account
    all_assets_with_state = set(in_memory_per_asset.keys())
    dry_assets = all_assets_with_state - live_assets

    live_table = _build_balance_table(
        "LIVE Trading", account_state, live_trades, live_assets, combined_ledger,
    )
    dry_table = _build_balance_table(
        "Dry Run", account_state, trades, dry_assets, combined_ledger,
    )

    # --- Recent Trades (merged: live + dry, sorted by timestamp) ---
    recent_table = Table(title="Recent Trades (last 10)", expand=True)
    recent_table.add_column("Time", width=8)
    recent_table.add_column("Asset", width=5)
    recent_table.add_column("Dir", width=7)
    recent_table.add_column("Price", justify="right", width=6)
    recent_table.add_column("Qty", justify="right", width=5)
    recent_table.add_column("Result", justify="right", width=8)
    recent_table.add_column("V", width=2)  # verification status

    # Tag each trade with its source so the V column can render correctly
    tagged = [(t, "dry") for t in trades] + [(t, "live") for t in live_trades]
    tagged.sort(key=lambda x: x[0].get("timestamp", ""))

    for t, source in tagged[-10:]:
        ts = t.get("timestamp", "")[:19].split("T")[-1][:8]
        asset = t.get("asset", "?")
        direction = t.get("direction", "?")
        price = t.get("price_cents", "?")
        contracts = t.get("contracts", "?")
        pnl_val = float(t.get("pnl", "0").replace("+", ""))
        pnl_style = "green" if pnl_val > 0 else "red"
        dir_style = "green" if direction == "BULLISH" else "red"

        # V column: blank for dry, +/. for live based on settlement_verified
        if source == "live":
            verified = (t.get("settlement_verified") or "").strip().lower() in ("true", "1", "yes")
            v_glyph = "[green]+[/]" if verified else "[yellow].[/]"
        else:
            v_glyph = " "

        recent_table.add_row(
            ts,
            asset,
            f"[{dir_style}]{direction[:4]}[/]",
            f"{price}c",
            str(contracts),
            f"[{pnl_style}]${pnl_val:+.2f}[/]",
            v_glyph,
        )

    # --- Retrain Status ---
    last_retrain = retrain_state.get("last_retrain")
    last_status = retrain_state.get("last_retrain_status", "unknown")

    if last_retrain:
        last_dt = datetime.fromisoformat(last_retrain)
        last_str = last_dt.strftime("%Y-%m-%d %H:%M UTC")
        ago = format_duration((now - last_dt).total_seconds())
        last_line = f"{last_str} ({ago} ago) - {last_status}"
    else:
        last_line = "Never"

    # Next retrain: Sunday 15 UTC (weekday) or Monday 15 UTC (weekend)
    next_sun = next_retrain_time(6, 15)  # Sunday 15 UTC = 8 AM PT
    next_mon = next_retrain_time(0, 15)  # Monday 15 UTC = 8 AM PT
    next_dt = min(next_sun, next_mon)
    next_type = "weekday" if next_dt == next_sun else "weekend"
    remaining = (next_dt - now).total_seconds()
    next_str = next_dt.strftime("%Y-%m-%d %H:%M UTC")
    remaining_str = format_duration(remaining)

    retrain_status = "RUNNING" if retrain_running else "idle"
    retrain_color = "yellow" if retrain_running else "green"

    retrain_text = (
        f"  Sun 15:00 UTC (8AM PT) -> weekday models\n"
        f"  Mon 15:00 UTC (8AM PT) -> weekend models\n"
        f"  Last: {last_line}\n"
        f"  Next: {next_type} {next_str} ({remaining_str})\n"
        f"  Status: [{retrain_color}]{retrain_status}[/]"
    )

    # --- Entry Price Band ---
    band_table = Table(title="PnL by Entry Price", expand=True)
    band_table.add_column("Band", width=8)
    band_table.add_column("W/L", justify="right", width=8)
    band_table.add_column("WR", justify="right", width=6)
    band_table.add_column("PnL", justify="right", width=10)

    for lo, hi in [(20, 55), (55, 65), (65, 75), (75, 85), (85, 95)]:
        band = [t for t in trades if lo <= int(t.get("price_cents", 0)) < hi]
        if not band:
            continue
        w = sum(1 for t in band if float(t.get("pnl", "0").replace("+", "")) > 0)
        p = sum(float(t.get("pnl", "0").replace("+", "")) for t in band)
        wr = w / len(band) * 100
        p_style = "green" if p >= 0 else "red"
        band_table.add_row(
            f"{lo}-{hi}c",
            f"{w}W/{len(band)-w}L",
            f"{wr:.0f}%",
            f"[{p_style}]${p:+.2f}[/]",
        )

    # --- PnL by Decision Minute ---
    dm_table = Table(title="PnL by Entry DM", expand=True)
    dm_table.add_column("DM", width=4)
    dm_table.add_column("W/L", justify="right", width=8)
    dm_table.add_column("WR", justify="right", width=6)
    dm_table.add_column("PnL", justify="right", width=10)

    for dm in range(2, 10):
        dm_trades = [t for t in trades if int(t.get("dm", 0)) == dm]
        if not dm_trades:
            continue
        w = sum(1 for t in dm_trades if float(t.get("pnl", "0").replace("+", "")) > 0)
        p = sum(float(t.get("pnl", "0").replace("+", "")) for t in dm_trades)
        wr = w / len(dm_trades) * 100
        p_style = "green" if p >= 0 else "red"
        dm_table.add_row(
            f"dm {dm}",
            f"{w}W/{len(dm_trades)-w}L",
            f"{wr:.0f}%",
            f"[{p_style}]${p:+.2f}[/]",
        )

    # --- Reconciliation panel ---
    recon_text = _build_reconciliation_panel(
        live_assets, in_memory_per_asset, combined_ledger, runtime_state,
    )
    # Border color depends on overall live drift severity
    recon_border = "green"
    if runtime_state and runtime_state.get("reconciliation", {}).get("kalshi_balance") is not None:
        drift = runtime_state["reconciliation"].get("drift") or 0.0
        if abs(drift) > 1.0:
            recon_border = "red"
        elif abs(drift) > 0.10:
            recon_border = "yellow"

    # --- Bot Log (use cached lines passed from manager) ---
    display_lines = log_lines if log_lines else ["  Waiting for bot logs..."]
    log_text = "\n".join(display_lines[-30:])

    # --- Assemble Layout ---
    layout = Layout()
    layout.split_column(
        Layout(Panel(header, style="bold blue"), size=3),
        Layout(name="main"),
    )
    layout["main"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )
    # Left: retrain status + entry price band + DM breakdown + reconciliation
    layout["left"].split_column(
        Layout(Panel(Text.from_markup(retrain_text), title="Model Retrain", border_style="magenta"), ratio=3),
        Layout(Panel(band_table, border_style="cyan"), ratio=3),
        Layout(Panel(dm_table, border_style="cyan"), ratio=3),
        Layout(Panel(recon_text, title="Reconciliation", border_style=recon_border), ratio=3),
    )
    # Right: balance section (Live + Dry stacked) on top, recent trades + log on bottom
    layout["right"].split_column(
        Layout(name="right_top", ratio=2),
        Layout(name="right_bottom", ratio=3),
    )

    # Build the balance section: stack Live + Dry tables, or single table if only one
    if live_table is not None and dry_table is not None:
        layout["right_top"].split_column(
            Layout(Panel(live_table, border_style="red"), ratio=1),
            Layout(Panel(dry_table, border_style="green"), ratio=2),
        )
    elif live_table is not None:
        layout["right_top"].update(Panel(live_table, border_style="red"))
    elif dry_table is not None:
        layout["right_top"].update(Panel(dry_table, border_style="green"))
    else:
        layout["right_top"].update(Panel(Text("(no accounts)"), border_style="dim"))

    layout["right_bottom"].split_row(
        Layout(Panel(recent_table, border_style="yellow"), ratio=1),
        Layout(Panel(Text(log_text), title="Bot Log", border_style="blue"), ratio=1),
    )

    return layout


class BotManager:
    """Manages the trading bot subprocess and retrain schedule."""

    def __init__(self, args):
        self.args = args
        self.bot_process = None
        self.retrain_process = None
        self.retrain_running = False
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        self.retrain_day = DAY_MAP.get(args.retrain_day.lower(), 6)
        self.retrain_hour = args.retrain_hour
        self._cached_log_lines: list[str] = []
        self._log_file_pos: int = 0

    def start_bot(self):
        """Launch the trading bot as a subprocess."""
        cmd = [
            sys.executable, "main.py",
            "--assets", self.args.assets,
        ]
        if self.args.real:
            cmd.append("--real")

        self.bot_process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    def stop_bot(self):
        """Stop the trading bot and all child processes."""
        if self.bot_process and self.bot_process.poll() is None:
            pid = self.bot_process.pid
            try:
                # On Windows, terminate() doesn't kill child processes.
                # Use taskkill /T to kill the entire process tree.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=15,
                )
            except Exception:
                # Fallback: force kill the process directly
                self.bot_process.kill()
            try:
                self.bot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.bot_process.kill()

    def run_retrain(self, day_type: str = "all"):
        """Run retrain in a background thread."""
        if self.retrain_running:
            return

        def _retrain():
            self.retrain_running = True
            state = load_retrain_state()

            try:
                result = subprocess.run(
                    [sys.executable, "scripts/weekly_retrain.py",
                     "--assets", self.args.assets,
                     "--days", "30",
                     "--day-type", day_type],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=7200,  # 2 hour timeout
                )
                status = "success" if result.returncode == 0 else f"failed (exit {result.returncode})"
            except subprocess.TimeoutExpired:
                status = "timeout (2h limit)"
            except Exception as e:
                status = f"error: {e}"

            state["last_retrain"] = datetime.now(timezone.utc).isoformat()
            state["last_retrain_status"] = f"{day_type}: {status}"
            save_retrain_state(state)
            self.retrain_running = False

        thread = threading.Thread(target=_retrain, daemon=True)
        thread.start()

    def check_retrain_schedule(self):
        """Check if it's time to retrain.

        Schedule:
          Sunday 15 UTC (8 AM PT) -> retrain weekday models
          Monday 15 UTC (8 AM PT) -> retrain weekend models
        """
        now = datetime.now(timezone.utc)

        # Sunday=6 -> weekday retrain, Monday=0 -> weekend retrain
        schedule = {6: "weekday", 0: "weekend"}
        day_type = schedule.get(now.weekday())
        if day_type is None:
            return
        if now.hour != 15:  # 15 UTC = 8 AM PT
            return

        state = load_retrain_state()
        last = state.get("last_retrain")
        if last:
            last_dt = datetime.fromisoformat(last)
            if (now - last_dt).total_seconds() < 82800:  # 23 hours
                return  # Already ran today

        self.run_retrain(day_type=day_type)

    def _read_new_log_lines(self):
        """Read new log lines incrementally from the most recent trading log."""
        log_dir = PROJECT_ROOT / "logs"
        # Only check the main trading.log (not the rotated copies)
        main_log = log_dir / "trading.log"
        if not main_log.exists():
            return

        current_file = str(main_log)

        # Reset position if file was rotated (shrunk)
        try:
            file_size = os.path.getsize(current_file)
        except OSError:
            return
        if file_size < self._log_file_pos:
            self._log_file_pos = 0

        try:
            with open(current_file, "r", errors="ignore") as f:
                f.seek(self._log_file_pos)
                new_data = f.read()
                self._log_file_pos = f.tell()
        except (FileNotFoundError, OSError):
            return

        if not new_data:
            return

        for line in new_data.splitlines():
            if "| INFO" in line:
                short = line.split(" | ", 2)[-1][:120]
                self._cached_log_lines.append(short)

        # Keep last 200 lines to avoid unbounded growth
        if len(self._cached_log_lines) > 200:
            self._cached_log_lines = self._cached_log_lines[-200:]

    def run(self):
        """Main loop — display dashboard and manage processes."""
        console = Console()

        # Start the bot
        self.start_bot()

        try:
            with Live(console=console, refresh_per_second=1, screen=True) as live:
                while self.running:
                    try:
                        # Check if bot died
                        if self.bot_process and self.bot_process.poll() is not None:
                            # Bot crashed, restart
                            time.sleep(5)
                            self.start_bot()

                        # Check retrain schedule
                        self.check_retrain_schedule()

                        # Load current data
                        trades = load_trades()
                        live_trades = load_live_trades()
                        account_state = load_account_state()
                        retrain_state = load_retrain_state()
                        runtime_state = load_runtime_state()

                        # Read new log lines (incremental, cached)
                        self._read_new_log_lines()

                        # Build and display dashboard
                        dashboard = build_dashboard(
                            self.bot_process,
                            trades,
                            live_trades,
                            account_state,
                            retrain_state,
                            self.retrain_day,
                            self.retrain_hour,
                            self.retrain_running,
                            "real" if self.args.real else "dry_run",
                            self.args.assets,
                            self.start_time,
                            log_lines=self._cached_log_lines,
                            runtime_state=runtime_state,
                        )
                        live.update(dashboard)

                        time.sleep(1)

                    except KeyboardInterrupt:
                        self.running = False
                        break

        finally:
            self.stop_bot()
            console.print("\n[yellow]Manager stopped. Bot terminated.[/]")


def main():
    parser = argparse.ArgumentParser(description="Kalshi Trading Bot Manager")
    parser.add_argument("--assets", default="BTC,ETH,SOL,XRP,HYPE,BNB,DOGE", help="Assets to trade")
    parser.add_argument("--real", action="store_true", help="Enable real trading (default: dry run)")
    parser.add_argument("--retrain-day", default="sunday", help="Day to retrain (default: sunday)")
    parser.add_argument("--retrain-hour", type=int, default=6, help="UTC hour to retrain (default: 6)")
    args = parser.parse_args()

    manager = BotManager(args)
    manager.run()


if __name__ == "__main__":
    main()
