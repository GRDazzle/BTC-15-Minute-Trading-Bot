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
BALANCE_CSV = PROJECT_ROOT / "output" / "balance.csv"
ACCOUNT_STATE = PROJECT_ROOT / "data" / "account_state.json"
CONFIG_JSON = PROJECT_ROOT / "config" / "trading.json"
RETRAIN_LOG = PROJECT_ROOT / "logs" / "weekly_retrain.log"
RETRAIN_STATE = PROJECT_ROOT / "data" / "retrain_state.json"

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


def build_dashboard(
    bot_process,
    trades,
    account_state,
    retrain_state,
    retrain_day,
    retrain_hour,
    retrain_running,
    mode,
    assets,
    start_time,
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

    # --- Balance Table ---
    bal_table = Table(title="Account Balance", expand=True)
    bal_table.add_column("Asset", style="cyan", width=8)
    bal_table.add_column("Balance", justify="right", width=12)
    bal_table.add_column("PnL", justify="right", width=12)
    bal_table.add_column("Trades", justify="right", width=8)
    bal_table.add_column("Win Rate", justify="right", width=10)

    total_bal = 0.0
    total_pnl = 0.0
    total_trades_count = 0
    total_wins = 0

    for series_key, acct in account_state.items():
        asset = series_key.replace("KX", "").replace("15M", "")
        bal = acct.get("balance_dollars", 0)
        pnl = acct.get("pnl_dollars", 0)
        total_bal += bal
        total_pnl += pnl

        # Count trades for this asset
        asset_trades = [t for t in trades if t.get("asset") == asset]
        wins = sum(1 for t in asset_trades if float(t.get("pnl", "0").replace("+", "")) > 0)
        count = len(asset_trades)
        wr = f"{wins/count*100:.0f}%" if count > 0 else "--"
        total_trades_count += count
        total_wins += wins

        pnl_style = "green" if pnl >= 0 else "red"
        bal_table.add_row(
            asset,
            f"${bal:.2f}",
            f"[{pnl_style}]${pnl:+.2f}[/]",
            str(count),
            wr,
        )

    total_wr = f"{total_wins/total_trades_count*100:.0f}%" if total_trades_count > 0 else "--"
    total_pnl_style = "green" if total_pnl >= 0 else "red"
    bal_table.add_row(
        "[bold]TOTAL[/]",
        f"[bold]${total_bal:.2f}[/]",
        f"[bold {total_pnl_style}]${total_pnl:+.2f}[/]",
        f"[bold]{total_trades_count}[/]",
        f"[bold]{total_wr}[/]",
        style="bold",
    )

    # --- Recent Trades ---
    recent_table = Table(title="Recent Trades (last 10)", expand=True)
    recent_table.add_column("Time", width=8)
    recent_table.add_column("Asset", width=5)
    recent_table.add_column("Dir", width=7)
    recent_table.add_column("Price", justify="right", width=6)
    recent_table.add_column("Qty", justify="right", width=5)
    recent_table.add_column("Result", justify="right", width=8)

    for t in trades[-10:]:
        ts = t.get("timestamp", "")[:19].split("T")[-1][:8]
        asset = t.get("asset", "?")
        direction = t.get("direction", "?")
        price = t.get("price_cents", "?")
        contracts = t.get("contracts", "?")
        pnl_val = float(t.get("pnl", "0").replace("+", ""))
        pnl_style = "green" if pnl_val > 0 else "red"
        dir_style = "green" if direction == "BULLISH" else "red"

        recent_table.add_row(
            ts,
            asset,
            f"[{dir_style}]{direction[:4]}[/]",
            f"{price}c",
            str(contracts),
            f"[{pnl_style}]${pnl_val:+.2f}[/]",
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

    next_dt = next_retrain_time(retrain_day, retrain_hour)
    remaining = (next_dt - now).total_seconds()
    next_str = next_dt.strftime("%Y-%m-%d %H:%M UTC")
    remaining_str = format_duration(remaining)

    retrain_status = "RUNNING" if retrain_running else "idle"
    retrain_color = "yellow" if retrain_running else "green"

    day_name = [k for k, v in DAY_MAP.items() if v == retrain_day][0].capitalize()

    retrain_text = (
        f"  Schedule: Every {day_name} at {retrain_hour:02d}:00 UTC\n"
        f"  Last retrain: {last_line}\n"
        f"  Next retrain: {next_str} ({remaining_str})\n"
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

    # --- Assemble Layout ---
    layout = Layout()
    layout.split_column(
        Layout(Panel(header, style="bold blue"), size=3),
        Layout(name="main"),
        Layout(Panel(Text.from_markup(retrain_text), title="Model Retrain", border_style="magenta"), size=7),
    )
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(Panel(bal_table, border_style="green"), ratio=3),
        Layout(Panel(band_table, border_style="cyan"), ratio=2),
    )
    layout["right"].update(Panel(recent_table, border_style="yellow"))

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
        )

    def stop_bot(self):
        """Stop the trading bot."""
        if self.bot_process and self.bot_process.poll() is None:
            self.bot_process.terminate()
            try:
                self.bot_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.bot_process.kill()

    def run_retrain(self):
        """Run weekly retrain in a background thread."""
        if self.retrain_running:
            return

        def _retrain():
            self.retrain_running = True
            state = load_retrain_state()

            try:
                result = subprocess.run(
                    [sys.executable, "scripts/weekly_retrain.py",
                     "--assets", self.args.assets,
                     "--days", "30"],
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
            state["last_retrain_status"] = status
            save_retrain_state(state)
            self.retrain_running = False

        thread = threading.Thread(target=_retrain, daemon=True)
        thread.start()

    def check_retrain_schedule(self):
        """Check if it's time to retrain."""
        now = datetime.now(timezone.utc)
        if now.weekday() != self.retrain_day:
            return
        if now.hour != self.retrain_hour:
            return

        state = load_retrain_state()
        last = state.get("last_retrain")
        if last:
            last_dt = datetime.fromisoformat(last)
            if (now - last_dt).total_seconds() < 82800:  # 23 hours
                return  # Already ran today

        self.run_retrain()

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
                        account_state = load_account_state()
                        retrain_state = load_retrain_state()

                        # Build and display dashboard
                        dashboard = build_dashboard(
                            self.bot_process,
                            trades,
                            account_state,
                            retrain_state,
                            self.retrain_day,
                            self.retrain_hour,
                            self.retrain_running,
                            "real" if self.args.real else "dry_run",
                            self.args.assets,
                            self.start_time,
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
    parser.add_argument("--assets", default="BTC,ETH,SOL,XRP", help="Assets to trade")
    parser.add_argument("--real", action="store_true", help="Enable real trading (default: dry run)")
    parser.add_argument("--retrain-day", default="sunday", help="Day to retrain (default: sunday)")
    parser.add_argument("--retrain-hour", type=int, default=6, help="UTC hour to retrain (default: 6)")
    args = parser.parse_args()

    manager = BotManager(args)
    manager.run()


if __name__ == "__main__":
    main()
