# Kalshi 15-Minute Multi-Asset Trading Bot

Binary options trading bot for Kalshi's 15-minute crypto price prediction markets. Supports BTC, ETH, SOL, XRP, HYPE, BNB, DOGE. Uses a 3-way dynamic ensemble (XGBoost + LSTM + Fusion) with exponential confidence scaling and per-asset live/dry-run routing for safe live deployment.

## Architecture

### 3-Way Dynamic Ensemble

```
ensemble_p = xgb_w * xgb_p + lstm_w * lstm_p + fusion_w * fusion_p
```

- **XGBoost (v9)**: 37 features, ~88% test accuracy. Looks back 15 minutes via persistent tick buffer. Top features: `price_vs_open`, `velocity_900s`, `velocity_300s`, plus v9 quant indicators (z-score 300s/900s, OBV slope, CVD, SMA crossovers).
- **LSTM (v4)**: Conv1D + BatchNorm + BiLSTM + Attention, ~87% test accuracy. Looks back 3 minutes (180 x 1-second bars). StandardScaler normalization fitted on training data and saved with the model.
- **Fusion**: Rule-based signal processors (TickVelocity, DeribitPCR). Acts as a confidence dampener around 0.50.
- **Dynamic weighting**: `w = min_w + (max_w - min_w) * confidence^4.5`. Uncertain signals get low weight, confident signals get amplified.

### Weekday vs Weekend Models

Each asset has up to 3 model variants:
- **Standard** (`{ASSET}_xgb.json`, `{ASSET}_lstm.pt`) — fallback, trained on all-day data
- **Weekday** (`{ASSET}_weekday_*`) — used Mon-Fri, trained on weekday-only data
- **Weekend** (`{ASSET}_weekend_*`) — used Sat-Sun, trained on weekend-only data

The strategy auto-selects based on UTC day-of-week. Falls back to standard if a variant isn't loaded.

### Data Sources

- **Training**: Coinbase Exchange historical trades (`scripts/fetch_coinbase_trades.py`), 30 days of 1-second tick resolution
- **Live ticks**: Coinbase Exchange WebSocket (`wss://ws-feed.exchange.coinbase.com`), ~100+ ticks/min/asset
- **Live Kalshi prices**: Kalshi WebSocket (real-time bid/ask) with REST poller fallback every 2s
- **Kalshi polling data**: JSONL files for backtesting (2-second snapshots), used by the ensemble combo sweep

---

## Running

```bash
# Terminal UI manager (recommended) -- launches the bot, monitors state, schedules retrains
python manager.py                          # Default: BTC,ETH,SOL,XRP,HYPE,BNB,DOGE
python manager.py --assets BTC,ETH         # Specific assets only
python manager.py --real                   # Pass --real to bot (per-asset dry_run still applies)

# Direct bot (without manager)
python main.py --assets BTC,ETH,SOL,XRP,HYPE,BNB,DOGE

# Manual retrain (foreground, with progress)
python scripts/weekly_retrain.py
```

Live trading is gated **per-asset** via `config/trading.json` (see Live Trading section below). Setting `REAL_TRADE=TRUE` is no longer required if you're using the per-asset `dry_run: false` config flag.

### Manager UI

The terminal UI shows:

- **Header**: bot status, mode, uptime, asset list
- **Left column**:
  1. **Model Retrain** — schedule, last/next retrain, status
  2. **PnL by Entry Price** — band breakdown (55-65, 65-75, 75-85, 85-95)
  3. **PnL by Entry DM** — by decision minute
  4. **Reconciliation** — 3-way ledger/in-memory/Kalshi check (live), 2-way for dry, plus pending verification counts
- **Right column top**:
  - **LIVE Trading** table (only renders when at least one asset is live)
  - **Dry Run** table (always renders, shows all assets in dry mode)
  - Per-asset rolling WR over last 15 trades (`L15` column), with color coding
  - Per-asset drift glyph `!` next to PnL when in-memory diverges from ledger by > $0.10
- **Right column bottom**: Recent Trades panel (merges dry + live, V column shows verification status) and Bot Log

The manager polls `data/account_state.json`, `output/trades.csv`, `output/trades_live.csv`, `data/runtime_state.json`, and `config/trading.json` every second. It does NOT call the Kalshi API directly.

---

## Live Trading

Live mode is opt-in **per asset**. Other assets remain in dry-run with their state unchanged.

### Going live (cutover)

```bash
# 1. Stop the bot first (Ctrl+C in the manager)

# 2. Preview the cutover -- shows what would change without modifying anything
python scripts/go_live.py SOL XRP --balance 50 --dry-run

# 3. Run for real (interactive 'yes' confirmation)
python scripts/go_live.py SOL XRP --balance 50

# 4. Manually deposit at least N x balance into your Kalshi account
#    For SOL+XRP at $50 each: deposit $100 minimum

# 5. Restart the bot
python manager.py
```

What `go_live.py` does:

1. **Snapshots** `trades.csv`, `balance.csv`, `account_state.json` to date-tagged archive files (full safety net before any modification)
2. **Strips** the asset's rows out of `trades.csv` and `balance.csv` -- their pre-cutover history lives only in the archive
3. **Creates** empty `output/trades_live.csv` and `output/balance_live.csv` with the live schemas
4. **Resets** the asset's sub-account in `account_state.json` to `balance=$<bal>`, `pnl=$0`, `reserved=$0`
5. **Edits** `config/trading.json`: sets `"dry_run": false` and `"initial_balance": <bal>` for the asset

After restart, the bot will:
- Log `LIVE TRADING enabled for SOL` etc.
- Run `startup_reconcile()` (see below) to verify Kalshi state
- Start the verification loop for live trades

### Rolling back to dry-run

```bash
# 1. Stop the bot

# 2. Roll back. Default --keep-balance preserves current live balance as the new dry-run starting point
python scripts/go_dry.py SOL XRP

# Alternative: reset to a fresh balance
python scripts/go_dry.py SOL XRP --reset --reset-balance 25

# 3. Restart
python manager.py
```

What `go_dry.py` does:

1. **Refuses** if pending verifications exist (use `--force` to override -- not recommended; you may end up with state drift)
2. **Archives** `trades_live.csv` -> `trades_live_archive_YYYY-MM-DD.csv`, similarly for `balance_live.csv`
3. **Truncates** the live CSVs to header rows (so any other live assets keep working)
4. **Updates** `account_state.json`: keeps current balance (default) or resets to fresh
5. **Edits** `config/trading.json`: sets `dry_run: true` for the asset

**Important:** This script does NOT withdraw money from Kalshi. Your cash stays in your Kalshi account. Withdraw it manually through Kalshi's interface if you want it back in your bank.

### Settlement payout verification (live trades only)

Live trades use a two-phase settlement flow:

1. **Phase 1** (immediately when the outcome arrives): bot computes `expected_revenue = filled * $1.00`, credits the live sub-account immediately so the balance updates fast, marks the trade as needing verification.

2. **Phase 2** (>= 60s later, in `_verification_loop`): bot calls Kalshi `/portfolio/settlements`, finds the matching record by `market_ticker`, extracts the actual revenue Kalshi credited, computes `delta = actual - expected`, applies the delta to the sub-account, marks the trade verified.

Logs on non-zero drift:
```
[verify] SOL KXSOL15M-... drift: expected=$10.0000 actual=$9.9800 delta=$-0.0200 -> applied
```

If Kalshi's settlement record isn't in their history yet, the bot retries every 60s. After 5 minutes, it logs a `STALE` warning once and keeps retrying. Verification is fully **dormant for dry-run trades** -- they're auto-marked verified at settlement time.

### Startup position reconciliation (live mode only)

When the bot starts and at least one asset is live, `startup_reconcile()` runs before any signal processing:

1. **Open positions**: queries `/portfolio/positions`. Should be empty or only the current window's. Anything unexpected (e.g. positions held during downtime) gets logged as a warning.
2. **Resting orders**: queries `/portfolio/orders?status=resting`. Should always be empty for 15-min markets. Any leftover resting orders (from a crash) are **cancelled automatically**.
3. **Recent settlements**: queries `/portfolio/settlements?limit=200`. Logs any settlement whose `market_ticker` isn't in the local `trades.csv` (means a trade settled while the bot was down).
4. **Three-way balance check**: sums all live sub-accounts vs Kalshi balance. Drift > $1 -> warning + pointer to `scripts/reconcile.py --kalshi`.

The script logs everything but does NOT abort startup. False-aborts on transient API errors are worse than running with a known drift you can investigate.

### Ledger reconciliation (`scripts/reconcile.py`)

A standalone read-only script that rebuilds each sub-account's balance from history:

```
ledger(asset) = initial_balance + sum(pnl from trades.csv) + sum(deposits from deposits_log.csv)
```

```bash
# Basic ledger check (no API calls)
python scripts/reconcile.py

# Three-way check including live Kalshi balance (live assets only)
python scripts/reconcile.py --kalshi
```

Output shows per-asset `Initial / +PnL / +Deposits / =Ledger / In-Memory / Diff` with an "L" marker for live assets, separate LIVE total, and exit code:
- **0** if no drift on live assets (dry-run drift is informational only)
- **1** if any live asset has drift > $0.50

### Position sizing in live mode

**Identical to dry-run.** Same power-0.67 scaling formula for both modes:

```python
max_contracts = base * (current_balance / initial_balance) ** 0.67
```

Capped at `MAX_CONTRACTS_CAP = 500`. Example with `max_contracts_per_trade=10`:

| Balance | Ratio | Max contracts |
|---:|---:|---:|
| $50  | 1.0x  | 10 |
| $100 | 2.0x  | 15 |
| $200 | 4.0x  | 25 |
| $400 | 8.0x  | 40 |
| $1,000 | 20x | 73 |
| $5,000 | 100x | 215 |

The actual per-trade size is the minimum of: `max_contracts_per_trade`, `available_balance / cost_per_contract`, and `score-scaled affordability`. At small balances ($25-$50), the per-asset cap is the binding constraint.

`config/trading.json` is hot-reloaded at each window boundary, so you can tune `max_contracts_per_trade` mid-session without restarting.

### Circuit breaker

When a sub-account drops 40% from its session peak balance, the circuit breaker trips and pauses trading for that asset for ~1 hour. Live and dry-run breakers are tracked independently (live breaker only fires on real losses). The breaker resets when:

- 1 hour of pause time elapses
- A deposit is applied via `add_balance.py`
- The bot restarts (session_peak_balance resets)

### Manager UI in live mode

When at least one asset has `dry_run: false`, the manager UI changes:

- **Account Balance** splits into stacked **LIVE Trading** and **Dry Run** tables
- **Reconciliation panel** shows the live 3-way check (Ledger / In-Memory / Kalshi) and dry 2-way (Ledger / In-Memory) plus pending verification counts
- **Recent Trades V column**: `+` (green) for verified, `.` (yellow) for pending, blank for dry trades
- **Per-asset drift glyph `!`** next to PnL when in-memory balance diverges from ledger by more than $0.10
- Border color of the Reconciliation panel reflects severity (green/yellow/red)

The manager reads `data/runtime_state.json` (published by the bot's reconciliation and verification loops) for Kalshi balance and verification stats. The manager itself never calls the Kalshi API.

### Adding balance mid-session

```bash
python scripts/add_balance.py SOL 25      # Queue a $25 deposit to SOL
```

This writes to `data/deposits.json`. The bot polls this queue at each 15-minute window boundary, applies the deposit to the sub-account, resets the circuit breaker for that asset, and appends a row to `data/deposits_log.csv` (used by `reconcile.py`).

The bot's `_process_deposits()` runs in the window loop, so a queued deposit takes effect within 15 minutes. Direct edits to `account_state.json` while the bot is running will be overwritten -- use the deposit queue instead.

---

## Output Files

| File | Description |
|------|-------------|
| `output/trades.csv` | Dry-run trade audit trail (entry, settlement, PnL) |
| `output/trades_live.csv` | Live trade audit trail (richer schema with order_id, fill_price, fees, expected/verified revenue, settlement_verified, verified_at) |
| `output/balance.csv` | Dry-run balance snapshots |
| `output/balance_live.csv` | Live balance snapshots |
| `output/signal_log.csv` | Every ensemble decision (ml_p, fusion_p, action, etc.) |
| `output/trades_archive_YYYY-MM-DD.csv` | Pre-cutover snapshot (created by `go_live.py`) |
| `output/trades_live_archive_YYYY-MM-DD.csv` | Pre-rollback snapshot (created by `go_dry.py`) |
| `data/account_state.json` | All sub-account balances (live + dry mixed in one file) |
| `data/runtime_state.json` | Bot-published reconciliation + verification stats (read by manager UI) |
| `data/deposits.json` | Pending deposit queue |
| `data/deposits_log.csv` | Applied deposit history (used by `reconcile.py`) |
| `data/aggtrades/{ASSET}/` | Coinbase historical trade CSVs (training data, auto-pruned >45 days by retrain) |
| `data/kalshi_polls/` | Kalshi bid/ask polls + outcomes (auto-pruned >45 days by retrain) |
| `logs/trading.log` | Bot runtime log (10 MB rotation, 7-day retention) |
| `logs/weekly_retrain.log` | Retrain pipeline log |

---

## Pipeline Overview

```
1. Download           fetch_coinbase_trades.py        (Coinbase, 30 days)
2. Generate XGB Data  generate_training_data.py       (37 v9 features)
3. Train XGBoost      train_xgb.py                    (per-asset, dm 2+)
4. Generate LSTM Data generate_lstm_training_data.py  (180s sequences)
5. Train LSTM         train_lstm.py                   (Conv1D + BiLSTM, sequential)
6. Ensemble Sweep     ensemble_combo_sweep.py         (writes config/trading.json)
```

The retrain pipeline runs weekly via the manager (Sunday/Monday 15:00 UTC = 8 AM PT). It also retrains weekday and weekend variants of all models.

---

## Automated Retrain

Models are retrained weekly on a schedule. The live bot **hot-reloads** new models and config at the next 15-minute window boundary -- no restart required.

### Schedule

The manager monitors the clock and triggers:
- **Sunday 15:00 UTC** (8 AM PT) -> retrain weekday models
- **Monday 15:00 UTC** (8 AM PT) -> retrain weekend models

The "all" path runs all 3 model variants (standard, weekday, weekend) sequentially.

### Selective replacement (default)

By default, weekly retrain trains **all assets** but only **promotes** (replaces the live model file + re-sweeps the config) for assets with **negative PnL** over the last 7 days. Profitable assets keep their existing models and configs untouched.

```bash
python scripts/weekly_retrain.py                  # Train all, promote negatives only
python scripts/weekly_retrain.py --promote-all    # Force-promote everything
python scripts/weekly_retrain.py --pnl-days 14    # Use 14-day PnL window instead of 7
```

Newly trained models are saved with a date-tagged suffix (e.g. `BTC_2026-04-08_xgb.json`). Promoted models get copied to the live path (`BTC_xgb.json`). Un-promoted dated files are kept as snapshots/rollback targets.

### Pipeline steps

1. **Download Coinbase trades** -- 30 days, 7 assets, ~7 GB total
2. **Generate XGB training data** -- 37-feature CSVs per asset
3. **Train XGB models (date-tagged)** -- walk-forward validation
4. **Generate LSTM training data** -- 180s tick sequences
5. **Train LSTM models (date-tagged, sequential)** -- Conv1D + BiLSTM, GPU-accelerated, run one at a time to avoid memory crashes
6. **Train weekday variants** (all 4 steps with `--day-filter weekday --model-suffix _weekday_<date>`)
7. **Train weekend variants** (same with `--day-filter weekend --model-suffix _weekend_<date>`)
8. **Purge old data** -- aggTrade CSVs and Kalshi polls > 45 days
9. **Filter by PnL** -- compute negative-PnL asset list from `output/trades.csv`
10. **Promote** -- copy date-tagged files to live paths for negative-PnL assets only
11. **Ensemble combo sweep (promoted only)** -- runs `ensemble_combo_sweep.py --kalshi-days 14` for the promoted set, writes new params to `config/trading.json`

The bot picks up new models at the next window boundary via mtime tracking. Both XGB and LSTM hot-reload independently.

### Manual retrain

```bash
python scripts/weekly_retrain.py                          # Default: all assets, all variants
python scripts/weekly_retrain.py --skip-download          # Reuse existing data
python scripts/weekly_retrain.py --assets BTC,ETH         # Subset
python scripts/weekly_retrain.py --day-type weekday       # Just weekday models (Sunday flow)
python scripts/weekly_retrain.py --day-type weekend       # Just weekend models (Monday flow)
```

Retrain logs are streamed to `logs/weekly_retrain.log` and (if a Telegram bot is configured) Telegram notifications are sent on completion.

### Subprocess priority

Weekly retrain spawns training subprocesses with `BELOW_NORMAL_PRIORITY_CLASS` on Windows so the live bot's window loops aren't starved when retrain is running.

---

## Configuration

### config/trading.json

Per-asset trading parameters. Updated automatically by `ensemble_combo_sweep.py`. Hot-reloaded at each window boundary.

```json
{
  "defaults": {
    "initial_balance": 25.0,
    "max_contracts_per_trade": 10,
    "max_price_cents": 80,
    "min_price_cents": 60,
    "require_confirm": false,
    "blocked_dms": []
  },
  "assets": {
    "SOL": {
      "initial_balance": 50.0,
      "max_contracts_per_trade": 10,
      "max_price_cents": 80,
      "min_price_cents": 60,
      "dry_run": false,
      "ensemble": {
        "ml_weight": 0.20,
        "threshold": 0.58,
        "max_price_cents": 80
      },
      "ensemble_weekday": { "ml_weight": 0.30, "threshold": 0.70, "max_price_cents": 75 },
      "ensemble_weekend": { "ml_weight": 0.20, "threshold": 0.65, "max_price_cents": 75 }
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `defaults.initial_balance` | Default starting balance for any asset not in `assets` |
| `defaults.require_confirm` | Require 2 consecutive checkpoint signals to agree before entry (consensus mode) |
| `defaults.blocked_dms` | List of decision minutes to skip globally (e.g. `[4, 5]`) |
| `assets.{ASSET}.initial_balance` | Per-asset starting balance |
| `assets.{ASSET}.max_contracts_per_trade` | Per-asset position size cap (base, before power-0.67 scaling) |
| `assets.{ASSET}.max_price_cents` | Per-asset YES ceiling (skip if ask > this) |
| `assets.{ASSET}.min_price_cents` | Per-asset NO floor (skip if ask < this) |
| `assets.{ASSET}.dry_run` | **Per-asset live/dry override**. `false` enables real Kalshi orders. Set by `go_live.py` / `go_dry.py`. |
| `assets.{ASSET}.ensemble` | Standard model params (ml_weight, threshold, max_price_cents) |
| `assets.{ASSET}.ensemble_weekday` | Weekday variant params, used Mon-Fri |
| `assets.{ASSET}.ensemble_weekend` | Weekend variant params, used Sat-Sun |

The **most restrictive `max_price_cents`** across all relevant ensemble blocks is the one applied to execution.

---

## Project Structure

```
.
|-- main.py                                Entry point (loads config, starts strategy)
|-- manager.py                             Terminal UI manager (launches bot, schedules retrains)
|-- config/trading.json                    Per-asset config + ensemble params
|
|-- strategies/kalshi_strategy.py          Multi-asset strategy with verification + reconciliation loops
|-- execution/kalshi_execution.py          TradeRecord, execute_trade, settle_window, verify_settlement
|
|-- sdk/kalshi/                            Portable Kalshi SDK
|   |-- client.py                          API client + rate limiter (get_balance, get_positions, get_orders, etc.)
|   |-- auth.py                            RSA-PSS request signing
|   |-- orders.py                          Order helpers + settlement lookup + position helpers
|   |-- markets.py                         Market lookup + outcome fetch
|   |-- account.py                         Local virtual sub-account management
|   +-- ticker.py                          Asset -> series mapping (KXSOL15M etc.)
|
|-- core/strategy_brain/
|   |-- signal_processors/                 ML, spike, velocity, sentiment, kalshi price processors
|   +-- fusion_engine/                     Multi-signal fusion engine
|
|-- data_sources/coinbase/
|   +-- websocket.py                       Coinbase Exchange WebSocket (live tick stream)
|
|-- ml/
|   |-- features.py                        v9 feature extraction (37 features, shared training + inference)
|   |-- lstm_features.py                   LSTM 21-feature extraction
|   +-- training_data/                     Generated feature CSVs and LSTM .npz files
|
|-- models/                                Saved XGB / LSTM models per asset (and weekday/weekend variants)
|
|-- backtester/                            Window-based backtesting infrastructure
|
|-- scripts/
|   |-- fetch_coinbase_trades.py           Download Coinbase historical trades for training
|   |-- generate_training_data.py          Replay ticks -> XGB feature CSVs
|   |-- train_xgb.py                       Train per-asset XGBoost
|   |-- generate_lstm_training_data.py     Replay ticks -> LSTM .npz sequences
|   |-- train_lstm.py                      Train per-asset LSTM (sequential to avoid memory crashes)
|   |-- ensemble_combo_sweep.py            Sweep ml_weight x threshold x max_price, write config
|   |-- weekly_retrain.py                  Full retrain pipeline (date-tagged + selective promotion)
|   |
|   |-- go_live.py                         Cutover script: dry -> live for specified assets
|   |-- go_dry.py                          Rollback script: live -> dry
|   |-- reconcile.py                       Read-only ledger check (with optional --kalshi 3-way)
|   |-- add_balance.py                     Queue a deposit to a sub-account
|   +-- backtest_ticks.py                  Standalone backtester
|
|-- data/
|   |-- aggtrades/                         Coinbase tick CSVs (training data)
|   |-- kalshi_polls/                      Kalshi bid/ask + outcome JSONLs (sweep input)
|   |-- account_state.json                 All sub-account balances (live + dry, single file)
|   |-- runtime_state.json                 Bot-published reconciliation + verification stats
|   |-- deposits.json                      Pending deposit queue
|   +-- deposits_log.csv                   Applied deposit audit trail
|
|-- output/
|   |-- trades.csv                         Dry-run trade log
|   |-- trades_live.csv                    Live trade log (richer schema)
|   |-- balance.csv / balance_live.csv     Balance history per mode
|   |-- signal_log.csv                     Pre-execution signal log
|   |-- *_archive_YYYY-MM-DD.*             Cutover/rollback snapshots
|   +-- ensemble_combo_sweep/              Sweep results
|
+-- logs/
    |-- trading.log                        Bot runtime log
    +-- weekly_retrain.log                 Retrain pipeline log
```

---

## Quick Start (full pipeline from scratch)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure Kalshi credentials
#    - Place .pem private key at the path referenced in sdk/kalshi/client.py load_config()
#    - Set KALSHI_API_KEY_ID environment variable

# 3. Download 30 days of Coinbase tick data for all assets
python scripts/fetch_coinbase_trades.py --assets BTC,ETH,SOL,XRP,HYPE,BNB,DOGE --days 30

# 4. Generate training features
python scripts/generate_training_data.py --asset BTC,ETH,SOL,XRP,HYPE,BNB,DOGE --days 30
python scripts/generate_lstm_training_data.py --asset BTC,ETH,SOL,XRP,HYPE,BNB,DOGE --days 30

# 5. Train models (XGB + LSTM)
python scripts/train_xgb.py --asset BTC,ETH,SOL,XRP,HYPE,BNB,DOGE --min-dm 2
for ASSET in BTC ETH SOL XRP HYPE BNB DOGE; do
    python scripts/train_lstm.py --asset $ASSET --min-dm 2
done

# 6. Sweep ensemble params (writes config/trading.json)
python scripts/ensemble_combo_sweep.py --asset BTC,ETH,SOL,XRP,HYPE,BNB,DOGE --kalshi-days 14

# 7. Run the bot in dry-run mode (default for all assets)
python manager.py

# 8. (Optional, when ready) Promote specific assets to live
python scripts/go_live.py SOL XRP --balance 50
python manager.py     # Restart, will detect live config
```

---

## Common operations cheat-sheet

```bash
# Watch the bot in the terminal UI
python manager.py

# Run a one-off ledger check
python scripts/reconcile.py
python scripts/reconcile.py --kalshi    # 3-way (live assets only)

# Add money to a sub-account mid-session
python scripts/add_balance.py SOL 25

# Manual retrain right now
python scripts/weekly_retrain.py
python scripts/weekly_retrain.py --skip-download    # If aggTrade data is fresh
python scripts/weekly_retrain.py --promote-all      # Force-promote all retrained models

# Promote SOL+XRP to live with $50 each
python scripts/go_live.py SOL XRP --balance 50 --dry-run    # Preview
python scripts/go_live.py SOL XRP --balance 50              # Execute

# Roll SOL back to dry, keeping current live balance
python scripts/go_dry.py SOL

# Roll back, resetting to fresh $25
python scripts/go_dry.py SOL --reset --reset-balance 25

# Inspect runtime state (manager reads this)
cat data/runtime_state.json
```

---

## Disclaimer

This software is for educational and research purposes. Trading binary options carries significant risk. Past backtest performance does not guarantee future results. Always start in dry-run mode and validate the strategy for at least 1-2 weeks before committing real money.
