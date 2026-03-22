# Kalshi 15-Minute Multi-Asset Trading Bot

Binary options trading bot for Kalshi's 15-minute crypto price prediction markets (BTC, ETH, SOL, XRP). Uses Binance tick data for signal generation, an XGBoost ML model blended with rule-based fusion signals, and half-Kelly position sizing.

## Pipeline Overview

```
1. Download Data        download_binance_aggtrades.py  (Data Vision, 30 days)
         |
         v
2. Generate Features    generate_training_data.py      (31 features)
         |
         v
3. Train Model          train_xgb.py                   (per-asset XGBoost)
         |
         v
4. Backtest             backtest_ticks.py
         |
         v
5. Ensemble Sweep       ensemble_sweep.py  -->  config/trading.json  (accuracy-based)
         |
         v
6. PnL Sweep            pnl_sweep.py  -->  config/trading.json  (dollar PnL-based)
         |
         v
7. Run Bot              main.py (dry-run or live)
         |
         +--> Live data collection (Binance WS + Kalshi polls)
         +--> Rolling PnL tune every 2h (auto, uses live data)
         +--> Hot-reload config at each window boundary
```

---

## Prerequisites

- Python 3.10+
- Dependencies: `pip install -r requirements.txt`
- Kalshi API credentials (`.pem` private key + API key ID)
- Internet access for Binance data download and Kalshi API

---

## Step 1: Download Historical Tick Data

Downloads daily aggTrades CSVs from Binance Data Vision at full tick resolution.

```bash
python scripts/download_binance_aggtrades.py --assets BTC,ETH,SOL,XRP --days 30
```

| Flag | Default | Description |
|------|---------|-------------|
| `--assets` | *required* | Comma-separated assets (BTC, ETH, SOL, XRP) |
| `--days` | 30 | Number of days to download |

**Output:** `data/aggtrades/{ASSET}/{SYMBOL}-aggTrades-YYYY-MM-DD.csv`

Each CSV contains: `agg_trade_id, price, qty, first_id, last_id, timestamp, is_buyer_maker, best_price_match`

---

## Step 2: Generate Training Data

Replays tick windows and extracts a 31-feature vector at every 10-second checkpoint within each 15-minute window (decision minutes 2-9), yielding ~60x more training samples than window-level data.

```bash
python scripts/generate_training_data.py --asset BTC --days 30
python scripts/generate_training_data.py --asset BTC,ETH,SOL,XRP --days 30 --min-move 0.0001
```

| Flag | Default | Description |
|------|---------|-------------|
| `--asset` | *required* | Comma-separated assets |
| `--days` | all | Limit to last N days of data |
| `--min-move` | 0.0001 | Min price move % to include a window (filters noise) |

**Output:** `ml/training_data/{ASSET}_features.csv`

Each row is one checkpoint with 31 features, a BULLISH/BEARISH label, and metadata (window_id, decision_minute).

### Feature Set (31 features)

| Category | Features |
|----------|----------|
| Tick-derived (16) | velocity_5s/10s/30s/60s, acceleration, volatility_30s/60s, volume_30s/60s, volume_acceleration, buy_volume_ratio_30s/60s, tick_intensity_30s, large_trade_count, vwap_deviation, aggressor_ratio_30s/60s |
| Price-history (4) | ma_deviation_20, momentum_5/10, price_range_20 |
| Time (3) | hour_sin, hour_cos, minute_in_window |
| Rule-based (2) | tickvel_direction, tickvel_confidence |
| Additional (6) | rsi_14, bollinger_pos, return_skew_60s, price_vs_open, btc_velocity_60s |

---

## Step 3: Train XGBoost Model

Trains a per-asset XGBoost binary classifier using walk-forward validation (70% train / 17% val / 13% test split by time).

```bash
python scripts/train_xgb.py --asset BTC
python scripts/train_xgb.py --asset BTC,ETH,SOL,XRP --tune --min-dm 2
```

| Flag | Default | Description |
|------|---------|-------------|
| `--asset` | *required* | Comma-separated assets |
| `--tune` | off | Run hyperparameter grid search (81 combos) |
| `--min-dm` | 2 | Exclude decision minutes below this value |
| `--dedup` | none | Window dedup strategy: none, last, middle, random |
| `--exclude-hours` | none | Comma-separated UTC hours to exclude (e.g. `1,9,10,17`) |

**Output:**
- `models/{ASSET}_xgb.json` -- saved model (loaded by live bot and backtester)
- `output/ml/{ASSET}_feature_importance.png` -- feature importance chart

Default XGBoost params: `n_estimators=400, max_depth=4, learning_rate=0.05`, early stopping after 20 rounds on logloss.

**Important:** `--min-dm 2` should match the backtest and sweep settings. Decision minutes 0-1 have low accuracy (49-78%) and are excluded from training by default.

---

## Step 4: Backtest

Runs the trained model (or fusion engine, or ensemble) against historical tick data and reports accuracy, trade rate, and per-dm breakdown.

```bash
# Fusion only (no ML)
python scripts/backtest_ticks.py --asset BTC --days 30

# ML only
python scripts/backtest_ticks.py --asset BTC --days 30 --ml --min-dm 2

# Ensemble (ML + fusion blend)
python scripts/backtest_ticks.py --asset BTC --days 30 --ensemble --ml-weight 0.70 --ens-threshold 0.70 --min-dm 2
```

| Flag | Default | Description |
|------|---------|-------------|
| `--asset` | *required* | Comma-separated assets |
| `--days` | all | Limit to last N days |
| `--no-blackout` | off | Disable blackout window filtering |
| `--ml` | off | Use ML model instead of fusion engine |
| `--ensemble` | off | Use weighted ML + fusion blend |
| `--ml-weight` | 0.65 | ML weight in ensemble (only with `--ensemble`) |
| `--ens-threshold` | 0.60 | Confidence threshold (only with `--ensemble`) |
| `--min-dm` | 0 | Min decision minute (use 2 to match training) |

**Output:** `output/backtest_ticks/{ASSET}_ticks[_ml|_ensemble_*]_backtest.csv`

---

## Step 5: Ensemble Sweep

Finds the optimal ML/fusion weight and confidence threshold per asset by **accuracy** (correct predictions). Uses Data Vision tick data (high volume, from Binance.com). Does NOT use Kalshi price data.

```bash
python scripts/ensemble_sweep.py --asset BTC,ETH,SOL,XRP --days 30 --min-dm 2
```

| Flag | Default | Description |
|------|---------|-------------|
| `--asset` | *required* | Comma-separated assets |
| `--days` | all | Limit to last N days |
| `--min-dm` | 2 | Min decision minute |

**Sweep grid:**
- `ml_weight`: 0.00, 0.05, 0.10, ..., 1.00 (21 values)
- `threshold`: 0.55, 0.58, 0.60, 0.62, 0.65, 0.70 (6 values)

**Output:**
- `output/ensemble_sweep/{ASSET}_ensemble_sweep.csv` -- all 126 combos with metrics
- `config/trading.json` -- auto-updated with best per-asset params and half-Kelly entry bands

Results are ranked by `net_correct = correct_predictions - wrong_predictions`, which balances accuracy and trade volume.

### Half-Kelly Entry Bands

Derived from backtested accuracy. Trades outside these bands have negative expected value after fees.

- `max_price = floor(accuracy * 100) - 2c fee` (max YES price)
- `min_price = ceil((1 - accuracy) * 100) + 2c fee + 1c` (min NO price)

---

## Step 6: PnL Sweep

Finds the optimal ensemble params by **dollar PnL** using real Kalshi bid/ask prices and settlement outcomes. This step runs after the ensemble sweep and **overwrites** its config with PnL-optimized params.

```bash
python scripts/pnl_sweep.py --asset BTC,ETH,SOL,XRP --min-dm 2
python scripts/pnl_sweep.py --asset BTC --hours 12 --sequential
```

| Flag | Default | Description |
|------|---------|-------------|
| `--asset` | *required* | Comma-separated assets |
| `--min-dm` | 2 | Min decision minute |
| `--hours` | all | Only use last N hours of Kalshi data |
| `--sequential` | off | Run assets one at a time (vs parallel) |

**Data sources:**
- **Binance ticks:** `data/aggtrades/` (live WS data + REST-fetched data from Binance.US)
- **Kalshi prices:** `data/kalshi_polls/` (bid/ask snapshots + settlement outcomes)

**Sweep grid:** 336 combos (21 weights x 16 thresholds, finer resolution than accuracy sweep)

**Output:**
- `output/pnl_sweep/{ASSET}_pnl_sweep.csv` -- all combos with PnL metrics
- `config/trading.json` -- auto-updated with PnL-optimized params

---

## Step 7: Run the Bot

```bash
# Dry run (default) -- all 4 assets
python main.py --assets BTC,ETH,SOL,XRP

# Dry run with immediate param tune
python main.py --assets BTC,ETH,SOL,XRP --tune-delay 0

# Dry run, delay first tune by 2 hours (default)
python main.py --assets BTC,ETH,SOL,XRP --tune-delay 7200

# Dry run with debug logging
python main.py --assets BTC,ETH,SOL,XRP --log-level DEBUG

# Live trading (requires explicit env var)
REAL_TRADE=TRUE python main.py --assets BTC,ETH,SOL,XRP
```

| Flag | Default | Description |
|------|---------|-------------|
| `--assets` | BTC | Comma-separated assets |
| `--env` | prod | API environment: prod or demo |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |
| `--state-file` | data/account_state.json | Sub-account state persistence |
| `--config` | config/trading.json | Per-asset trading config |
| `--tune-delay` | 7200 | Seconds before first param tune (0 = immediate) |

**Safety:** `DRY_RUN=True` by default. The bot fetches real Kalshi market data and outcomes but simulates orders locally. Set `REAL_TRADE=TRUE` to place real orders.

### What Happens at Runtime

1. **Warmup (90s):** Collects Binance WebSocket tick data, no trades fired
2. **Window loop (every 2s):** For each 15-min window, at decision minutes 2-9:
   - Fetches current Kalshi market prices (yes_ask, no_ask)
   - Runs ensemble blend: `ensemble_p = ml_weight * ML_p + (1 - ml_weight) * fusion_p`
   - If `ensemble_p >= threshold` -> BULLISH (buy YES)
   - If `ensemble_p <= 1 - threshold` -> BEARISH (buy NO)
   - Validates price against half-Kelly bands from config
   - Sizes position by confidence and sub-account balance
3. **Settlement loop:** Polls Kalshi outcomes after window close, credits/debits sub-accounts
4. **Reconciliation (every 30m):** Compares local sub-account totals vs Kalshi API balance
5. **Live data collection (continuous):**
   - **Kalshi poller (every 5s):** Writes bid/ask snapshots + outcomes to `data/kalshi_polls/`
   - **Binance trade writer:** Persists every WS aggTrade to `data/aggtrades/` daily CSVs
6. **Rolling param tune (every 2h):** Runs `pnl_sweep.py --hours 12` using collected live data, updates `config/trading.json`
7. **Hot-reload:** At each window boundary, checks config/model file timestamps and reloads if changed (no restart needed)

### Output Files

| File | Description |
|------|-------------|
| `logs/trading.log` | Full debug log (10 MB rotation, 7-day retention) |
| `output/trades.csv` | Trade audit trail with entry, settlement, PnL |
| `output/balance.csv` | Balance snapshots at startup, trades, settlements |
| `output/signal_log.csv` | Every ensemble decision (ml_p, fusion_p, action, etc.) |
| `data/account_state.json` | Persisted sub-account state (survives restarts) |
| `data/aggtrades/{ASSET}/` | Live Binance aggTrade CSVs (auto-pruned >14 days) |
| `data/kalshi_polls/` | Kalshi bid/ask polls + outcomes (auto-pruned >14 days) |

---

## Fetch Recent aggTrades (REST)

Fills the gap between Data Vision daily CSVs (always 1+ day stale) and live data. Useful when the bot was offline or for bootstrapping. Uses Binance.US REST API.

```bash
python scripts/fetch_recent_aggtrades.py --assets BTC,ETH,SOL,XRP --hours 48
```

| Flag | Default | Description |
|------|---------|-------------|
| `--assets` | *required* | Comma-separated assets |
| `--hours` | 12 | Hours of history to fetch |

Resumes from the last timestamp in existing CSVs (handles the Binance.com vs Binance.US ID space difference). Writes to the same directory and format as Data Vision and live WS data.

**Note:** When the bot is running, the live Binance trade writer handles this automatically. The REST fetcher is primarily for `weekly_retrain.py` (Step 5.5) and manual catch-up.

---

## Automated Retrain

The full pipeline can be run on a schedule (weekly or as needed) to retrain models on the latest data. The live bot hot-reloads new parameters at the next 15-minute window boundary -- no restart required.

```bash
# Full retrain (download + generate + train + sweep + pnl_sweep)
python scripts/weekly_retrain.py

# Skip download (reuse existing data)
python scripts/weekly_retrain.py --skip-download

# Custom assets/days
python scripts/weekly_retrain.py --assets BTC,ETH --days 14
```

| Flag | Default | Description |
|------|---------|-------------|
| `--assets` | BTC,ETH,SOL,XRP | Comma-separated assets |
| `--days` | 30 | Days of data to use |
| `--min-dm` | 2 | Min decision minute for training/sweep |
| `--skip-download` | off | Skip data download step |

### Retrain Pipeline Steps

1. **Download aggTrades** -- Data Vision daily CSVs (30 days, Binance.com)
2. **Generate training data** -- Replay tick windows into 31-feature CSVs
3. **Train XGBoost** -- Per-asset models with walk-forward validation
4. **Ensemble sweep** -- 126 combos, accuracy-based (Data Vision ticks only)
5. **Purge old data** -- Kalshi polls + aggTrade CSVs older than 14 days
6. **Fetch recent aggTrades (REST)** -- Binance.US, 48h (fills gap if bot was offline)
7. **PnL sweep** -- 336 combos, dollar PnL-based (Binance.US ticks + Kalshi prices)
8. **Param tune** -- 12h rolling tune from signal_log.csv

**Hot-reload:** The live bot checks `config/trading.json` and `models/*.json` modification times at each window boundary. When files change, it reloads ensemble weights, price bands, and ML models without stopping. Logs all reloads to `logs/trading.log`.

### Scheduling with Windows Task Scheduler

To run the retrain automatically on a weekly schedule:

**Create the task:**

1. Open Task Scheduler (`taskschd.msc` or search "Task Scheduler" in Start)
2. Click **Create Basic Task**
3. Name: `Kalshi Weekly Retrain`
4. Trigger: **Weekly**, pick a day/time (e.g. Sunday 03:00)
5. Action: **Start a program**
   - Program/script: `python`
   - Add arguments: `scripts/weekly_retrain.py`
   - Start in: `G:\workspace\BuzzTheGambler\Kalshi\15-min\BTC-15-Minute-Trading-Bot`
6. Finish

Or create it from PowerShell (run as admin):

```powershell
$action = New-ScheduledTaskAction `
    -Execute "python" `
    -Argument "scripts/weekly_retrain.py" `
    -WorkingDirectory "G:\workspace\BuzzTheGambler\Kalshi\15-min\BTC-15-Minute-Trading-Bot"

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3am

Register-ScheduledTask `
    -TaskName "Kalshi Weekly Retrain" `
    -Action $action `
    -Trigger $trigger `
    -Description "Retrain ML models and update ensemble params"
```

**Check status:**

```powershell
Get-ScheduledTask -TaskName "Kalshi Weekly Retrain"
```

**Run it manually (outside schedule):**

```powershell
Start-ScheduledTask -TaskName "Kalshi Weekly Retrain"
```

**Disable (pause without deleting):**

```powershell
Disable-ScheduledTask -TaskName "Kalshi Weekly Retrain"
```

**Re-enable:**

```powershell
Enable-ScheduledTask -TaskName "Kalshi Weekly Retrain"
```

**Delete entirely:**

```powershell
Unregister-ScheduledTask -TaskName "Kalshi Weekly Retrain" -Confirm:$false
```

**Logs:** Check `logs/weekly_retrain.log` for pipeline output and timing.

---

## Configuration

### config/trading.json

Per-asset trading parameters. Updated automatically by `ensemble_sweep.py` and `pnl_sweep.py`.

```json
{
  "defaults": {
    "initial_balance": 25.0,
    "max_contracts_per_trade": 10,
    "max_price_cents": 85,
    "min_price_cents": 15
  },
  "assets": {
    "BTC": {
      "initial_balance": 50.0,
      "max_contracts_per_trade": 20,
      "max_price_cents": 83,
      "min_price_cents": 18,
      "ensemble": {
        "ml_weight": 0.45,
        "threshold": 0.58,
        "min_dm": 2,
        "accuracy": 91.7,
        "pnl_sweep_source": "pnl_sweep"
      }
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `initial_balance` | Starting sub-account balance ($) |
| `max_contracts_per_trade` | Position size cap |
| `max_price_cents` | Half-Kelly YES ceiling (skip if ask > this) |
| `min_price_cents` | Half-Kelly NO floor (skip if ask < this) |
| `ensemble.ml_weight` | ML probability weight in blend (0.0-1.0) |
| `ensemble.threshold` | Min blended probability to trigger trade |
| `ensemble.min_dm` | Min decision minute for trading |
| `ensemble.pnl_sweep_source` | Indicates params came from PnL sweep (vs accuracy sweep) |

---

## Project Structure

```
.
|-- main.py                             Entry point
|-- config/trading.json                 Per-asset config + ensemble params
|-- strategies/kalshi_strategy.py       Multi-asset strategy orchestrator
|-- execution/kalshi_execution.py       Trade execution + dry-run settlement
|-- sdk/kalshi/                         Portable Kalshi SDK
|   |-- client.py                         API client + rate limiter
|   |-- auth.py                           RSA-PSS signing
|   |-- orders.py                         Order placement + balance
|   |-- markets.py                        Market lookup + outcomes
|   |-- account.py                        Sub-account management
|   +-- ticker.py                         Asset -> series mapping
|-- core/strategy_brain/
|   |-- signal_processors/
|   |   |-- ml_processor.py              XGBoost signal processor
|   |   |-- spike_detector.py            Price spike detection
|   |   |-- tick_velocity_processor.py   Tick velocity signals
|   |   +-- kalshi_price_processor.py    Kalshi price as signal
|   +-- fusion_engine/
|       +-- signal_fusion.py             Multi-signal fusion
|-- data_sources/binance/
|   +-- websocket.py                     Binance WS (ticker + aggTrade combined stream)
|-- ml/
|   |-- features.py                     31-feature extraction (shared)
|   +-- training_data/                  Generated feature CSVs
|-- models/                             Saved XGBoost models per asset
|-- backtester/
|   |-- simulator.py                    Backtest engine
|   |-- data_loader.py                  Kline data loader
|   |-- data_loader_ticks.py            AggTrades data loader
|   |-- data_loader_kalshi.py           Kalshi JSONL data loader
|   +-- reporter.py                     Results formatting + CSV export
|-- scripts/
|   |-- download_binance_aggtrades.py   Step 1: fetch Data Vision tick data
|   |-- fetch_recent_aggtrades.py       REST fetcher for Binance.US aggTrades
|   |-- generate_training_data.py       Step 2: extract features
|   |-- train_xgb.py                    Step 3: train models
|   |-- backtest_ticks.py               Step 4: backtest
|   |-- ensemble_sweep.py               Step 5: accuracy-based weight sweep
|   |-- pnl_sweep.py                    Step 6: PnL-based weight sweep
|   |-- weekly_retrain.py               Automated retrain pipeline
|   |-- param_tune.py                   Rolling param tune from signal log
|   +-- param_sweep.py                  Signal parameter sweep (720 combos)
|-- data/
|   |-- aggtrades/                      Binance tick CSVs (Data Vision + live WS + REST)
|   |-- kalshi_polls/                   Kalshi bid/ask polls + outcomes (JSONL)
|   +-- account_state.json              Dry-run sub-account state
|-- output/
|   |-- trades.csv                      Live/dry-run trade log
|   |-- balance.csv                     Balance snapshots
|   |-- signal_log.csv                  Ensemble decisions log
|   |-- backtest_ticks/                 Backtest results
|   |-- ensemble_sweep/                 Accuracy sweep results
|   +-- pnl_sweep/                      PnL sweep results
+-- logs/                               Runtime logs
```

---

## Quick Start (Full Pipeline)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download 30 days of tick data
python scripts/download_binance_aggtrades.py --assets BTC,ETH,SOL,XRP --days 30

# 3. Generate training features
python scripts/generate_training_data.py --asset BTC,ETH,SOL,XRP --days 30

# 4. Train models
python scripts/train_xgb.py --asset BTC,ETH,SOL,XRP --min-dm 2

# 5. Accuracy sweep (writes config/trading.json)
python scripts/ensemble_sweep.py --asset BTC,ETH,SOL,XRP --days 30 --min-dm 2

# 6. PnL sweep (overwrites with PnL-optimized params, needs Kalshi data)
python scripts/pnl_sweep.py --asset BTC,ETH,SOL,XRP --min-dm 2

# 7. Verify with backtest
python scripts/backtest_ticks.py --asset BTC,ETH,SOL,XRP --ensemble --ml-weight 0.70 --ens-threshold 0.70 --min-dm 2

# 8. Dry run (collects Kalshi + Binance data for future PnL sweeps)
python main.py --assets BTC,ETH,SOL,XRP

# 9. Go live (when ready)
REAL_TRADE=TRUE python main.py --assets BTC,ETH,SOL,XRP
```

---

## Disclaimer

This software is for educational and research purposes. Trading binary options carries significant risk. Past backtest performance does not guarantee future results. Always start with dry-run mode and understand the risks before trading with real money.
