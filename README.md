# Kalshi 15-Minute Multi-Asset Trading Bot

Binary options trading bot for Kalshi's 15-minute crypto price prediction markets. Supports BTC, ETH, SOL, XRP, HYPE, BNB, DOGE. Uses a stacked ensemble (LSTM → XGB) with multi-exchange features, Kalshi market awareness, and per-asset parameter optimization via walk-forward selection.

## Architecture

### Stacked Ensemble (v11)

```
LSTM(180s bars) → lstm_p
XGB(51 features + lstm_p) → final probability → threshold gate → trade
```

**LSTM runs first**, producing P(BULLISH) from 180 seconds of 1-second price bars. This prediction (`lstm_p`) is fed as a feature into XGB, which makes the final decision. XGB learns *when* to trust LSTM and when to ignore it — no hand-tuned weighting formula.

**XGB (v11):** 51 features + lstm_p = 52 inputs per checkpoint. Per-asset trimmed feature sets (36-44 features depending on asset). Hyperparameters selected by walk-forward PnL optimization.

**LSTM (v4):** Conv1D + BatchNorm + BiLSTM + Attention, ~87% accuracy. Looks back 3 minutes (180 x 1-second bars). StandardScaler normalization fitted on training data.

### Multi-Exchange Data (v11)

Three exchange feeds provide a composite market view:

| Exchange | Data Type | Live Feed | Historical |
|---|---|---|---|
| **Coinbase** | Tick trades | WebSocket (primary) | REST download |
| **Kraken** | Tick trades | WebSocket | REST pagination |
| **Bitstamp** | Tick trades / 1-min OHLCV | WebSocket | REST OHLCV |

Cross-exchange features capture price divergence, volume ratios, and market disagreement — signals that a single exchange misses.

### Feature Set (51 features)

| Category | Count | Features |
|---|---:|---|
| Tick-derived | 10 | velocity, volatility, volume, buy ratio, aggressor ratio, tick intensity |
| Price structure | 2 | VWAP deviation, price range |
| Time | 3 | hour sin/cos, minute in window |
| Momentum | 3 | return skew, price vs open, momentum trend |
| SMA | 3 | price vs SMA 5/15/1h |
| Market condition | 4 | volume 180s, choppiness, range pct, vol acceleration |
| Crash detection | 2 | flips per tick, momentum strength |
| Quant indicators | 6 | z-score 300s/900s, OBV slope, CVD |
| Stability (v10) | 4 | price/velocity stability, direction changes, range vs trend |
| Kalshi market (v10) | 4 | yes_ask, spread, mid, mins_to_close |
| Cross-exchange (v11) | 6 | Kraken/Bitstamp price diff, Kraken velocity/volume, exchange price std |
| **LSTM stacked** | **1** | **lstm_p** (stacked prediction) |

### Training Labels

Labels come from **Kalshi's actual settlement outcomes** (CF Benchmarks composite price), not Coinbase price direction. This eliminates the ~17% label mismatch caused by cross-exchange price divergence on small moves.

Settlement data fetched via `scripts/fetch_kalshi_settlements.py` from the Kalshi API.

### Walk-Forward Model Selection

Hyperparameters, threshold, min_dm, and max_price are all selected by a unified walk-forward process:

```
1. For each hyperparameter combo (243 XGB params × 3 min_dm values):
   a. Walk-forward train/test across 4 time folds
   b. Sweep threshold × max_price on each fold's predictions (free, no retraining)
   c. Average OOS PnL across folds = the combo's score
2. Pick the combo with best average OOS PnL
3. Train final model on ALL data with those hyperparams
4. Deploy
```

This replaces both the old logloss-based model selection AND the separate ensemble sweep with a single PnL-optimized pipeline.

### Per-Asset Configuration

Each asset gets its own:
- **Trimmed feature set** (ablation study removes noise features per asset)
- **XGB hyperparameters** (depth, learning rate, regularization)
- **min_dm** (minimum decision minute — skip noisy early signals)
- **threshold** (signal confidence gate)
- **max_price** (maximum Kalshi price to trade at)

These are all selected by the walk-forward sweep and stored in `config/trading.json`.

---

## Data Sources

```
data/aggtrades_coinbase/{ASSET}/  — Coinbase tick trades (primary signal source)
data/aggtrades_kraken/{ASSET}/    — Kraken tick trades (cross-exchange features)
data/ohlcv_bitstamp/{ASSET}/      — Bitstamp 1-min OHLCV (cross-exchange features)
data/kalshi_settlements/          — Kalshi settlement outcomes (training labels)
data/kalshi_polls/                — Kalshi bid/ask polls (kalshi_* features)
```

### Download Historical Data

```bash
# Coinbase (tick trades, primary)
python scripts/fetch_coinbase_trades.py --assets BTC,ETH,SOL,XRP,HYPE,BNB,DOGE --days 60

# Kraken (tick trades, cross-exchange)
python scripts/fetch_kraken_trades.py --assets BTC,ETH,SOL,XRP,HYPE,BNB --days 60

# Bitstamp (1-min OHLCV, cross-exchange)
python scripts/fetch_bitstamp_ohlcv.py --assets BTC,ETH,SOL,XRP,HYPE,BNB,DOGE --days 60

# Kalshi settlement outcomes (training labels)
python scripts/fetch_kalshi_settlements.py --assets BTC,ETH,SOL,XRP,HYPE,BNB,DOGE
```

### Data Retention

| Setting | Value |
|---|---|
| Download window | 60 days |
| Training window | 45 days |
| Data purge threshold | 75 days |
| Kalshi sweep window | 28 days |

---

## Running

```bash
# Terminal UI manager (recommended)
python manager.py

# Direct bot (without manager)
python main.py --assets BTC,ETH,SOL,XRP,HYPE,BNB,DOGE
```

The bot connects to:
- 7 Coinbase WebSockets (primary tick data)
- 7 Kraken WebSockets (cross-exchange ticks)
- 7 Bitstamp WebSockets (cross-exchange ticks)
- 1 Kalshi WebSocket (real-time bid/ask)
- Kalshi REST poller (JSONL recording for backtesting)

### What Happens at Runtime

1. **Warmup (90s):** Collect ticks from all 3 exchanges
2. **Window loop (every 2s):** For each 15-min window, at decision minutes >= per-asset min_dm:
   - Run LSTM on 180s price bars → `lstm_p`
   - Run XGB with 51 features + `lstm_p` → final probability
   - If probability >= threshold (0.75) → BULLISH or BEARISH
   - Validate price against max_price band
   - Size position by balance (power-0.67 scaling)
3. **Settlement loop:** Poll Kalshi outcomes, credit/debit sub-accounts
4. **Verification loop (live only):** Fetch actual Kalshi settlement revenue, correct any drift
5. **Reconciliation (every 30m):** Compare live sub-accounts to Kalshi balance

---

## Training Pipeline

### Full Pipeline (automated)

```bash
# Run everything: data gen → LSTM OOF → walk-forward selection
python scripts/run_full_pipeline.py --assets BTC,ETH,SOL,XRP --days 45 --stacked

# With parallel data gen (requires 64GB+ RAM)
# Pipeline auto-parallelizes XGB and LSTM data gen across assets
```

### Pipeline Steps

```
1. Generate XGB training data     (parallel, 4 assets simultaneously)
   - Loads Coinbase ticks + Kraken ticks + Bitstamp OHLCV + Kalshi polls
   - Computes 51 features at each 10s checkpoint
   - Labels from Kalshi settlement outcomes (CF Benchmarks)

2. Generate LSTM training data    (parallel, 4 assets simultaneously)
   - 180s sequences at each checkpoint
   - Same Kalshi settlement labels

3. Generate stacked features      (sequential, GPU-intensive)
   - 5-fold OOF: train LSTM on 4 folds, predict held-out fold
   - Avoids data leakage (each lstm_p prediction is out-of-sample)
   - Produces {ASSET}_features_stacked.csv with lstm_p column

4. Walk-forward full combo sweep  (sequential, CPU-intensive)
   - 729 XGB hyperparam combos × 3 min_dm values × 4 folds
   - Each model tested against 12 threshold × max_price combos (free)
   - Scored by average OOS PnL across folds
   - Winner's hyperparams used to train final model on ALL data
   - Config updated with best threshold + max_price + min_dm
```

### Manual Steps

```bash
# Download fresh data
python scripts/fetch_coinbase_trades.py --assets BTC,ETH,SOL,XRP --days 60
python scripts/fetch_kraken_trades.py --assets BTC,ETH,SOL,XRP --days 60
python scripts/fetch_bitstamp_ohlcv.py --assets BTC,ETH,SOL,XRP --days 60
python scripts/fetch_kalshi_settlements.py --assets BTC,ETH,SOL,XRP

# Generate training data only
python scripts/generate_training_data.py --asset BTC --days 45

# Train XGB only (uses per-asset trimmed features)
python scripts/train_xgb.py --asset BTC --min-dm 2

# Walk-forward selection only
python scripts/walk_forward_full.py --asset BTC,ETH,SOL,XRP --n-folds 4

# Feature ablation study
python scripts/feature_ablation.py --asset BTC,ETH,SOL,XRP --cutoff 2026-04-07

# Generate training report
python scripts/generate_training_report.py --assets BTC,ETH,SOL,XRP
```

### Weekly Retrain

The manager triggers retrains on schedule:
- **Sunday 15:00 UTC** (8 AM PT) → weekday models
- **Monday 15:00 UTC** (8 AM PT) → weekend models

Retrain trains all assets but only **promotes** (replaces live model) for assets with negative 7-day PnL. Profitable assets keep their existing models. Uses date-tagged model snapshots for rollback.

```bash
python scripts/weekly_retrain.py                  # Train all, promote negatives only
python scripts/weekly_retrain.py --promote-all    # Force-promote everything
python scripts/weekly_retrain.py --pnl-days 14    # Use 14-day PnL window
```

---

## Live Trading

Live mode is opt-in **per asset** via `config/trading.json`.

### Going Live

```bash
python scripts/go_live.py SOL XRP --balance 50      # Preview + execute
python scripts/go_live.py SOL XRP --balance 50 --dry-run  # Preview only
```

### Rolling Back

```bash
python scripts/go_dry.py SOL XRP                    # Keep current balance
python scripts/go_dry.py SOL XRP --reset             # Reset to $25
```

### Settlement Verification (live trades only)

Live trades have two-phase settlement:
1. **Phase 1:** Credit expected revenue immediately (`filled × $1`)
2. **Phase 2:** Verify against Kalshi `/portfolio/settlements`, apply any drift

### Ledger Reconciliation

```bash
python scripts/reconcile.py                          # Ledger vs in-memory check
python scripts/reconcile.py --kalshi                 # 3-way check including Kalshi balance
```

### Position Sizing

```python
max_contracts = base * (current_balance / initial_balance) ** 0.67
```

Capped at 500 contracts. Hot-reloadable via config.

### Circuit Breaker

40% drawdown from session peak → pause trading for 1 hour. Resets on deposit or bot restart.

---

## Configuration

### config/trading.json

```json
{
  "defaults": {
    "initial_balance": 25.0,
    "max_contracts_per_trade": 10,
    "max_price_cents": 80,
    "min_price_cents": 55
  },
  "assets": {
    "BTC": {
      "initial_balance": 25.0,
      "max_contracts_per_trade": 10,
      "max_price_cents": 80,
      "min_price_cents": 55,
      "ensemble": {
        "ml_weight": 0.1,
        "threshold": 0.75,
        "max_price_cents": 80,
        "min_dm": 3,
        "walk_forward_pnl": 247.98,
        "walk_forward_wr": 83.4
      }
    }
  }
}
```

All config values are hot-reloaded at each 15-minute window boundary (no restart needed for config changes).

---

## Output Files

| File | Description |
|------|-------------|
| `output/trades.csv` | Dry-run trade audit trail |
| `output/trades_live.csv` | Live trade audit trail (richer schema) |
| `output/balance.csv` / `balance_live.csv` | Balance history |
| `output/signal_log.csv` | Every ensemble decision |
| `output/training_reports/` | Timestamped training reports |
| `data/account_state.json` | Sub-account balances |
| `data/runtime_state.json` | Bot-published reconciliation stats |
| `data/deposits.json` | Pending deposit queue |
| `data/deposits_log.csv` | Applied deposit history |
| `logs/trading.log` | Bot runtime log (10 MB rotation) |

---

## Project Structure

```
.
├── main.py                              Entry point
├── manager.py                           Terminal UI + retrain scheduler
├── config/trading.json                  Per-asset config (hot-reloadable)
│
├── strategies/kalshi_strategy.py        Strategy: stacked ensemble + multi-exchange WS
├── execution/kalshi_execution.py        Trade execution + settlement verification
│
├── sdk/kalshi/                          Portable Kalshi SDK
│   ├── client.py                        API client + rate limiter
│   ├── auth.py                          RSA-PSS signing
│   ├── orders.py                        Orders + settlements + positions
│   ├── markets.py                       Market lookup + outcomes
│   ├── account.py                       Sub-account management
│   └── ticker.py                        Asset → series mapping
│
├── core/strategy_brain/
│   └── signal_processors/
│       ├── ml_processor.py              XGB inference (per-asset trimmed features)
│       ├── lstm_processor.py            LSTM inference
│       └── ...                          Tick velocity, spike detection, etc.
│
├── data_sources/
│   ├── coinbase/websocket.py            Coinbase WS (primary ticks)
│   ├── kraken/websocket.py              Kraken WS (cross-exchange)
│   └── bitstamp/websocket.py            Bitstamp WS (cross-exchange)
│
├── ml/
│   ├── features.py                      51-feature extraction (v11)
│   ├── kalshi_features.py               Kalshi poll index + event ticker mapping
│   ├── multi_exchange.py                Kraken tick index + Bitstamp OHLCV index
│   ├── lstm_features.py                 LSTM sequence extraction
│   └── lstm_model.py                    PriceLSTM architecture
│
├── scripts/
│   ├── run_full_pipeline.py             Full pipeline: data gen → OOF → walk-forward
│   ├── generate_training_data.py        XGB features (multi-exchange + Kalshi labels)
│   ├── generate_lstm_training_data.py   LSTM sequences
│   ├── generate_stacked_features.py     OOF LSTM predictions for stacking
│   ├── walk_forward_full.py             Walk-forward hyperparameter + execution param sweep
│   ├── walk_forward_select.py           Walk-forward hyperparameter sweep
│   ├── train_xgb.py                     XGB training (per-asset trimmed features)
│   ├── train_lstm.py                    LSTM training
│   ├── weekly_retrain.py                Automated retrain pipeline
│   ├── feature_ablation.py              Feature group ablation study
│   ├── feature_trim_perasset.py         Per-asset greedy feature trimming
│   ├── hyperparam_tune.py               Hyperparameter grid search
│   ├── fetch_coinbase_trades.py         Download Coinbase historical trades
│   ├── fetch_kraken_trades.py           Download Kraken historical trades
│   ├── fetch_bitstamp_ohlcv.py          Download Bitstamp 1-min OHLCV
│   ├── fetch_kalshi_settlements.py      Download Kalshi settlement outcomes
│   ├── signal_analysis.py               Signal pattern analysis
│   ├── reconcile.py                     Ledger reconciliation
│   ├── generate_training_report.py      Training report generator
│   ├── go_live.py                       Cutover: dry → live
│   ├── go_dry.py                        Rollback: live → dry
│   └── add_balance.py                   Queue mid-session deposits
│
├── models/                              XGB + LSTM models per asset
├── data/                                Tick data, polls, settlements, state
├── output/                              Trades, balance, signals, reports
└── logs/                                Runtime logs
```

---

## Common Operations

```bash
# Watch the bot
python manager.py

# Check ledger reconciliation
python scripts/reconcile.py

# Add money to an asset mid-session
python scripts/add_balance.py SOL 25

# Run the full training pipeline
python scripts/run_full_pipeline.py --assets BTC,ETH,SOL,XRP --days 45 --stacked

# Generate a training report
python scripts/generate_training_report.py

# Download all exchange data
python scripts/fetch_coinbase_trades.py --assets BTC,ETH,SOL,XRP --days 60
python scripts/fetch_kraken_trades.py --assets BTC,ETH,SOL,XRP --days 60
python scripts/fetch_bitstamp_ohlcv.py --assets BTC,ETH,SOL,XRP --days 60
python scripts/fetch_kalshi_settlements.py

# Promote to live trading
python scripts/go_live.py SOL XRP --balance 50

# Roll back to dry-run
python scripts/go_dry.py SOL XRP
```

---

## Disclaimer

This software is for educational and research purposes. Trading binary options carries significant risk. Past backtest performance does not guarantee future results. Always start in dry-run mode and validate the strategy thoroughly before committing real money.
