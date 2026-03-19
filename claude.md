# Project: Kalshi 15-Min Multi-Asset Trading Bot

## Core Strategy
- **Logic:** 15-minute price prediction (Next-Bar) for BTC, ETH, SOL, and XRP.
- **Source Data:** Binance Spot WebSockets/AggTrades (Real-time) and Binance CSVs (Historical).
- **Execution:** Kalshi Binary Markets (0-100 cents settlement).
- **Timeframes:** Syncing with Kalshi's T-minus windows (e.g., 10:00, 10:15, 10:30).

## Coding Standards
- **Language:** Python 3.10+
- **Style:** Functional/Modular. Prioritize "Deterministic Behavior" over complex abstractions.
- **Safety:** ALWAYS use `DRY_RUN=True` by default. Real trades must require an explicit environment variable `REAL_TRADE=TRUE`.
- **Error Handling:** Use explicit try-except blocks for Network/API calls. Log full tracebacks to `logs/trading.log`.

## External Reference (Existing Patterns)
- **Reference Project:** `G:\workspace\BuzzTheGambler\Kalshi\15-min\TradingBot`
- **Instruction:** When implementing Kalshi connectivity, refer to the authentication and market-polling logic in the `TradingBot` directory. Maintain parity with established naming conventions there.

## Project Structure Goals
1. **`sdk/kalshi/`**: A portable, standalone Kalshi SDK.
   - Must handle: RSA-PSS Signing, Session Management, `get_market`, `create_order`, and `get_balance`.
   - Goal: This folder should be copy-pasteable into future projects.
2. **`strategies/`**: Asset-specific logic (BTC vs SOL).
3. **`data/`**: Handlers for Binance WebSocket and historical CSV ingestion.

## Key Technical Constraints (Kalshi)
- **Auth:** Requires `.pem` private key for header signing.
- **Tickers:** Dynamically map Binance symbols to Kalshi series (e.g., `BTC-26MAR18-T1200`).
- **Fees:** Factor in Kalshi's per-contract fee into the expected value (EV) calculation.

## Common Commands
- **Install:** `pip install -r requirements.txt`
- **Test:** `pytest tests/`
- **Run (Demo):** `python main.py --env demo --assets BTC,ETH,SOL`
- **Backtest:** `python scripts/backtest.py --file data/binance_btc_15m.csv`

## Data Structures & Ground Truth
- **Kalshi Polling Data:** Located in `G:\workspace\BuzzTheGambler\Kalshi\15-min\TradingBot\data\KX{ASSET}15M\`
- **Format:** `.jsonl` files (e.g., `2026-03-18_0400_UTC.jsonl`)
- **Key Schema Patterns:**
    1. **Poll Entry:** `{"type": "poll", "market_ticker": "...", "yes_bid": 50, "yes_ask": 51, "mins_to_close": 14.9}`
       - *Use Case:* Simulates the entry price available at any given second before expiry.
    2. **Outcome Entry:** `{"type": "outcome", "event_ticker": "...", "outcome": "yes"}`
       - *Use Case:* Matches the `event_ticker` from a poll to its final settlement for PnL calculation.

## Strategy Refinement Instructions
- **Backtesting Logic:** - Join Binance `aggTrade` timestamps with Kalshi `ts` (timestamps) in the JSONL files.
    - Calculate "Market Alpha": `Binance_Prediction_Probability - (Kalshi_Yes_Ask / 100)`.
- **Validation:** Use the `outcome` entries to verify if a "Yes" prediction actually resulted in a "Yes" settlement.