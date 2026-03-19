#!/usr/bin/env python3
"""Smoke test: verify SDK can authenticate and fetch data from Kalshi.

Run:  python scripts/smoke_test.py

This makes REAL API calls — it only reads data, never places orders.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdk.kalshi import KalshiClient, load_config
from sdk.kalshi.markets import fetch_current_market, fetch_event_outcome
from sdk.kalshi.orders import fetch_balance


def main():
    print("=" * 60)
    print("Kalshi SDK Smoke Test")
    print("=" * 60)

    # 1. Load config and create client
    print("\n[1] Loading config...")
    cfg = load_config()
    print(f"    base_url:   {cfg.base_url}")
    print(f"    api_key_id: {cfg.api_key_id[:8]}...")
    print(f"    has_pem:    {cfg.private_key_pem is not None}")

    client = KalshiClient(cfg)

    # 2. Test authentication via balance fetch
    print("\n[2] Fetching balance (tests auth)...")
    balance = fetch_balance(client)
    if balance is not None:
        print(f"    Balance: ${balance:.2f}")
    else:
        print("    FAILED to fetch balance")
        sys.exit(1)

    # 3. Fetch current BTC market
    print("\n[3] Fetching current BTC market...")
    market = fetch_current_market(client, "KXBTC15M")
    if market:
        print(f"    market_ticker:  {market['market_ticker']}")
        print(f"    event_ticker:   {market['event_ticker']}")
        print(f"    close_time:     {market['close_time']}")
        print(f"    yes_ask:        {market['yes_ask']}c")
        print(f"    no_ask:         {market['no_ask']}c")
        print(f"    mins_to_close:  {market['mins_to_close']:.1f}")
    else:
        print("    No open BTC market found (might be outside trading hours)")

    # 4. Test outcome fetch (try the event from step 3 if available)
    if market and market.get("event_ticker"):
        print(f"\n[4] Checking outcome for {market['event_ticker']}...")
        outcome = fetch_event_outcome(client, market["event_ticker"])
        if outcome:
            print(f"    Outcome: {outcome}")
        else:
            print("    No outcome yet (market still open or not settled)")
    else:
        print("\n[4] Skipping outcome test (no market found)")

    print("\n" + "=" * 60)
    print("Smoke test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
