"""
Kraken WebSocket client for real-time trade data.

Connects to Kraken WebSocket v2 feed. Kraken supports multiplexing
multiple pairs on a single connection.

Feeds into the same raw_tick_buffer format as Coinbase for ML inference.
"""
import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import websockets

ASSET_TO_PAIR = {
    # Kraken v2 WS uses "BTC/USD" (not "XBT/USD" which was the v1 name).
    # The v2 API explicitly rejects XBT/USD: {'error': 'Currency pair not supported XBT/USD'}.
    "BTC": "BTC/USD",
    "ETH": "ETH/USD",
    "SOL": "SOL/USD",
    "XRP": "XRP/USD",
    "HYPE": "HYPE/USD",
    "BNB": "BNB/USD",
    # DOGE not listed on Kraken
}

WS_URL = "wss://ws.kraken.com/v2"


class KrakenWebSocket:
    """Kraken WebSocket for real-time trade matches."""

    def __init__(self, asset: str):
        self.asset = asset.upper()
        self.pair = ASSET_TO_PAIR.get(self.asset)
        if not self.pair:
            raise ValueError(f"Unsupported asset on Kraken: {asset}")
        self._on_trade: Optional[Callable] = None
        self._running = False

    async def stream_trades(self, on_trade: Callable[[dict[str, Any]], Any]):
        """Connect and stream trades.

        on_trade callback receives:
            {"timestamp": datetime, "price": float, "quantity": float, "side": "buy"|"sell"}
        """
        self._on_trade = on_trade
        self._running = True

        subscribe_msg = json.dumps({
            "method": "subscribe",
            "params": {
                "channel": "trade",
                "symbol": [self.pair],
            },
        })

        while self._running:
            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    await ws.send(subscribe_msg)

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            # Surface subscribe/error responses so silent
                            # subscription failures don't go unnoticed.
                            if msg.get("method") == "subscribe" and not msg.get("success", True):
                                from loguru import logger
                                logger.error(
                                    "[kraken-%s] Subscribe FAILED for %s: %s",
                                    self.asset, self.pair, msg.get("error", msg),
                                )
                            elif msg.get("error"):
                                from loguru import logger
                                logger.warning(
                                    "[kraken-%s] WS error: %s", self.asset, msg
                                )
                            if msg.get("channel") == "trade" and "data" in msg:
                                for t in msg["data"]:
                                    trade = {
                                        "timestamp": datetime.fromisoformat(
                                            t["timestamp"].replace("Z", "+00:00")
                                        ),
                                        "price": float(t["price"]),
                                        "quantity": float(t["qty"]),
                                        "side": t.get("side", "buy"),
                                    }
                                    if self._on_trade:
                                        result = self._on_trade(trade)
                                        if asyncio.iscoroutine(result):
                                            await result
                        except (KeyError, ValueError):
                            continue

            except Exception:
                if self._running:
                    await asyncio.sleep(2)

    def stop(self):
        self._running = False
