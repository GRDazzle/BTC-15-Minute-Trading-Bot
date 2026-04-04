"""
Coinbase WebSocket client for real-time trade data.

Connects to Coinbase Exchange WebSocket feed for dense tick data.
Not geo-blocked like Binance.com — provides high-volume trade data
from a global exchange.

Feeds into the same raw_tick_buffer as Binance for ML model inference.
"""
import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import websockets


# Coinbase product IDs for our assets
ASSET_TO_PRODUCT = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "HYPE": "HYPE-USD",
    "BNB": "BNB-USD",
    "DOGE": "DOGE-USD",
}

WS_URL = "wss://ws-feed.exchange.coinbase.com"


class CoinbaseWebSocket:
    """Coinbase Exchange WebSocket for real-time trade matches."""

    def __init__(self, asset: str):
        self.asset = asset.upper()
        self.product_id = ASSET_TO_PRODUCT.get(self.asset)
        if not self.product_id:
            raise ValueError(f"Unsupported asset: {asset}")
        self._on_trade: Optional[Callable] = None
        self._running = False

    async def stream_trades(self, on_trade: Callable[[dict[str, Any]], Any]):
        """Connect and stream trade matches.

        on_trade callback receives:
            {"timestamp": datetime, "price": float, "quantity": float, "side": "buy"|"sell"}
        """
        self._on_trade = on_trade
        self._running = True

        subscribe_msg = json.dumps({
            "type": "subscribe",
            "channels": [
                {"name": "matches", "product_ids": [self.product_id]},
            ],
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
                            if msg.get("type") == "match" or msg.get("type") == "last_match":
                                trade = {
                                    "timestamp": datetime.fromisoformat(
                                        msg["time"].replace("Z", "+00:00")
                                    ),
                                    "price": float(msg["price"]),
                                    "quantity": float(msg["size"]),
                                    "side": msg["side"],
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
