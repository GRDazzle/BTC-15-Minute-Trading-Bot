"""
Bitstamp WebSocket client for real-time trade data.

Connects to Bitstamp WebSocket feed. Free, no auth needed.
Each asset requires a separate channel subscription.

Feeds into the same raw_tick_buffer format as Coinbase for ML inference.
"""
import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import websockets

ASSET_TO_CHANNEL = {
    "BTC": "live_trades_btcusd",
    "ETH": "live_trades_ethusd",
    "SOL": "live_trades_solusd",
    "XRP": "live_trades_xrpusd",
    "HYPE": "live_trades_hypeusd",
    "BNB": "live_trades_bnbusd",
    "DOGE": "live_trades_dogeusd",
}

WS_URL = "wss://ws.bitstamp.net"


class BitstampWebSocket:
    """Bitstamp WebSocket for real-time trade matches."""

    def __init__(self, asset: str):
        self.asset = asset.upper()
        self.channel = ASSET_TO_CHANNEL.get(self.asset)
        if not self.channel:
            raise ValueError(f"Unsupported asset on Bitstamp: {asset}")
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
            "event": "bts:subscribe",
            "data": {"channel": self.channel},
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
                            if msg.get("event") == "trade":
                                data = msg.get("data", {})
                                # type: 0 = buy, 1 = sell
                                side = "buy" if int(data.get("type", 0)) == 0 else "sell"
                                trade = {
                                    "timestamp": datetime.fromtimestamp(
                                        int(data["timestamp"]), tz=timezone.utc
                                    ),
                                    "price": float(data["price"]),
                                    "quantity": float(data["amount"]),
                                    "side": side,
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
