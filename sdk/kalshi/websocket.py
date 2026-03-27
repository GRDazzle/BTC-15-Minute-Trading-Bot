"""
Kalshi WebSocket client for real-time market data.

Subscribes to the `ticker` channel for live yes_bid/yes_ask prices,
replacing REST polling. Uses the same RSA-PSS auth as the REST client.

Usage:
    ws = KalshiWebSocket(config)
    await ws.connect()
    await ws.subscribe_ticker(["KXBTC15M-26MAR271200-00", ...])
    # Prices arrive via on_ticker callback
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import websockets

from sdk.kalshi.client import KalshiConfig, _load_private_key, sign_request


# WebSocket endpoints
WS_URL_PROD = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_URL_DEMO = "wss://demo-api.kalshi.co/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"


class KalshiWebSocket:
    """Kalshi Exchange WebSocket for real-time ticker updates."""

    def __init__(self, config: KalshiConfig, demo: bool = False):
        self.config = config
        self.ws_url = WS_URL_DEMO if demo else WS_URL_PROD
        self._private_key = None
        if config.private_key_pem:
            self._private_key = _load_private_key(config.private_key_pem)

        self._ws = None
        self._cmd_id = 0
        self._running = False
        self._ticker_sid: Optional[int] = None

        # Callbacks
        self.on_ticker: Optional[Callable[[dict[str, Any]], None]] = None
        self.on_orderbook: Optional[Callable[[dict[str, Any]], None]] = None
        self.on_error: Optional[Callable[[dict[str, Any]], None]] = None

    def _next_id(self) -> int:
        self._cmd_id += 1
        return self._cmd_id

    def _auth_headers(self) -> dict[str, str]:
        """Build WebSocket auth headers using RSA-PSS signing."""
        if self._private_key is None:
            raise RuntimeError("Private key not loaded")
        ts_ms = str(int(time.time() * 1000))
        sig = sign_request(self._private_key, ts_ms, "GET", WS_PATH)
        return {
            "KALSHI-ACCESS-KEY": self.config.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }

    async def connect_and_stream(self):
        """Connect to Kalshi WebSocket and process messages until stopped."""
        self._running = True

        while self._running:
            try:
                headers = self._auth_headers()
                async with websockets.connect(
                    self.ws_url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(raw_msg)
                            await self._handle_message(data)
                        except (json.JSONDecodeError, KeyError):
                            continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    if self.on_error:
                        self.on_error({"type": "connection_error", "error": str(e)})
                    await asyncio.sleep(5)

        self._ws = None

    async def _handle_message(self, data: dict):
        """Route incoming messages to appropriate handlers."""
        msg_type = data.get("type", "")

        if msg_type == "subscribed":
            # Track subscription ID for ticker channel
            msg = data.get("msg", {})
            if msg.get("channel") == "ticker":
                self._ticker_sid = msg.get("sid")

        elif msg_type == "ticker":
            if self.on_ticker:
                self.on_ticker(data.get("msg", {}))

        elif msg_type == "orderbook_snapshot":
            if self.on_orderbook:
                self.on_orderbook(data)

        elif msg_type == "orderbook_delta":
            if self.on_orderbook:
                self.on_orderbook(data)

        elif msg_type == "error":
            if self.on_error:
                self.on_error(data.get("msg", {}))

    async def subscribe_ticker(self, market_tickers: list[str]):
        """Subscribe to ticker updates for given market tickers."""
        if self._ws is None:
            return
        cmd = {
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {
                "channels": ["ticker"],
                "market_tickers": market_tickers,
            },
        }
        await self._ws.send(json.dumps(cmd))

    async def update_ticker_markets(self, market_tickers: list[str], action: str = "add_markets"):
        """Add or remove markets from the ticker subscription."""
        if self._ws is None or self._ticker_sid is None:
            return
        cmd = {
            "id": self._next_id(),
            "cmd": "update_subscription",
            "params": {
                "sids": [self._ticker_sid],
                "market_tickers": market_tickers,
                "action": action,
            },
        }
        await self._ws.send(json.dumps(cmd))

    async def unsubscribe_all(self):
        """Unsubscribe from all channels."""
        if self._ws is None or self._ticker_sid is None:
            return
        cmd = {
            "id": self._next_id(),
            "cmd": "unsubscribe",
            "params": {
                "sids": [self._ticker_sid],
            },
        }
        await self._ws.send(json.dumps(cmd))
        self._ticker_sid = None

    def stop(self):
        """Stop the WebSocket connection."""
        self._running = False
