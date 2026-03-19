"""Portable Kalshi REST client (v2) with RSA-PSS request signing.

Reads secrets from an env file. Supports both inline PEM and file-path PEM.
Includes token-bucket rate limiting (20 reads/sec, 10 writes/sec).

Docs: https://docs.kalshi.com/getting_started/quick_start_authenticated_requests
"""
from __future__ import annotations

import base64
import datetime as _dt
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

DEFAULT_ENV_PATH = Path(r"C:\Users\graso\clawd-buzz\secrets\kalshi.env")
DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file (ignores comments and blank lines)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing env file: {path}")
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@dataclass
class KalshiConfig:
    """Connection configuration for the Kalshi API."""
    base_url: str
    api_key_id: str
    private_key_pem: bytes | None = None


def load_config(env_path: Path | str = DEFAULT_ENV_PATH) -> KalshiConfig:
    """Load a KalshiConfig from an env file.

    Env keys:
      KALSHI_API_BASE_URL  (optional, defaults to production)
      KALSHI_API_KEY       (required)
      KALSHI_PRIVATE_KEY_PATH  (path to .key/.pem file — preferred)
      KALSHI_PRIVATE_KEY_PEM   (inline PEM — fallback)
    """
    env_path = Path(env_path)
    env = _load_env_file(env_path)

    base_url = env.get("KALSHI_API_BASE_URL", "").strip() or DEFAULT_BASE_URL
    api_key_id = env.get("KALSHI_API_KEY", "").strip()

    pem_path_str = env.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
    pem_inline = env.get("KALSHI_PRIVATE_KEY_PEM", "").strip()

    if not api_key_id:
        raise ValueError("KALSHI_API_KEY is missing/empty in env file")

    private_key_pem: bytes | None = None
    if pem_path_str:
        p = Path(pem_path_str)
        if not p.is_absolute():
            p = (env_path.parent / pem_path_str).resolve()
        private_key_pem = p.read_bytes()
    elif pem_inline:
        private_key_pem = pem_inline.replace("\\n", "\n").encode("utf-8")

    return KalshiConfig(
        base_url=base_url.rstrip("/"),
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
    )


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------
def _load_private_key(pem_bytes: bytes):
    """Load an RSA private key from PEM bytes."""
    return serialization.load_pem_private_key(
        pem_bytes, password=None, backend=default_backend(),
    )


def sign_request(private_key, timestamp_ms: str, method: str, path: str) -> str:
    """Create base64 RSA-PSS(SHA256) signature.

    Kalshi signs: ``{timestamp_ms}{METHOD}{path_without_query}``
    where *path* includes the ``/trade-api/v2`` prefix.
    """
    path_wo_query = path.split("?", 1)[0]
    msg = f"{timestamp_ms}{method.upper()}{path_wo_query}".encode("utf-8")
    sig = private_key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


# ---------------------------------------------------------------------------
# Rate limiting (token bucket)
# ---------------------------------------------------------------------------
class _TokenBucket:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate: float, burst: int):
        """
        Args:
            rate: tokens refilled per second
            burst: maximum tokens (bucket size)
        """
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            # Not enough tokens — wait a short interval and retry
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class KalshiClient:
    """Minimal, portable Kalshi REST client with built-in rate limiting."""

    # Kalshi documented limits
    _READ_RATE = 20    # reads per second
    _WRITE_RATE = 10   # writes per second

    def __init__(self, cfg: Optional[KalshiConfig] = None):
        self.cfg = cfg or load_config()
        self._private_key = None
        self.session = requests.Session()
        self._read_bucket = _TokenBucket(self._READ_RATE, self._READ_RATE)
        self._write_bucket = _TokenBucket(self._WRITE_RATE, self._WRITE_RATE)

    # -- internal helpers --------------------------------------------------

    def _ensure_private_key(self):
        if self._private_key is not None:
            return
        if not self.cfg.private_key_pem:
            raise RuntimeError(
                "Missing private key. Set KALSHI_PRIVATE_KEY_PATH in your env file."
            )
        try:
            self._private_key = _load_private_key(self.cfg.private_key_pem)
        except Exception as e:
            raise RuntimeError(
                "Failed to load private key PEM. If you pasted it inline, "
                "it likely lost newlines. Use KALSHI_PRIVATE_KEY_PATH instead."
            ) from e

    def _ts_ms(self) -> str:
        return str(int(_dt.datetime.now(tz=_dt.timezone.utc).timestamp() * 1000))

    def _auth_headers(self, method: str, path: str, ts_ms: str) -> dict[str, str]:
        """Build Kalshi auth headers (key, timestamp, signature)."""
        self._ensure_private_key()
        # Derive the base path from base_url (e.g. /trade-api/v2)
        base_path = "/" + self.cfg.base_url.split("//", 1)[-1].split("/", 1)[-1]
        if base_path in ("/", "//"):
            base_path = ""
        full_path = (base_path.rstrip("/") + path) if base_path else path

        return {
            "KALSHI-ACCESS-KEY": self.cfg.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": sign_request(
                self._private_key, ts_ms, method, full_path,
            ),
        }

    # -- generic request ---------------------------------------------------

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        auth: bool = False,
        timeout_s: int = 20,
    ) -> tuple[int, Any]:
        """Execute an API request. Returns ``(status_code, parsed_json_or_text)``.

        Automatically rate-limits: reads (GET) at 20/sec, writes (POST/PUT/DELETE)
        at 10/sec to stay within Kalshi's documented API limits.
        """
        if not path.startswith("/"):
            raise ValueError("path must start with /")

        # Rate-limit before sending
        if method.upper() == "GET":
            self._read_bucket.acquire()
        else:
            self._write_bucket.acquire()

        url = self.cfg.base_url + path
        headers: dict[str, str] = {"Accept": "application/json"}

        if auth:
            ts = self._ts_ms()
            headers.update(self._auth_headers(method, path, ts))

        resp = self.session.request(
            method.upper(), url,
            headers=headers, params=params, json=json_body,
            timeout=timeout_s,
        )
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        return resp.status_code, data

    # -- convenience endpoints ---------------------------------------------

    def get_balance(self) -> tuple[int, Any]:
        """GET /portfolio/balance (authenticated)."""
        return self.request("GET", "/portfolio/balance", auth=True)

    def get_markets(
        self,
        *,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> tuple[int, Any]:
        """GET /markets with optional filters."""
        params: dict[str, Any] = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        return self.request("GET", "/markets", params=params, auth=False)

    def create_order(self, payload: dict[str, Any]) -> tuple[int, Any]:
        """POST /portfolio/orders (authenticated)."""
        return self.request("POST", "/portfolio/orders", json_body=payload, auth=True)

    def get_order(self, order_id: str) -> tuple[int, Any]:
        """GET /portfolio/orders/{order_id} (authenticated)."""
        return self.request("GET", f"/portfolio/orders/{order_id}", auth=True)

    def cancel_order(self, order_id: str) -> tuple[int, Any]:
        """DELETE /portfolio/orders/{order_id} (authenticated)."""
        return self.request("DELETE", f"/portfolio/orders/{order_id}", auth=True)

    def get_settlements(self, limit: int = 100) -> tuple[int, Any]:
        """GET /portfolio/settlements (authenticated)."""
        return self.request(
            "GET", "/portfolio/settlements",
            params={"limit": limit}, auth=True,
        )
