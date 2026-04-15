"""Kalshi REST API client — orderbook polling and trade history.

Authenticates with API key + private key (RSA-PSS). Falls back to
unauthenticated for public market data when credentials are absent.
All methods return normalized dicts. No method raises — errors are
logged and return None or empty lists so the polling loop continues.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from . import config as cfg

log = logging.getLogger("al3x.kalshi_feed")

_TIMEOUT = 10.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KalshiFeed:
    """Thin async wrapper around Kalshi's Trade API v2.

    Authentication: Kalshi uses RSA-PSS signatures. The API key ID and
    PEM private key are loaded from environment variables:
        KALSHI_API_KEY_ID      — your API key identifier
        KALSHI_PRIVATE_KEY_PEM — the PEM-encoded RSA private key

    If credentials are absent, only public endpoints (market data) are
    accessible. Authenticated endpoints (positions, order placement)
    will return None with a warning.
    """

    def __init__(self) -> None:
        self._api_key_id = os.environ.get("KALSHI_API_KEY_ID", "")
        self._private_key_pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM", "")
        self._base = cfg.KALSHI_BASE_URL
        self._client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers={"Content-Type": "application/json"},
            follow_redirects=True,
        )
        self._authed = bool(self._api_key_id and self._private_key_pem)
        if not self._authed:
            log.info("Kalshi: no credentials — public data only (paper mode)")

    async def close(self) -> None:
        await self._client.aclose()

    def _sign_request(self, method: str, path: str,
                       body: str = "") -> Dict[str, str]:
        """Return Authorization headers for RSA-PSS signing."""
        if not self._authed:
            return {}
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            ts_ms = str(int(time.time() * 1000))
            msg = ts_ms + method.upper() + path + body
            private_key = serialization.load_pem_private_key(
                self._private_key_pem.encode(),
                password=None,
            )
            sig = private_key.sign(
                msg.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
            sig_b64 = base64.b64encode(sig).decode()
            return {
                "KALSHI-ACCESS-KEY": self._api_key_id,
                "KALSHI-ACCESS-SIGNATURE": sig_b64,
                "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            }
        except Exception as e:  # noqa: BLE001
            log.warning("Kalshi signing failed: %s", e)
            return {}

    async def _get(self, path: str,
                    params: Optional[Dict] = None) -> Optional[Any]:
        url = self._base + path
        headers = self._sign_request("GET", path)
        try:
            r = await self._client.get(url, params=params, headers=headers)
            if r.status_code == 200:
                return r.json()
            log.info("Kalshi GET %s → HTTP %s", path, r.status_code)
            return None
        except Exception as e:  # noqa: BLE001
            log.info("Kalshi GET %s error: %s", path, e)
            return None

    async def _post(self, path: str,
                     body: Dict) -> Optional[Any]:
        if not self._authed:
            log.warning("Kalshi POST %s skipped — no credentials", path)
            return None
        url = self._base + path
        body_str = json.dumps(body)
        headers = self._sign_request("POST", path, body_str)
        try:
            r = await self._client.post(url, content=body_str,
                                         headers=headers)
            if r.status_code in (200, 201):
                return r.json()
            log.warning("Kalshi POST %s → HTTP %s: %s",
                        path, r.status_code, r.text[:200])
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("Kalshi POST %s error: %s", path, e)
            return None

    # ---- Market discovery -----------------------------------------------

    async def get_nyc_temp_markets(self,
                                    target_date_str: str
                                    ) -> List[Dict[str, Any]]:
        """Return all active NYC high-temp Kalshi markets for target_date.

        target_date_str: 'YYYY-MM-DD' — we strip dashes to form the
        Kalshi event ticker suffix (e.g., '20260414').
        """
        date_suffix = target_date_str.replace("-", "")
        event_ticker = f"{cfg.KALSHI_NYC_EVENT_PREFIX}{date_suffix}"
        data = await self._get(
            "/markets",
            params={"event_ticker": event_ticker, "limit": 100},
        )
        if not data:
            return []
        markets = data.get("markets") or []
        result = []
        for m in markets:
            threshold = _parse_threshold(m.get("title") or "")
            result.append({
                "market_ticker": m.get("ticker", ""),
                "event_ticker": event_ticker,
                "title": m.get("title", ""),
                "threshold_f": threshold,
                "status": m.get("status", ""),
                "close_time": m.get("close_time"),
                "yes_bid": _safe_price(m.get("yes_bid")),
                "yes_ask": _safe_price(m.get("yes_ask")),
                "last_price": _safe_price(m.get("last_price")),
                "volume": m.get("volume", 0),
                "open_interest": m.get("open_interest", 0),
            })
        return [m for m in result if m["status"] == "open"]

    # ---- Orderbook ------------------------------------------------------

    async def get_orderbook(self,
                              market_ticker: str,
                              depth: int = 5
                              ) -> Optional[Dict[str, Any]]:
        """Return normalized orderbook snapshot for one market."""
        data = await self._get(
            f"/markets/{market_ticker}/orderbook",
            params={"depth": depth},
        )
        if not data:
            return None
        ob = data.get("orderbook") or data
        yes_bids = ob.get("yes") or []   # [[price, qty], ...]
        no_bids = ob.get("no") or []
        # Kalshi: YES bids ↔ NO asks, YES asks ↔ NO bids
        bid_levels = [{"price": _safe_price(row[0]),
                        "qty": int(row[1])}
                      for row in yes_bids if len(row) >= 2]
        ask_levels = [{"price": _safe_price(row[0]),
                        "qty": int(row[1])}
                      for row in no_bids if len(row) >= 2]
        # Sort: bids descending, asks ascending
        bid_levels.sort(key=lambda x: x["price"], reverse=True)
        ask_levels.sort(key=lambda x: x["price"])
        best_bid = bid_levels[0]["price"] if bid_levels else None
        # YES ask = 1 - best NO bid
        best_ask = (round(1.0 - ask_levels[0]["price"], 4)
                    if ask_levels else None)
        return {
            "market_ticker": market_ticker,
            "captured_at": _utc_now(),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_depth": bid_levels[:depth],
            "ask_depth": ask_levels[:depth],
            "spread_cents": (round((best_ask - best_bid) * 100, 1)
                             if best_bid and best_ask else None),
        }

    # ---- Trade history --------------------------------------------------

    async def get_recent_trades(self,
                                 market_ticker: str,
                                 limit: int = 50
                                 ) -> List[Dict[str, Any]]:
        data = await self._get(
            f"/markets/{market_ticker}/trades",
            params={"limit": limit},
        )
        if not data:
            return []
        trades = data.get("trades") or []
        result = []
        for t in trades:
            result.append({
                "trade_id": t.get("trade_id") or t.get("id", ""),
                "market_ticker": market_ticker,
                "executed_at": t.get("created_time") or _utc_now(),
                "price": _safe_price(t.get("yes_price")),
                "count": int(t.get("count") or 0),
                "side": "yes",
                "taker_side": t.get("taker_side"),
            })
        return result

    # ---- Portfolio ------------------------------------------------------

    async def get_positions(self) -> List[Dict[str, Any]]:
        if not self._authed:
            return []
        data = await self._get("/portfolio/positions")
        if not data:
            return []
        return data.get("market_positions") or []

    async def get_balance(self) -> Optional[float]:
        if not self._authed:
            return None
        data = await self._get("/portfolio/balance")
        if not data:
            return None
        cents = data.get("balance")
        return float(cents) / 100.0 if cents is not None else None

    # ---- Order placement ------------------------------------------------

    async def place_order(self, market_ticker: str, side: str,
                           action: str, contracts: int,
                           limit_price: float,
                           client_order_id: str = ""
                           ) -> Optional[Dict[str, Any]]:
        """Place a limit order. Returns order dict or None on failure.

        side:   'yes' or 'no'
        action: 'buy' or 'sell'
        limit_price: 0.0 – 1.0 (will be multiplied by 100 for Kalshi)
        """
        if not self._authed:
            log.info("Order skipped (no credentials): %s %s %s x%d @ %.2f",
                     action, side, market_ticker, contracts, limit_price)
            return None
        body = {
            "ticker": market_ticker,
            "action": action,
            "side": side,
            "type": "limit",
            "count": contracts,
            "yes_price": int(round(limit_price * 100)),
            "client_order_id": client_order_id,
        }
        return await self._post("/portfolio/orders", body)


# ---- Helpers ------------------------------------------------------------

def _safe_price(v: Any) -> Optional[float]:
    """Convert Kalshi integer cents (0–100) or decimal to 0–1 float."""
    if v is None:
        return None
    try:
        vv = float(v)
        # Kalshi API sometimes returns integer cents (0-100)
        if vv > 1.0:
            return round(vv / 100.0, 4)
        return round(vv, 4)
    except (TypeError, ValueError):
        return None


def _parse_threshold(title: str) -> Optional[float]:
    """Extract temperature threshold from a Kalshi market title.

    Examples handled:
      'Will the NYC high temp exceed 72°F on Apr 14?' → 72.0
      'NYC High Temp > 68F'                           → 68.0
    """
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*[°º]?\s*F", title, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None
