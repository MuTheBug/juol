"""Binance USDT-M Futures API client with algoOrder support.

Uses the LIVE Binance endpoint (https://fapi.binance.com), NOT the testnet.
Conditional orders (STOP_MARKET, TAKE_PROFIT_MARKET, etc.) are routed through
the new ``/fapi/v1/algoOrder`` endpoint introduced 2025-12-09.

Reference:
  https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/New-Algo-Order
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any, Optional
from urllib.parse import urlencode

import requests

from tori_trades_bot.config import BASE_URL, BINANCE_API_KEY, BINANCE_API_SECRET

logger = logging.getLogger(__name__)

# Order types that MUST go through the Algo Order endpoint
ALGO_ORDER_TYPES = frozenset({
    "STOP",
    "STOP_MARKET",
    "TAKE_PROFIT",
    "TAKE_PROFIT_MARKET",
    "TRAILING_STOP_MARKET",
})


class BinanceAPIError(Exception):
    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(f"[{code}] {msg}")


class BinanceFuturesClient:
    """Minimal Binance USDT-M Futures REST client."""

    def __init__(
        self,
        api_key: str = BINANCE_API_KEY,
        api_secret: str = BINANCE_API_SECRET,
        base_url: str = BASE_URL,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self.api_key})

    # ── Signing ─────────────────────────────────────────────────────────

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(sorted(params.items()))
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        signed: bool = True,
    ) -> Any:
        params = dict(params or {})
        if signed:
            params = self._sign(params)
        url = f"{self.base_url}{path}"
        resp = self._session.request(method, url, params=params, timeout=10)
        data = resp.json()
        if resp.status_code >= 400 or (isinstance(data, dict) and "code" in data and data["code"] < 0):
            code = data.get("code", resp.status_code)
            msg = data.get("msg", resp.text)
            raise BinanceAPIError(code, msg)
        return data

    # ── Public endpoints ────────────────────────────────────────────────

    def server_time(self) -> int:
        return self._request("GET", "/fapi/v1/time", signed=False)["serverTime"]

    def exchange_info(self, symbol: str | None = None) -> dict:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v1/exchangeInfo", params=params, signed=False)

    def klines(
        self,
        symbol: str,
        interval: str = "4h",
        limit: int = 500,
    ) -> list[list]:
        return self._request("GET", "/fapi/v1/klines", params={
            "symbol": symbol, "interval": interval, "limit": limit,
        }, signed=False)

    def mark_price(self, symbol: str) -> dict:
        return self._request("GET", "/fapi/v1/premiumIndex", params={
            "symbol": symbol,
        }, signed=False)

    # ── Account / Position ──────────────────────────────────────────────

    def account(self) -> dict:
        return self._request("GET", "/fapi/v2/account")

    def balance(self) -> list[dict]:
        return self._request("GET", "/fapi/v2/balance")

    def usdt_balance(self) -> float:
        for b in self.balance():
            if b["asset"] == "USDT":
                return float(b["availableBalance"])
        return 0.0

    def position_risk(self, symbol: str | None = None) -> list[dict]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v2/positionRisk", params=params)

    def open_position(self, symbol: str) -> Optional[dict]:
        """Return the active position for *symbol*, or None if flat."""
        for p in self.position_risk(symbol):
            amt = float(p.get("positionAmt", 0))
            if amt != 0:
                return p
        return None

    # ── Leverage & Margin ───────────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        return self._request("POST", "/fapi/v1/leverage", params={
            "symbol": symbol, "leverage": leverage,
        })

    def set_margin_type(self, symbol: str, margin_type: str = "CROSSED") -> dict:
        try:
            return self._request("POST", "/fapi/v1/marginType", params={
                "symbol": symbol, "marginType": margin_type,
            })
        except BinanceAPIError as e:
            # -4046 = margin type already set
            if e.code == -4046:
                return {"msg": "No need to change margin type."}
            raise

    # ── Regular orders (MARKET / LIMIT) ─────────────────────────────────

    def new_order(self, **kwargs: Any) -> dict:
        """Place a MARKET or LIMIT order via /fapi/v1/order.

        Do NOT pass conditional order types here — use ``new_algo_order``.
        """
        order_type = kwargs.get("type", "")
        if order_type in ALGO_ORDER_TYPES:
            raise ValueError(
                f"Order type {order_type} must use new_algo_order(), "
                f"not new_order()."
            )
        return self._request("POST", "/fapi/v1/order", params=kwargs)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._request("DELETE", "/fapi/v1/order", params={
            "symbol": symbol, "orderId": order_id,
        })

    # ── Algo orders (STOP_MARKET, TAKE_PROFIT_MARKET, etc.) ────────────
    #    Endpoint: POST /fapi/v1/algoOrder
    #    Key change: `stopPrice` is now `triggerPrice`

    def new_algo_order(self, **kwargs: Any) -> dict:
        """Place a conditional / algo order via /fapi/v1/algoOrder.

        Parameters
        ----------
        symbol : str
        side : str               BUY | SELL
        type : str               STOP_MARKET | TAKE_PROFIT_MARKET | STOP |
                                 TAKE_PROFIT | TRAILING_STOP_MARKET
        triggerPrice : str       Price that triggers the order (replaces
                                 the old ``stopPrice`` parameter).
        quantity : str           (optional for closePosition=true)
        price : str              Limit price for STOP / TAKE_PROFIT types.
        closePosition : str      "true" to close full position on trigger.
        workingType : str        MARK_PRICE (default) | CONTRACT_PRICE
        priceProtect : str       "TRUE" | "FALSE"
        timeInForce : str        GTC (default) | IOC | FOK
        reduceOnly : str         "true" | "false"
        activationPrice : str    For TRAILING_STOP_MARKET.
        callbackRate : str       For TRAILING_STOP_MARKET (1–5).
        newClientOrderId : str   Custom order ID.

        Returns the algo-order response including ``algoId``.
        """
        return self._request("POST", "/fapi/v1/algoOrder", params=kwargs)

    def cancel_algo_order(self, symbol: str, algo_id: int) -> dict:
        return self._request("DELETE", "/fapi/v1/algoOrder", params={
            "symbol": symbol, "algoId": algo_id,
        })

    def open_algo_orders(self, symbol: str | None = None) -> list[dict]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v1/openAlgoOrders", params=params)

    def cancel_all_algo_orders(self, symbol: str) -> None:
        for order in self.open_algo_orders(symbol):
            try:
                self.cancel_algo_order(symbol, order["algoId"])
            except BinanceAPIError:
                logger.warning("Failed to cancel algo order %s", order["algoId"])

    # ── Convenience: place market + stop-loss combo ─────────────────────

    def market_entry_with_stop(
        self,
        symbol: str,
        side: str,
        quantity: str,
        stop_price: str,
        working_type: str = "CONTRACT_PRICE",
    ) -> tuple[dict, dict]:
        """Open a position at market and set a STOP_MARKET via algoOrder.

        Parameters
        ----------
        symbol : str        e.g. "BTCUSDT"
        side : str          "BUY" (long) or "SELL" (short)
        quantity : str      Base-asset quantity (stringified)
        stop_price : str    Trigger price for the stop-loss
        working_type : str  MARK_PRICE or CONTRACT_PRICE

        Returns (market_order_response, algo_order_response).
        """
        # 1. Market entry
        market = self.new_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity,
        )

        # 2. Stop-loss via algo order (opposite side)
        stop_side = "SELL" if side == "BUY" else "BUY"
        algo = self.new_algo_order(
            symbol=symbol,
            side=stop_side,
            type="STOP_MARKET",
            triggerPrice=stop_price,
            closePosition="true",
            workingType=working_type,
        )

        return market, algo

    def update_stop_loss(
        self,
        symbol: str,
        new_trigger_price: str,
        side: str,
        working_type: str = "CONTRACT_PRICE",
    ) -> dict:
        """Cancel existing algo stop-losses and place a new one."""
        self.cancel_all_algo_orders(symbol)
        return self.new_algo_order(
            symbol=symbol,
            side=side,
            type="STOP_MARKET",
            triggerPrice=new_trigger_price,
            closePosition="true",
            workingType=working_type,
        )

    # ── Symbol info helpers ─────────────────────────────────────────────

    def get_symbol_info(self, symbol: str) -> dict | None:
        info = self.exchange_info(symbol)
        for s in info.get("symbols", []):
            if s["symbol"] == symbol:
                return s
        return None

    def get_quantity_precision(self, symbol: str) -> int:
        info = self.get_symbol_info(symbol)
        if info:
            return info.get("quantityPrecision", 3)
        return 3

    def get_price_precision(self, symbol: str) -> int:
        info = self.get_symbol_info(symbol)
        if info:
            return info.get("pricePrecision", 2)
        return 2
