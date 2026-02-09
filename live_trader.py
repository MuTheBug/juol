import asyncio
import logging
import math
import numpy as np
from typing import Dict, Any, Optional
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
from telegram_notifier import TelegramNotifier
from config import BINANCE_API_KEY, BINANCE_API_SECRET, PRIMARY_ASSET

logger = logging.getLogger(__name__)

class LiveTrader:
    def __init__(self, api_key: str, api_secret: str, notifier: TelegramNotifier):
        self.client: Optional[AsyncClient] = None
        self.api_key = api_key
        self.api_secret = api_secret
        self.notifier = notifier
        self.is_running = False

    async def connect(self):
        logger.info("Connecting to Binance...")
        self.client = await AsyncClient.create(self.api_key, self.api_secret)
        self.is_running = True

    async def close(self):
        if self.client:
            await self.client.close_connection()
        self.is_running = False

    async def fetch_latest_klines(self, symbol: str, interval: str, limit: int = 500):
        """Fetches recent OHLCV data."""
        try:
            klines = await self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            return klines
        except BinanceAPIException as e:
            logger.error(f"Error fetching klines: {e}")
            self.notifier.notify_error(f"Klines Fetch Error: {e.message}")
            return None

    async def get_balance(self, asset: str = "USDT") -> float:
        """Gets futures account balance."""
        try:
            account = await self.client.futures_account()
            for balance in account['assets']:
                if balance['asset'] == asset:
                    return float(balance['walletBalance'])
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    async def place_order(self, symbol: str, side: str, order_type: str,
                          quantity: float, price: Optional[float] = None,
                          stop_price: Optional[float] = None):
        """Places a futures order with retries."""
        for attempt in range(3):
            try:
                params = {
                    'symbol': symbol,
                    'side': side,
                    'type': order_type,
                    'quantity': quantity
                }
                if price: params['price'] = price
                if stop_price: params['stopPrice'] = stop_price

                # In futures, some order types have specific parameters
                if order_type == 'STOP_MARKET':
                    params['stopPrice'] = stop_price
                    del params['type']
                    order = await self.client.futures_create_order(type='STOP_MARKET', **params)
                else:
                    order = await self.client.futures_create_order(**params)

                return order
            except BinanceAPIException as e:
                logger.error(f"Order placement failed (attempt {attempt+1}): {e}")
                if attempt == 2:
                    self.notifier.notify_error(f"Order Failed: {e.message}")
                await asyncio.sleep(2 ** attempt) # Exponential backoff
        return None

    async def get_symbol_info(self, symbol: str):
        """Fetches symbol rules (precision, filters)."""
        info = await self.client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                return s
        return None

    def format_quantity(self, quantity: float, step_size: float) -> float:
        """Formats quantity according to step size."""
        precision = int(round(-np.log10(step_size), 0))
        return floor(quantity, precision) # Need math.floor or custom

    async def execute_trade(self, symbol: str, side: int, notional: float,
                             leverage: int, sl_price: float, tp_price: float):
        """Orchestrates a full trade entry: Leverage -> Market -> SL -> TP."""
        try:
            # 0. Get Precision Info
            info = await self.get_symbol_info(symbol)
            if not info:
                logger.error(f"Could not find info for {symbol}")
                return None

            price_precision = info['pricePrecision']
            qty_step = 0.001 # Default
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    qty_step = float(f['stepSize'])

            # 1. Set Leverage
            await self.client.futures_change_leverage(symbol=symbol, leverage=leverage)

            # 2. Calculate and Format Quantity
            ticker = await self.client.futures_symbol_ticker(symbol=symbol)
            curr_price = float(ticker['price'])
            quantity = notional / curr_price

            # Format quantity and prices
            def format_qty(val, step):
                prec = int(round(-np.log10(step), 0))
                factor = 10 ** prec
                return math.floor(val * factor) / factor

            quantity = format_qty(quantity, qty_step)
            sl_price = round(sl_price, price_precision)
            tp_price = round(tp_price, price_precision)

            binance_side = "BUY" if side == 1 else "SELL"
            opposite_side = "SELL" if side == 1 else "BUY"

            # 3. Entry Market Order
            entry_order = await self.place_order(symbol, binance_side, "MARKET", quantity)
            if not entry_order: return None

            # 4. Stop Loss
            await self.place_order(symbol, opposite_side, "STOP_MARKET", quantity, stop_price=sl_price)

            # 5. Take Profit
            await self.place_order(symbol, opposite_side, "LIMIT", quantity, price=tp_price)

            return entry_order
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            self.notifier.notify_error(f"Trade Execution Failed: {str(e)}")
            return None
