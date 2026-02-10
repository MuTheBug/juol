"""Autonomous live trading bot for the Tori Trades Trendline Strategy.

Connects to Binance USDT-M Futures (LIVE, not testnet), monitors 4-hour
candles, detects trendline setups (bounce & break), and manages positions
with trailing stops using the new algoOrder endpoints.

Usage:
    python run_bot.py
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from tori_trades_bot.binance_api import BinanceAPIError, BinanceFuturesClient
from tori_trades_bot.config import (
    DEFAULT_LEVERAGE,
    KLINE_LIMIT,
    MARGIN_TYPE,
    POLL_INTERVAL_SECONDS,
    RISK_PER_TRADE_PCT,
    SYMBOLS,
    WICK_BUFFER_ATR,
)
from tori_trades_bot.trendline import (
    Signal,
    Trendline,
    detect_trendlines,
    generate_signals,
    update_trailing_stop,
)
from tori_trades_bot.utils import compute_atr

logger = logging.getLogger("tori_bot")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _klines_to_df(raw: list[list]) -> pd.DataFrame:
    """Convert raw Binance kline response to a DataFrame."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    # Drop the last (incomplete) candle
    if len(df) > 1:
        df = df.iloc[:-1]
    return df[["open", "high", "low", "close", "volume"]]


def _fmt_price(price: float, precision: int) -> str:
    return f"{price:.{precision}f}"


def _fmt_qty(qty: float, precision: int) -> str:
    return f"{qty:.{precision}f}"


# ── Per-symbol state ────────────────────────────────────────────────────────

class SymbolState:
    """Tracks live trendlines and position state for one symbol."""

    def __init__(self, symbol: str, client: BinanceFuturesClient):
        self.symbol = symbol
        self.client = client

        self.qty_precision = client.get_quantity_precision(symbol)
        self.price_precision = client.get_price_precision(symbol)

        self.df: Optional[pd.DataFrame] = None
        self.atr: Optional[pd.Series] = None
        self.trendlines: list[Trendline] = []
        self.last_candle_time: Optional[pd.Timestamp] = None

        # Position tracking
        self.in_position: bool = False
        self.position_side: Optional[str] = None  # "long" or "short"
        self.entry_signal: Optional[Signal] = None
        self.current_stop: Optional[float] = None

    def refresh_candles(self) -> bool:
        """Fetch latest 4h candles. Return True if a new candle appeared."""
        raw = self.client.klines(self.symbol, interval="4h", limit=KLINE_LIMIT)
        self.df = _klines_to_df(raw)
        self.atr = compute_atr(self.df)

        latest_time = self.df.index[-1]
        if self.last_candle_time is not None and latest_time <= self.last_candle_time:
            return False  # no new candle

        self.last_candle_time = latest_time
        return True

    def sync_position(self) -> None:
        """Sync in-memory position state with Binance."""
        pos = self.client.open_position(self.symbol)
        if pos is None:
            if self.in_position:
                logger.info("[%s] Position closed externally", self.symbol)
            self.in_position = False
            self.position_side = None
            self.entry_signal = None
            self.current_stop = None
        else:
            amt = float(pos["positionAmt"])
            self.in_position = True
            self.position_side = "long" if amt > 0 else "short"


# ── Main bot ────────────────────────────────────────────────────────────────

class ToriTradesBot:
    """Autonomous live trading bot."""

    def __init__(self, symbols: list[str] | None = None):
        self.client = BinanceFuturesClient()
        self.symbols = symbols or SYMBOLS
        self.states: dict[str, SymbolState] = {}
        self._running = False

    # ── Setup ───────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Initialise exchange settings and symbol states."""
        logger.info("=" * 60)
        logger.info("  Tori Trades Bot — LIVE Binance USDT-M Futures")
        logger.info("=" * 60)

        # Verify connectivity
        st = self.client.server_time()
        logger.info("Binance server time: %s", datetime.fromtimestamp(st / 1000, tz=timezone.utc))

        # Account balance
        bal = self.client.usdt_balance()
        logger.info("Available USDT balance: %.2f", bal)
        if bal <= 0:
            logger.error("No USDT balance available. Fund the account first.")
            sys.exit(1)

        # Configure each symbol
        for sym in self.symbols:
            logger.info("Configuring %s (leverage=%dx, margin=%s)",
                        sym, DEFAULT_LEVERAGE, MARGIN_TYPE)
            try:
                self.client.set_leverage(sym, DEFAULT_LEVERAGE)
                self.client.set_margin_type(sym, MARGIN_TYPE)
            except BinanceAPIError as e:
                logger.warning("Config warning for %s: %s", sym, e)

            state = SymbolState(sym, self.client)
            state.refresh_candles()
            state.sync_position()
            self.states[sym] = state

            # Initial trendline detection
            if state.df is not None and len(state.df) > 80:
                state.trendlines = detect_trendlines(
                    state.df, up_to_iloc=len(state.df) - 1, atr=state.atr,
                )
                logger.info("  %s: %d candles loaded, %d trendlines detected, position=%s",
                            sym, len(state.df), len(state.trendlines),
                            state.position_side or "flat")

        logger.info("Setup complete. Starting main loop (poll every %ds).",
                     POLL_INTERVAL_SECONDS)

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Blocking main loop."""
        self._running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self.setup()

        while self._running:
            try:
                self._tick()
            except BinanceAPIError as e:
                logger.error("Binance API error: %s", e)
            except Exception:
                logger.exception("Unexpected error in tick")
            time.sleep(POLL_INTERVAL_SECONDS)

        logger.info("Bot stopped.")

    def _handle_shutdown(self, signum: int, frame) -> None:
        logger.info("Shutdown signal received (%s). Stopping ...", signum)
        self._running = False

    # ── Tick (runs every POLL_INTERVAL_SECONDS) ─────────────────────────

    def _tick(self) -> None:
        for sym, state in self.states.items():
            # Sync position with exchange
            state.sync_position()

            # Refresh candles — only act on new 4h candle close
            new_candle = state.refresh_candles()
            if not new_candle:
                # Even without new candle, update trailing stop if in position
                if state.in_position and state.entry_signal and state.df is not None:
                    self._maybe_update_trailing_stop(state)
                continue

            logger.info("[%s] New 4h candle: %s  close=%.4f",
                        sym, state.last_candle_time, state.df["close"].iloc[-1])

            # Recalculate trendlines
            i = len(state.df) - 1
            state.trendlines = detect_trendlines(state.df, up_to_iloc=i, atr=state.atr)

            if state.in_position:
                self._manage_position(state, i)
            else:
                self._look_for_entry(state, i)

    # ── Entry logic ─────────────────────────────────────────────────────

    def _look_for_entry(self, state: SymbolState, bar_iloc: int) -> None:
        signals = generate_signals(state.df, bar_iloc, state.trendlines, state.atr)
        if not signals:
            return

        # Pick best signal (most touchpoints, tightest risk)
        best = sorted(signals, key=lambda s: (-s.touchpoints, s.risk_atr))[0]
        logger.info("[%s] SIGNAL: %s  touchpoints=%d  entry=%.4f  stop=%.4f  risk_atr=%.2f",
                     state.symbol, best.signal_type, best.touchpoints,
                     best.entry_price, best.stop_price, best.risk_atr)

        self._execute_entry(state, best)

    def _execute_entry(self, state: SymbolState, sig: Signal) -> None:
        """Place a market entry + algo stop-loss."""
        try:
            balance = self.client.usdt_balance()
        except BinanceAPIError:
            logger.error("[%s] Failed to fetch balance", state.symbol)
            return

        # Position sizing: risk RISK_PER_TRADE_PCT of balance
        risk_per_unit = abs(sig.entry_price - sig.stop_price)
        if risk_per_unit == 0:
            return
        risk_amount = balance * (RISK_PER_TRADE_PCT / 100.0)
        quantity = risk_amount / risk_per_unit

        # Cap to leveraged equity
        max_notional = balance * DEFAULT_LEVERAGE
        if quantity * sig.entry_price > max_notional:
            quantity = max_notional / sig.entry_price

        qty_str = _fmt_qty(quantity, state.qty_precision)
        stop_str = _fmt_price(sig.stop_price, state.price_precision)

        is_long = "long" in sig.signal_type
        entry_side = "BUY" if is_long else "SELL"

        logger.info("[%s] ENTERING %s  qty=%s  stop=%s",
                     state.symbol, entry_side, qty_str, stop_str)

        try:
            market_resp, algo_resp = self.client.market_entry_with_stop(
                symbol=state.symbol,
                side=entry_side,
                quantity=qty_str,
                stop_price=stop_str,
            )
            logger.info("[%s] Market order filled: %s", state.symbol, market_resp.get("orderId"))
            logger.info("[%s] Algo stop placed: algoId=%s", state.symbol, algo_resp.get("algoId"))

            state.in_position = True
            state.position_side = "long" if is_long else "short"
            state.entry_signal = sig
            state.current_stop = sig.stop_price

        except BinanceAPIError as e:
            logger.error("[%s] Entry failed: %s", state.symbol, e)

    # ── Position management ─────────────────────────────────────────────

    def _manage_position(self, state: SymbolState, bar_iloc: int) -> None:
        """Check for exit conditions and trail the stop."""
        if state.entry_signal is None or state.df is None:
            return

        sig = state.entry_signal
        atr_val = state.atr.iloc[bar_iloc]
        close = state.df["close"].iloc[bar_iloc]

        # Check if safety/action line is broken (for bounces)
        if sig.safety_line is None:
            tl = sig.action_line
            tl_val = tl.value_at(bar_iloc)
            should_exit = (
                (state.position_side == "long" and close < tl_val) or
                (state.position_side == "short" and close > tl_val)
            )
            if should_exit:
                logger.info("[%s] Trendline broken — closing position at market", state.symbol)
                self._close_position_market(state)
                return

        # Update trailing stop
        self._maybe_update_trailing_stop(state)

    def _maybe_update_trailing_stop(self, state: SymbolState) -> None:
        """Trail the algo stop-loss along the trendline."""
        if state.entry_signal is None or state.df is None or state.current_stop is None:
            return

        bar_iloc = len(state.df) - 1
        new_stop = update_trailing_stop(state.entry_signal, bar_iloc, state.df, state.atr)

        # Only update if stop has moved meaningfully (> 0.1 % of current stop)
        if state.current_stop is not None:
            if state.position_side == "long" and new_stop <= state.current_stop:
                return
            if state.position_side == "short" and new_stop >= state.current_stop:
                return
            if abs(new_stop - state.current_stop) / state.current_stop < 0.001:
                return

        stop_str = _fmt_price(new_stop, state.price_precision)
        stop_side = "SELL" if state.position_side == "long" else "BUY"

        logger.info("[%s] Trailing stop: %.4f -> %s", state.symbol, state.current_stop, stop_str)

        try:
            self.client.update_stop_loss(
                symbol=state.symbol,
                new_trigger_price=stop_str,
                side=stop_side,
            )
            state.current_stop = new_stop
        except BinanceAPIError as e:
            logger.error("[%s] Failed to update stop: %s", state.symbol, e)

    def _close_position_market(self, state: SymbolState) -> None:
        """Close position at market and cancel algo orders."""
        try:
            self.client.cancel_all_algo_orders(state.symbol)
        except BinanceAPIError:
            pass

        pos = self.client.open_position(state.symbol)
        if pos is None:
            state.in_position = False
            return

        amt = abs(float(pos["positionAmt"]))
        if amt == 0:
            state.in_position = False
            return

        close_side = "SELL" if state.position_side == "long" else "BUY"
        qty_str = _fmt_qty(amt, state.qty_precision)

        try:
            resp = self.client.new_order(
                symbol=state.symbol,
                side=close_side,
                type="MARKET",
                quantity=qty_str,
                reduceOnly="true",
            )
            logger.info("[%s] Position closed: orderId=%s", state.symbol, resp.get("orderId"))
        except BinanceAPIError as e:
            logger.error("[%s] Close failed: %s", state.symbol, e)

        state.in_position = False
        state.position_side = None
        state.entry_signal = None
        state.current_stop = None
