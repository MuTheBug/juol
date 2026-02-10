"""Utility helpers: data loading, resampling, indicator calculations."""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from tori_trades_bot.config import ATR_PERIOD, SWING_ORDER


# ── Data loading ────────────────────────────────────────────────────────────

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
]

NUMERIC_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]


def load_klines(path: str | pathlib.Path) -> pd.DataFrame:
    """Load a 1-hour kline CSV and return a clean DataFrame indexed by time."""
    df = pd.read_csv(path)
    # Ensure column names match what we expect
    if list(df.columns) != KLINE_COLS:
        df.columns = KLINE_COLS
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    return df


def load_funding(path: str | pathlib.Path) -> pd.DataFrame:
    """Load an 8-hour funding-rate CSV."""
    df = pd.read_csv(path)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.set_index("fundingTime").sort_index()
    return df


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-hour OHLCV to 4-hour OHLCV."""
    ohlcv = df_1h[["open", "high", "low", "close", "volume"]].copy()
    df_4h = ohlcv.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return df_4h


# ── Technical indicators ────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Compute Average True Range over *period* bars."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Swing-point detection ───────────────────────────────────────────────────

def find_swing_lows(
    df: pd.DataFrame,
    order: int = SWING_ORDER,
    col: str = "low",
) -> pd.DataFrame:
    """Return rows of *df* that are local minima (swing lows).

    A swing low at bar *i* means ``df[col][i]`` is the minimum among
    bars ``[i-order .. i+order]``.
    """
    values = df[col].values
    n = len(values)
    mask = np.zeros(n, dtype=bool)
    for i in range(order, n - order):
        window = values[i - order: i + order + 1]
        if values[i] == window.min() and np.sum(window == values[i]) == 1:
            mask[i] = True
    return df[mask].copy()


def find_swing_highs(
    df: pd.DataFrame,
    order: int = SWING_ORDER,
    col: str = "high",
) -> pd.DataFrame:
    """Return rows of *df* that are local maxima (swing highs)."""
    values = df[col].values
    n = len(values)
    mask = np.zeros(n, dtype=bool)
    for i in range(order, n - order):
        window = values[i - order: i + order + 1]
        if values[i] == window.max() and np.sum(window == values[i]) == 1:
            mask[i] = True
    return df[mask].copy()
