"""Algorithmic trendline detection implementing the Tori Trades Playbook.

Key concepts
------------
* **Action Line** – the trendline used for entry.
* **Safety Line** – the trendline used as a stop / trailing stop.
* **Bounce setup** – enter when price *touches* an existing trendline.
* **Break setup**  – enter when price *closes through* a trendline.

A trendline is defined by two anchor swing points.  Additional swing points
that fall within tolerance are counted as extra touchpoints.  The playbook
requires >= 2 touchpoints and >= 1 week of data between the first anchor and
the current bar.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from tori_trades_bot.config import (
    BREAK_THRESHOLD_ATR,
    MAX_RISK_ATR,
    MAX_SLOPE_PCT_PER_BAR,
    MAX_VIOLATIONS,
    MIN_TRENDLINE_BARS,
    SWING_ORDER,
    TOUCH_TOLERANCE_ATR,
    WICK_BUFFER_ATR,
)
from tori_trades_bot.utils import find_swing_highs, find_swing_lows


# ── Trendline data structure ────────────────────────────────────────────────

@dataclasses.dataclass
class Trendline:
    """A trendline connecting swing points."""

    direction: str  # "up" = ascending support, "down" = descending resistance

    anchor1_iloc: int
    anchor1_price: float
    anchor2_iloc: int
    anchor2_price: float

    slope: float
    intercept: float

    touchpoints: list[int] = dataclasses.field(default_factory=list)

    broken: bool = False
    broken_at_iloc: Optional[int] = None

    @property
    def num_touches(self) -> int:
        return len(self.touchpoints)

    def value_at(self, iloc: int) -> float:
        return self.slope * iloc + self.intercept

    @classmethod
    def from_points(
        cls, direction: str,
        iloc1: int, price1: float,
        iloc2: int, price2: float,
    ) -> "Trendline":
        slope = (price2 - price1) / (iloc2 - iloc1) if iloc2 != iloc1 else 0.0
        intercept = price1 - slope * iloc1
        return cls(
            direction=direction,
            anchor1_iloc=iloc1, anchor1_price=price1,
            anchor2_iloc=iloc2, anchor2_price=price2,
            slope=slope, intercept=intercept,
            touchpoints=[iloc1, iloc2],
        )


# ── Optimised trendline detection ───────────────────────────────────────────

# Maximum number of recent swing points to consider for trendline building
_MAX_SWING_POINTS = 15


def _slope_ok(slope: float, price: float) -> bool:
    if price == 0:
        return False
    return abs(slope) / price <= MAX_SLOPE_PCT_PER_BAR


def _validate_trendline(
    df_low: np.ndarray,
    df_high: np.ndarray,
    df_close: np.ndarray,
    atr_arr: np.ndarray,
    tl: Trendline,
) -> Tuple[List[int], int]:
    """Count extra touches and close-violations between anchors.

    Uses numpy arrays for speed instead of DataFrame iloc access.
    """
    extra: list[int] = []
    violations = 0

    for i in range(tl.anchor1_iloc + 1, tl.anchor2_iloc):
        tl_val = tl.slope * i + tl.intercept
        a = atr_arr[i]
        if a == 0 or np.isnan(a):
            continue

        tol = a * TOUCH_TOLERANCE_ATR
        brk = a * BREAK_THRESHOLD_ATR

        if tl.direction == "up":
            if abs(df_low[i] - tl_val) <= tol:
                extra.append(i)
            elif df_close[i] < tl_val - brk:
                violations += 1
        else:
            if abs(df_high[i] - tl_val) <= tol:
                extra.append(i)
            elif df_close[i] > tl_val + brk:
                violations += 1

    return extra, violations


def detect_trendlines(
    df: pd.DataFrame,
    up_to_iloc: int,
    atr: pd.Series,
    swing_order: int = SWING_ORDER,
    min_span: int = MIN_TRENDLINE_BARS,
    max_violations: int = MAX_VIOLATIONS,
) -> List[Trendline]:
    """Detect valid trendlines using only recent swing points.

    Only examines the most recent ``_MAX_SWING_POINTS`` swing highs/lows
    to keep complexity bounded.
    """
    if up_to_iloc < min_span:
        return []

    # Look-back window for swing detection — limit to recent history
    lookback_start = max(0, up_to_iloc - 300)
    sub = df.iloc[lookback_start: up_to_iloc + 1]
    if len(sub) < min_span:
        return []

    swing_lows = find_swing_lows(sub, order=swing_order)
    swing_highs = find_swing_highs(sub, order=swing_order)

    # Convert to integer iloc positions in the *full* df
    def _to_ilocs(sw_df: pd.DataFrame) -> list[int]:
        result = []
        for idx in sw_df.index:
            pos = df.index.get_loc(idx)
            if isinstance(pos, slice):
                pos = pos.start
            result.append(int(pos))
        return result

    sl_ilocs = _to_ilocs(swing_lows)
    sl_prices = swing_lows["low"].values.tolist()
    sh_ilocs = _to_ilocs(swing_highs)
    sh_prices = swing_highs["high"].values.tolist()

    # Only keep the most recent swing points
    if len(sl_ilocs) > _MAX_SWING_POINTS:
        sl_ilocs = sl_ilocs[-_MAX_SWING_POINTS:]
        sl_prices = sl_prices[-_MAX_SWING_POINTS:]
    if len(sh_ilocs) > _MAX_SWING_POINTS:
        sh_ilocs = sh_ilocs[-_MAX_SWING_POINTS:]
        sh_prices = sh_prices[-_MAX_SWING_POINTS:]

    # Pre-extract numpy arrays for fast validation
    low_arr = df["low"].values
    high_arr = df["high"].values
    close_arr = df["close"].values
    atr_arr = atr.values

    trendlines: list[Trendline] = []

    # ── Upward support trendlines (ascending swing lows) ────────────────
    n_sl = len(sl_ilocs)
    for i in range(n_sl):
        for j in range(i + 1, n_sl):
            ia, pa = sl_ilocs[i], sl_prices[i]
            ib, pb = sl_ilocs[j], sl_prices[j]
            if pb <= pa or ib - ia < min_span:
                continue
            tl = Trendline.from_points("up", ia, pa, ib, pb)
            if not _slope_ok(tl.slope, pa):
                continue
            extra, viols = _validate_trendline(low_arr, high_arr, close_arr, atr_arr, tl)
            if viols > max_violations:
                continue
            tl.touchpoints = sorted(set(tl.touchpoints + extra))
            trendlines.append(tl)

    # ── Downward resistance trendlines (descending swing highs) ─────────
    n_sh = len(sh_ilocs)
    for i in range(n_sh):
        for j in range(i + 1, n_sh):
            ia, pa = sh_ilocs[i], sh_prices[i]
            ib, pb = sh_ilocs[j], sh_prices[j]
            if pb >= pa or ib - ia < min_span:
                continue
            tl = Trendline.from_points("down", ia, pa, ib, pb)
            if not _slope_ok(tl.slope, pa):
                continue
            extra, viols = _validate_trendline(low_arr, high_arr, close_arr, atr_arr, tl)
            if viols > max_violations:
                continue
            tl.touchpoints = sorted(set(tl.touchpoints + extra))
            trendlines.append(tl)

    return trendlines


# ── Signal detection ────────────────────────────────────────────────────────

@dataclasses.dataclass
class Signal:
    bar_iloc: int
    bar_time: pd.Timestamp
    signal_type: str    # bounce_long, bounce_short, break_long, break_short
    entry_price: float
    stop_price: float
    action_line: Trendline
    safety_line: Optional[Trendline]
    touchpoints: int
    risk_atr: float


def _find_opposing_trendline(
    trendlines: list[Trendline],
    direction_needed: str,
    bar_iloc: int,
    entry_price: float,
    atr_val: float,
) -> Optional[Trendline]:
    best: Optional[Trendline] = None
    best_dist = float("inf")
    for tl in trendlines:
        if tl.direction != direction_needed or tl.broken:
            continue
        tl_val = tl.value_at(bar_iloc)
        dist = abs(entry_price - tl_val)
        if dist < best_dist and dist < atr_val * MAX_RISK_ATR:
            best_dist = dist
            best = tl
    return best


def generate_signals(
    df: pd.DataFrame,
    bar_iloc: int,
    trendlines: list[Trendline],
    atr: pd.Series,
) -> list[Signal]:
    """Check the candle at *bar_iloc* for bounce / break signals."""
    signals: list[Signal] = []
    if bar_iloc < 1 or bar_iloc >= len(df):
        return signals

    atr_val = atr.iloc[bar_iloc]
    if np.isnan(atr_val) or atr_val == 0:
        return signals

    close = df["close"].iloc[bar_iloc]
    low = df["low"].iloc[bar_iloc]
    high = df["high"].iloc[bar_iloc]
    prev_close = df["close"].iloc[bar_iloc - 1]

    touch_tol = atr_val * TOUCH_TOLERANCE_ATR
    break_thr = atr_val * BREAK_THRESHOLD_ATR
    wick_buf = atr_val * WICK_BUFFER_ATR

    for tl in trendlines:
        if tl.broken or tl.num_touches < 2:
            continue
        if bar_iloc - tl.anchor1_iloc < MIN_TRENDLINE_BARS:
            continue

        tl_val = tl.value_at(bar_iloc)
        tl_prev = tl.value_at(bar_iloc - 1)

        # ── BOUNCE ──────────────────────────────────────────────────────
        if tl.direction == "up":
            if abs(low - tl_val) <= touch_tol and close > tl_val:
                stop = tl_val - wick_buf
                risk = abs(close - stop)
                if 0 < risk <= atr_val * MAX_RISK_ATR:
                    signals.append(Signal(
                        bar_iloc=bar_iloc, bar_time=df.index[bar_iloc],
                        signal_type="bounce_long", entry_price=close,
                        stop_price=stop, action_line=tl, safety_line=None,
                        touchpoints=tl.num_touches, risk_atr=risk / atr_val,
                    ))
        else:
            if abs(high - tl_val) <= touch_tol and close < tl_val:
                stop = tl_val + wick_buf
                risk = abs(stop - close)
                if 0 < risk <= atr_val * MAX_RISK_ATR:
                    signals.append(Signal(
                        bar_iloc=bar_iloc, bar_time=df.index[bar_iloc],
                        signal_type="bounce_short", entry_price=close,
                        stop_price=stop, action_line=tl, safety_line=None,
                        touchpoints=tl.num_touches, risk_atr=risk / atr_val,
                    ))

        # ── BREAK ───────────────────────────────────────────────────────
        if tl.direction == "up":
            if prev_close >= tl_prev and close < tl_val - break_thr:
                safety = _find_opposing_trendline(trendlines, "down", bar_iloc, close, atr_val)
                if safety is not None:
                    stop = safety.value_at(bar_iloc) + wick_buf
                    risk = abs(stop - close)
                    if 0 < risk <= atr_val * MAX_RISK_ATR:
                        tl.broken = True
                        tl.broken_at_iloc = bar_iloc
                        signals.append(Signal(
                            bar_iloc=bar_iloc, bar_time=df.index[bar_iloc],
                            signal_type="break_short", entry_price=close,
                            stop_price=stop, action_line=tl, safety_line=safety,
                            touchpoints=tl.num_touches, risk_atr=risk / atr_val,
                        ))
        else:
            if prev_close <= tl_prev and close > tl_val + break_thr:
                safety = _find_opposing_trendline(trendlines, "up", bar_iloc, close, atr_val)
                if safety is not None:
                    stop = safety.value_at(bar_iloc) - wick_buf
                    risk = abs(close - stop)
                    if 0 < risk <= atr_val * MAX_RISK_ATR:
                        tl.broken = True
                        tl.broken_at_iloc = bar_iloc
                        signals.append(Signal(
                            bar_iloc=bar_iloc, bar_time=df.index[bar_iloc],
                            signal_type="break_long", entry_price=close,
                            stop_price=stop, action_line=tl, safety_line=safety,
                            touchpoints=tl.num_touches, risk_atr=risk / atr_val,
                        ))

    return signals


def update_trailing_stop(
    signal: Signal,
    bar_iloc: int,
    df: pd.DataFrame,
    atr: pd.Series,
) -> float:
    """Return an updated trailing-stop price.

    Bounce trades trail along the action line; break trades trail along
    the safety line.
    """
    atr_val = atr.iloc[bar_iloc]
    wick_buf = atr_val * WICK_BUFFER_ATR if not np.isnan(atr_val) else 0

    tl = signal.safety_line if signal.safety_line is not None else signal.action_line
    tl_val = tl.value_at(bar_iloc)

    if "long" in signal.signal_type:
        return max(tl_val - wick_buf, signal.stop_price)
    else:
        return min(tl_val + wick_buf, signal.stop_price)
