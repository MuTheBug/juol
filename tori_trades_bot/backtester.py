"""Backtesting engine for the Tori Trades Trendline Strategy.

Walks forward through 4-hour candle data, detects trendlines, generates
signals, manages positions with trailing stops, and accounts for commissions
and funding rates.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from tori_trades_bot.config import (
    COMMISSION_PCT,
    FUNDING_RATE_ENABLED,
    INITIAL_CAPITAL,
    DEFAULT_LEVERAGE,
    MIN_TRENDLINE_BARS,
    RISK_PER_TRADE_PCT,
    SWING_ORDER,
)
from tori_trades_bot.trendline import (
    Signal,
    Trendline,
    detect_trendlines,
    generate_signals,
    update_trailing_stop,
)
from tori_trades_bot.utils import compute_atr, load_funding, load_klines, resample_to_4h


# ── Trade record ────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Trade:
    symbol: str
    signal_type: str
    touchpoints: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    position_size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    funding_paid: float = 0.0
    risk_atr: float = 0.0
    stop_price: float = 0.0
    max_favorable: float = 0.0
    bars_held: int = 0


@dataclasses.dataclass
class Position:
    trade: Trade
    signal: Signal
    current_stop: float
    is_long: bool


# ── Backtester ──────────────────────────────────────────────────────────────

class Backtester:

    def __init__(
        self,
        symbol: str,
        kline_path: str | pathlib.Path,
        funding_path: Optional[str | pathlib.Path] = None,
        initial_capital: float = INITIAL_CAPITAL,
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.equity = initial_capital

        df_1h = load_klines(kline_path)
        self.df = resample_to_4h(df_1h)
        self.atr = compute_atr(self.df)

        self.funding: Optional[pd.DataFrame] = None
        if funding_path and FUNDING_RATE_ENABLED:
            self.funding = load_funding(funding_path)

        self.trades: list[Trade] = []
        self.positions: list[Position] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

        self._trendlines: list[Trendline] = []
        self._last_tl_calc: int = -999

    def run(self) -> list[Trade]:
        n = len(self.df)
        start_bar = max(MIN_TRENDLINE_BARS + SWING_ORDER + 20, 80)

        for i in range(start_bar, n):
            self._on_bar(i)

        for pos in list(self.positions):
            self._close_position(pos, n - 1)

        return self.trades

    def _on_bar(self, i: int) -> None:
        close = self.df["close"].iloc[i]
        bar_time = self.df.index[i]

        # 1. Update trailing stops & check stop-outs
        for pos in list(self.positions):
            new_stop = update_trailing_stop(pos.signal, i, self.df, self.atr)
            if pos.is_long:
                pos.current_stop = max(pos.current_stop, new_stop)
            else:
                pos.current_stop = min(pos.current_stop, new_stop)

            hit_stop = False
            exit_price = close
            if pos.is_long and self.df["low"].iloc[i] <= pos.current_stop:
                hit_stop = True
                exit_price = min(close, pos.current_stop)
            elif not pos.is_long and self.df["high"].iloc[i] >= pos.current_stop:
                hit_stop = True
                exit_price = max(close, pos.current_stop)

            # Bounce trades: also exit if close pierces the action line
            if not hit_stop and pos.signal.safety_line is None:
                tl = pos.signal.action_line
                tl_val = tl.value_at(i)
                if pos.is_long and close < tl_val:
                    hit_stop = True
                    exit_price = close
                elif not pos.is_long and close > tl_val:
                    hit_stop = True
                    exit_price = close

            if hit_stop:
                self._close_position(pos, i, exit_price)
                continue

            # Track max favorable excursion
            if pos.is_long:
                unr = (self.df["high"].iloc[i] - pos.trade.entry_price) * pos.trade.position_size
            else:
                unr = (pos.trade.entry_price - self.df["low"].iloc[i]) * pos.trade.position_size
            pos.trade.max_favorable = max(pos.trade.max_favorable, unr)

        # 2. Funding
        self._apply_funding(i)

        # 3. Recalculate trendlines periodically
        recalc_interval = max(SWING_ORDER, 6)
        if i - self._last_tl_calc >= recalc_interval:
            self._trendlines = detect_trendlines(self.df, up_to_iloc=i, atr=self.atr)
            self._last_tl_calc = i

        # 4. Generate signals (only if flat)
        if len(self.positions) == 0:
            signals = generate_signals(self.df, i, self._trendlines, self.atr)
            if signals:
                best = sorted(signals, key=lambda s: (-s.touchpoints, s.risk_atr))[0]
                self._open_position(best, i)

        # 5. Record equity
        unrealised = 0.0
        for pos in self.positions:
            if pos.is_long:
                unrealised += (close - pos.trade.entry_price) * pos.trade.position_size
            else:
                unrealised += (pos.trade.entry_price - close) * pos.trade.position_size
        self.equity_curve.append((bar_time, self.equity + unrealised))

    def _open_position(self, signal: Signal, bar_iloc: int) -> None:
        risk_per_unit = abs(signal.entry_price - signal.stop_price)
        if risk_per_unit == 0:
            return

        risk_amount = self.equity * (RISK_PER_TRADE_PCT / 100.0)
        position_size = risk_amount / risk_per_unit

        max_notional = self.equity * DEFAULT_LEVERAGE
        if position_size * signal.entry_price > max_notional:
            position_size = max_notional / signal.entry_price

        entry_comm = position_size * signal.entry_price * (COMMISSION_PCT / 100.0)
        self.equity -= entry_comm

        trade = Trade(
            symbol=self.symbol,
            signal_type=signal.signal_type,
            touchpoints=signal.touchpoints,
            entry_time=signal.bar_time,
            entry_price=signal.entry_price,
            position_size=position_size,
            commission=entry_comm,
            risk_atr=signal.risk_atr,
            stop_price=signal.stop_price,
        )
        self.positions.append(Position(
            trade=trade, signal=signal,
            current_stop=signal.stop_price,
            is_long="long" in signal.signal_type,
        ))

    def _close_position(self, pos: Position, bar_iloc: int, exit_price: float | None = None) -> None:
        if exit_price is None:
            exit_price = self.df["close"].iloc[bar_iloc]

        trade = pos.trade
        trade.exit_time = self.df.index[bar_iloc]
        trade.exit_price = exit_price
        trade.bars_held = bar_iloc - pos.signal.bar_iloc

        if pos.is_long:
            gross_pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            gross_pnl = (trade.entry_price - exit_price) * trade.position_size

        exit_comm = trade.position_size * exit_price * (COMMISSION_PCT / 100.0)
        trade.commission += exit_comm
        trade.pnl = gross_pnl - exit_comm
        notional = trade.position_size * trade.entry_price
        trade.pnl_pct = (trade.pnl / notional) * 100 if notional > 0 else 0

        self.equity += gross_pnl - exit_comm
        self.trades.append(trade)
        self.positions.remove(pos)

    def _apply_funding(self, bar_iloc: int) -> None:
        if self.funding is None or not self.positions:
            return
        bar_time = self.df.index[bar_iloc]
        if bar_time.hour not in (0, 8, 16) or bar_time.minute != 0:
            return
        mask = (self.funding.index >= bar_time) & (
            self.funding.index < bar_time + pd.Timedelta(hours=4)
        )
        rates = self.funding[mask]
        if rates.empty:
            return
        rate = rates["fundingRate"].iloc[0]
        for pos in self.positions:
            notional = pos.trade.position_size * self.df["close"].iloc[bar_iloc]
            funding_cost = notional * rate if pos.is_long else -notional * rate
            pos.trade.funding_paid += funding_cost
            self.equity -= funding_cost


# ── Run & report ────────────────────────────────────────────────────────────

def run_backtest(
    data_dir: str | pathlib.Path = ".",
    symbols: list[str] | None = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    from tori_trades_bot.config import SYMBOLS as DEFAULT_SYMBOLS
    data_dir = pathlib.Path(data_dir)
    symbols = symbols or DEFAULT_SYMBOLS

    all_trades: list[Trade] = []
    equity_curves: dict[str, list] = {}

    for sym in symbols:
        kline_file = data_dir / f"{sym}_1h_klines.csv"
        funding_file = data_dir / f"{sym}_funding_8h.csv"
        if not kline_file.exists():
            print(f"  [SKIP] {sym}: kline file not found")
            continue

        print(f"  Running {sym} ...", end=" ", flush=True)
        bt = Backtester(
            symbol=sym,
            kline_path=str(kline_file),
            funding_path=str(funding_file) if funding_file.exists() else None,
            initial_capital=initial_capital,
        )
        trades = bt.run()
        all_trades.extend(trades)
        equity_curves[sym] = bt.equity_curve
        print(f"{len(trades)} trades, final equity ${bt.equity:,.2f}")

    return {
        "trades": all_trades,
        "equity_curves": equity_curves,
        "initial_capital": initial_capital,
    }


def build_report(results: dict) -> str:
    trades = results["trades"]
    cap = results["initial_capital"]

    if not trades:
        return "No trades generated."

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  TORI TRADES TRENDLINE STRATEGY — BACKTEST REPORT")
    lines.append("=" * 80)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    total_comm = sum(t.commission for t in trades)
    total_funding = sum(t.funding_paid for t in trades)
    final_equity = cap + total_pnl

    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0
    avg_bars = np.mean([t.bars_held for t in trades])

    # Date range
    first_trade = min(t.entry_time for t in trades)
    last_trade = max(t.exit_time for t in trades if t.exit_time)

    # Max drawdown
    all_eq: list[tuple] = []
    for sym_curve in results["equity_curves"].values():
        all_eq.extend(sym_curve)
    max_dd = 0.0
    if all_eq:
        all_eq.sort(key=lambda x: x[0])
        peak = all_eq[0][1]
        for _, v in all_eq:
            peak = max(peak, v)
            dd = (peak - v) / peak * 100
            max_dd = max(max_dd, dd)

    lines.append("")
    lines.append(f"  Period:                {first_trade.strftime('%Y-%m-%d')} to {last_trade.strftime('%Y-%m-%d')}")
    lines.append(f"  Initial Capital:       ${cap:>14,.2f}")
    lines.append(f"  Final Equity:          ${final_equity:>14,.2f}")
    lines.append(f"  Net P&L:               ${total_pnl:>14,.2f}  ({total_pnl/cap*100:+.2f}%)")
    lines.append(f"  Total Commissions:     ${total_comm:>14,.2f}")
    lines.append(f"  Total Funding Paid:    ${total_funding:>14,.2f}")
    lines.append("")
    lines.append(f"  Total Trades:          {len(trades):>10}")
    lines.append(f"  Winners:               {len(wins):>10}  ({len(wins)/len(trades)*100:.1f}%)")
    lines.append(f"  Losers:                {len(losses):>10}  ({len(losses)/len(trades)*100:.1f}%)")
    lines.append(f"  Profit Factor:         {profit_factor:>10.2f}")
    lines.append(f"  Avg Win:               ${avg_win:>14,.2f}  ({avg_win_pct:+.2f}%)")
    lines.append(f"  Avg Loss:              ${avg_loss:>14,.2f}  ({avg_loss_pct:+.2f}%)")
    lines.append(f"  Avg Bars Held:         {avg_bars:>10.1f}  (~{avg_bars*4:.0f} hours)")
    lines.append(f"  Max Drawdown:          {max_dd:>10.2f}%")

    # Per-symbol
    lines.append("")
    lines.append("-" * 80)
    lines.append("  PER-SYMBOL BREAKDOWN")
    lines.append("-" * 80)
    for sym in sorted(set(t.symbol for t in trades)):
        st = [t for t in trades if t.symbol == sym]
        sw = [t for t in st if t.pnl > 0]
        sp = sum(t.pnl for t in st)
        wr = len(sw) / len(st) * 100 if st else 0
        lines.append(f"  {sym:12s}  trades={len(st):>4}  wins={len(sw):>4}  "
                      f"WR={wr:5.1f}%  P&L=${sp:>12,.2f}")

    # Per-setup-type
    lines.append("")
    lines.append("-" * 80)
    lines.append("  PER-SETUP-TYPE BREAKDOWN")
    lines.append("-" * 80)
    for st_type in sorted(set(t.signal_type for t in trades)):
        st = [t for t in trades if t.signal_type == st_type]
        sw = [t for t in st if t.pnl > 0]
        sp = sum(t.pnl for t in st)
        wr = len(sw) / len(st) * 100 if st else 0
        lines.append(f"  {st_type:20s}  trades={len(st):>4}  wins={len(sw):>4}  "
                      f"WR={wr:5.1f}%  P&L=${sp:>12,.2f}")

    # By touchpoints
    lines.append("")
    lines.append("-" * 80)
    lines.append("  BY TOUCHPOINT COUNT")
    lines.append("-" * 80)
    for tp in sorted(set(t.touchpoints for t in trades)):
        st = [t for t in trades if t.touchpoints == tp]
        sw = [t for t in st if t.pnl > 0]
        sp = sum(t.pnl for t in st)
        wr = len(sw) / len(st) * 100 if st else 0
        lines.append(f"  {tp}-touch        trades={len(st):>4}  wins={len(sw):>4}  "
                      f"WR={wr:5.1f}%  P&L=${sp:>12,.2f}")

    # Top 10 winners / losers
    lines.append("")
    lines.append("-" * 80)
    lines.append("  TOP 10 WINNING TRADES")
    lines.append("-" * 80)
    for t in sorted(trades, key=lambda x: x.pnl, reverse=True)[:10]:
        lines.append(
            f"  {t.symbol:10s} {t.signal_type:18s} "
            f"entry={t.entry_time.strftime('%Y-%m-%d %H:%M')} "
            f"P&L=${t.pnl:>10,.2f} ({t.pnl_pct:+.2f}%) bars={t.bars_held}"
        )

    lines.append("")
    lines.append("-" * 80)
    lines.append("  TOP 10 LOSING TRADES")
    lines.append("-" * 80)
    for t in sorted(trades, key=lambda x: x.pnl)[:10]:
        lines.append(
            f"  {t.symbol:10s} {t.signal_type:18s} "
            f"entry={t.entry_time.strftime('%Y-%m-%d %H:%M')} "
            f"P&L=${t.pnl:>10,.2f} ({t.pnl_pct:+.2f}%) bars={t.bars_held}"
        )

    # Monthly returns
    lines.append("")
    lines.append("-" * 80)
    lines.append("  MONTHLY P&L")
    lines.append("-" * 80)
    monthly: dict[str, float] = {}
    for t in trades:
        key = t.entry_time.strftime("%Y-%m")
        monthly[key] = monthly.get(key, 0) + t.pnl
    for month, pnl in sorted(monthly.items()):
        bar_len = min(40, int(abs(pnl) / 20))
        bar = ("+" * bar_len) if pnl > 0 else ("-" * bar_len)
        lines.append(f"  {month}  ${pnl:>12,.2f}  {bar}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)
