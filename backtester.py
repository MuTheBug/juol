import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from risk_manager import RiskManager
from config import ROUND_TRIP_COST

logger = logging.getLogger(__name__)

class Backtester:
    """
    Event-driven backtesting engine.
    Processes data candle-by-candle.
    """
    def __init__(self, df: pd.DataFrame, initial_equity: float = 10000.0,
                 sl_mult: float = 2.0, tp_mult: float = 3.0, conf_threshold: float = 0.58):
        self.df = df
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.equity_history = []
        self.trades = []
        self.total_fees = 0.0
        self.total_funding = 0.0
        self.current_position = None
        self.risk_manager = RiskManager(initial_equity)
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.conf_threshold = conf_threshold

    def run(self):
        logger.info(f"Starting event-driven backtest with ${self.initial_equity:.2f}...")

        for i in range(len(self.df)):
            candle = self.df.iloc[i]
            timestamp = self.df.index[i]

            # 1. Update Equity and Check Circuit Breakers
            self.equity_history.append({'timestamp': timestamp, 'equity': self.equity})

            if self.equity <= 0:
                logger.critical(f"Liquidation at {timestamp}! Equity: {self.equity}")
                break

            self.risk_manager.equity = self.equity # Sync equity
            if self.risk_manager.check_circuit_breakers():
                if self.current_position:
                    self._close_position(candle, timestamp, "Circuit Breaker")
                continue # Skip trading if halted

            # 2. Check Funding Rates (at 00, 08, 16 UTC)
            if self.current_position and timestamp.hour in [0, 8, 16]:
                self._apply_funding(candle, timestamp)

            # 3. Handle Open Position
            if self.current_position:
                self._handle_open_position(candle, timestamp)

            # 4. Process New Signals (if no position)
            else:
                self._process_signals(candle, timestamp)

        return self._generate_report()

    def _process_signals(self, candle, timestamp):
        # We assume signals are already in the dataframe as 'signal' and 'confidence'
        signal = candle.get('signal', 0)
        confidence = candle.get('confidence', 0)
        regime = candle.get('regime', 0)
        trend_aligned = candle.get('trend_aligned', 0)

        # 1. Mandatory Trend Alignment Filter
        if signal == 1 and trend_aligned == -1:
            # logger.debug(f"{timestamp}: Long rejected by trend filter.")
            return
        if signal == -1 and trend_aligned == 1:
            # logger.debug(f"{timestamp}: Short rejected by trend filter.")
            return

        # 2. Confidence Filter
        threshold = self.conf_threshold
        if trend_aligned == 0:
            threshold = 0.70 # Require higher confidence if trend is flat/unclear

        # 3. Trade Quality Filters (ROUND 2)
        adx = candle.get('ADX_14', 0)
        vol_ratio = candle.get('volume_sma_ratio', 1)
        vol_regime = candle.get('volatility_regime', 0)

        is_quality = True
        if adx < 20: is_quality = False
        if vol_regime >= 3: is_quality = False
        if vol_ratio < 0.8: is_quality = False

        if signal != 0:
            if confidence < threshold:
                # logger.debug(f"{timestamp}: Signal rejected by confidence ({confidence:.2f} < {threshold:.2f})")
                return
            if not is_quality:
                # logger.debug(f"{timestamp}: Signal rejected by quality (ADX: {adx:.1f}, Vol: {vol_ratio:.1f}, Reg: {vol_regime})")
                return

        if signal != 0 and confidence >= threshold and is_quality:
            # Check funding delay
            if self.risk_manager.check_funding_entry_delay(candle['fundingRate'], signal):
                # logger.info(f"{timestamp}: Delaying entry due to high funding.")
                return

            params = self.risk_manager.calculate_position_parameters(
                price=candle['close'],
                atr=candle['atr'],
                confidence=confidence,
                regime=regime,
                sl_mult=self.sl_mult,
                tp_mult=self.tp_mult
            )

            if params['notional_size'] <= 0:
                return

            # Open Position
            self.current_position = {
                'entry_timestamp': timestamp,
                'entry_price': candle['close'],
                'side': signal, # 1 for Long, -1 for Short
                'notional_size': params['notional_size'],
                'leverage': params['leverage'],
                'sl_price': candle['close'] - (signal * params['sl_dist']),
                'tp_price': candle['close'] + (signal * params['tp_dist']),
                'tsl_activation': params['tsl_activation'],
                'tsl_distance': params['tsl_distance'],
                'tsl_active': False,
                'entry_atr': candle['atr'],
            'hours_held': 0,
            'max_favorable_price': candle['close'],
            'max_adverse_price': candle['close'],
            'initial_drawdown': 0.0,
            'confidence': confidence,
            'regime': regime
            }

            # Deduct Entry Fee
            fee = params['notional_size'] * (ROUND_TRIP_COST / 2)
            self.equity -= fee
            self.total_fees += fee
            # logger.info(f"{timestamp}: Opened {signal} at {candle['close']}. Size: {params['notional_size']:.2f}")

    def _handle_open_position(self, candle, timestamp):
        pos = self.current_position
        pos['hours_held'] += 1

        # Track MFE and MAE
        if pos['side'] == 1: # Long
            pos['max_favorable_price'] = max(pos['max_favorable_price'], candle['high'])
            pos['max_adverse_price'] = min(pos['max_adverse_price'], candle['low'])
        else: # Short
            pos['max_favorable_price'] = min(pos['max_favorable_price'], candle['low'])
            pos['max_adverse_price'] = max(pos['max_adverse_price'], candle['high'])

        # Initial Drawdown
        if pos['hours_held'] <= 3:
            dd = (pos['entry_price'] - candle['low'] if pos['side'] == 1 else candle['high'] - pos['entry_price']) / pos['entry_price']
            pos['initial_drawdown'] = max(pos['initial_drawdown'], dd)

        # 1. Partial TP (50% at 1.5x ATR)
        if not pos.get('partial_tp_hit', False):
            ptp_price = pos['entry_price'] + (pos['side'] * 1.5 * pos['entry_atr'])
            hit_ptp = (candle['high'] >= ptp_price if pos['side'] == 1 else candle['low'] <= ptp_price)
            if hit_ptp:
                pos['partial_tp_hit'] = True
                # Realize half profit
                half_notional = pos['notional_size'] / 2
                pnl_pct = (ptp_price - pos['entry_price']) / pos['entry_price'] * pos['side']
                pnl_usd = pnl_pct * half_notional
                # Fees for half close
                fee = half_notional * (ROUND_TRIP_COST / 2)
                pnl_usd -= fee

                self.equity += pnl_usd
                self.risk_manager.update_equity(pnl_usd)

                # Record the half trade
                self.trades.append({
                    'entry_timestamp': pos['entry_timestamp'],
                    'exit_timestamp': timestamp,
                    'side': pos['side'],
                    'entry_price': pos['entry_price'],
                    'exit_price': ptp_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'reason': "Partial TP (50%)",
                    'hold_time': pos['hours_held']
                })

                # Adjust remaining position
                pos['notional_size'] = half_notional
                # Move stop to breakeven for remaining 50%
                pos['sl_price'] = pos['entry_price']

        # Check SL/TP
        hit_sl = (candle['low'] <= pos['sl_price'] if pos['side'] == 1 else candle['high'] >= pos['sl_price'])
        hit_tp = (candle['high'] >= pos['tp_price'] if pos['side'] == 1 else candle['low'] <= pos['tp_price'])

        if hit_sl:
            self._close_position(candle, timestamp, "Stop Loss", price=pos['sl_price'])
        elif hit_tp:
            self._close_position(candle, timestamp, "Take Profit", price=pos['tp_price'])
        else:
            self._update_trailing_stop(candle)
            # Improved Time Exit (ROUND 3)
            profit_atr = (candle['close'] - pos['entry_price']) * pos['side'] / pos['entry_atr']
            if pos['hours_held'] >= 12 and profit_atr < 0:
                self._close_position(candle, timestamp, "Time Exit (Early Loss)")
            elif pos['hours_held'] >= 24 and profit_atr < 0.5:
                self._close_position(candle, timestamp, "Time Exit (Stalled)")

    def _update_trailing_stop(self, candle):
        pos = self.current_position
        if not pos: return

        profit = (candle['close'] - pos['entry_price']) * pos['side']

        if not pos['tsl_active'] and profit >= pos['tsl_activation']:
            pos['tsl_active'] = True
            # logger.info("Trailing Stop Activated")

        if pos['tsl_active']:
            if pos['side'] == 1:
                new_sl = candle['close'] - pos['tsl_distance']
                pos['sl_price'] = max(pos['sl_price'], new_sl)
            else:
                new_sl = candle['close'] + pos['tsl_distance']
                pos['sl_price'] = min(pos['sl_price'], new_sl)

    def _close_position(self, candle, timestamp, reason, price=None):
        pos = self.current_position
        exit_price = price if price else candle['close']

        # Calculate PnL
        pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * pos['side']
        pnl_usd = pnl_pct * pos['notional_size']

        # Deduct Exit Fee
        fee = pos['notional_size'] * (ROUND_TRIP_COST / 2)
        pnl_usd -= fee
        self.total_fees += fee

        self.equity += pnl_usd
        # Cap equity to avoid overflow in metrics/plots
        self.equity = min(self.equity, 1e15)

        self.risk_manager.add_trade_result(pnl_pct)
        self.risk_manager.update_equity(pnl_usd)

        mfe = (pos['max_favorable_price'] - pos['entry_price']) / pos['entry_price'] * pos['side']
        mae = (pos['max_adverse_price'] - pos['entry_price']) / pos['entry_price'] * pos['side']

        trade_record = {
            'entry_timestamp': pos['entry_timestamp'],
            'exit_timestamp': timestamp,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'reason': reason,
            'hold_time': pos['hours_held'],
            'mfe': mfe,
            'mae': mae,
            'initial_drawdown': pos['initial_drawdown'],
            'confidence': pos['confidence'],
            'regime': pos['regime'],
            'entry_atr': pos['entry_atr']
        }
        self.trades.append(trade_record)
        self.current_position = None
        # logger.info(f"{timestamp}: Closed due to {reason}. PnL: ${pnl_usd:.2f} ({pnl_pct:.2%})")

    def _apply_funding(self, candle, timestamp):
        # Funding deducted from position notional
        # Binance funding rate is typically 8h
        funding_rate = candle['fundingRate']
        # If long, we pay if rate > 0. If short, we pay if rate < 0.
        # Payment = - (side * funding_rate * notional)
        payment = - (self.current_position['side'] * funding_rate * self.current_position['notional_size'])
        self.equity += payment
        self.total_funding += abs(payment) if payment < 0 else -payment # Sign convention
        self.risk_manager.update_equity(payment)
        # logger.info(f"{timestamp}: Funding payment: ${payment:.2f}")

    def _generate_report(self):
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_history)
        equity_df.set_index('timestamp', inplace=True)

        metrics = self.calculate_metrics(trades_df, equity_df)

        return {
            'trades': trades_df,
            'equity_curve': equity_df,
            'metrics': metrics
        }

    def calculate_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, Any]:
        if trades_df.empty:
            return {}

        initial_equity = self.initial_equity
        final_equity = self.equity
        total_return = (final_equity / initial_equity) - 1

        # Handle inf values
        if np.isinf(final_equity):
            total_return = np.nan

        # Time period in years
        duration_days = (equity_df.index[-1] - equity_df.index[0]).total_seconds() / (24 * 3600)
        years = max(duration_days / 365.25, 1/365.25)
        if final_equity > 0 and not np.isinf(final_equity):
            cagr = (final_equity / initial_equity) ** (1 / years) - 1
        else:
            cagr = -1.0 if final_equity <= 0 else np.nan

        # Daily returns for Sharpe/Sortino
        daily_equity = equity_df['equity'].resample('D').last().ffill()
        daily_returns = daily_equity.pct_change().dropna()

        # Annualised Sharpe (assuming 5% risk-free rate)
        # prompt says: subtract risk-free rate (use 5% annual for USD)
        rf_daily = (1 + 0.05) ** (1/365) - 1
        excess_returns = daily_returns - rf_daily
        std = excess_returns.std()
        sharpe = (excess_returns.mean() / std * np.sqrt(365)) if len(daily_returns) > 1 and std != 0 else 0

        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std()
        sortino = (daily_returns.mean() - rf_daily) / downside_deviation * np.sqrt(365) if len(downside_returns) > 1 else 0

        # Max Drawdown
        rolling_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Max Drawdown Duration
        is_drawdown = equity_df['equity'] < rolling_max
        # Find durations
        drawdown_duration = 0
        current_duration = 0
        for val in is_drawdown:
            if val:
                current_duration += 1
                drawdown_duration = max(drawdown_duration, current_duration)
            else:
                current_duration = 0

        # Trade metrics
        win_rate = len(trades_df[trades_df['pnl_usd'] > 0]) / len(trades_df)
        wins = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd']
        losses = trades_df[trades_df['pnl_usd'] <= 0]['pnl_usd']
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else np.inf
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        avg_hold_time = trades_df['hold_time'].mean()
        trades_per_month = len(trades_df) / (duration_days / 30.44)

        # Monthly performance
        monthly_equity = daily_equity.resample('ME').last()
        monthly_returns = monthly_equity.pct_change()
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        profitable_months_pct = len(monthly_returns[monthly_returns > 0]) / len(monthly_returns.dropna()) if not monthly_returns.dropna().empty else 0

        # Consecutive wins/losses
        pnl_binary = (trades_df['pnl_usd'] > 0).astype(int)
        runs = (pnl_binary != pnl_binary.shift()).cumsum()
        consecutive = trades_df.groupby(runs)['pnl_usd'].count()
        max_wins = consecutive[pnl_binary.groupby(runs).first() == 1].max() if any(pnl_binary == 1) else 0
        max_losses = consecutive[pnl_binary.groupby(runs).first() == 0].max() if any(pnl_binary == 0) else 0

        return {
            'Total Return': total_return,
            'CAGR': cagr,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar,
            'Max DD Duration (h)': drawdown_duration,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Expectancy': expectancy,
            'Avg Hold Time (h)': avg_hold_time,
            'Trades/Month': trades_per_month,
            'Best Month': best_month,
            'Worst Month': worst_month,
            '% Profitable Months': profitable_months_pct,
            'Max Consecutive Wins': max_wins,
            'Max Consecutive Losses': max_losses,
            'Fees Paid Total': self.total_fees,
            'Funding Paid Total': self.total_funding
        }
