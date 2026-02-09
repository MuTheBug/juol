import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from config import (MAX_LEVERAGE, MAX_POSITION_SIZE, ROUND_TRIP_COST)

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Independent Risk Management Engine.
    Handles position sizing, leverage, SL/TP, and circuit breakers.
    """
    def __init__(self, initial_equity: float):
        self.equity = initial_equity
        self.peak_equity = initial_equity
        self.daily_loss = 0.0
        self.weekly_loss = 0.0
        self.trade_history = [] # List of trade results (profit/loss percentage)
        self.is_halted = False

    def calculate_kelly_size(self, window: int = 50) -> float:
        """
        Calculates Fractional Kelly (Half Kelly).
        Full Kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        """
        if len(self.trade_history) < 10:
            return 0.05 # Conservative default for first trades

        recent_trades = self.trade_history[-window:]
        wins = [t for t in recent_trades if t > 0]
        losses = [t for t in recent_trades if t <= 0]

        if not wins or not losses:
            return 0.05

        win_rate = len(wins) / len(recent_trades)
        loss_rate = 1 - win_rate
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        full_kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        half_kelly = full_kelly / 2

        return max(0.0, min(half_kelly, MAX_POSITION_SIZE))

    def get_base_leverage(self, regime: int) -> int:
        """
        Dynamic leverage based on market regime.
        Regime mapping (GMM): 0: TRENDING, 1: RANGING, 2: VOLATILE
        """
        if regime == 0: # TRENDING
            return 10
        elif regime == 1: # RANGING
            return 5
        elif regime == 2: # VOLATILE/CHAOTIC
            return 3
        return 1

    def calculate_position_parameters(self,
                                     price: float,
                                     atr: float,
                                     confidence: float,
                                     regime: int,
                                     sl_mult: float = 1.5,
                                     tp_mult: float = 2.5) -> Dict[str, Any]:
        """
        Calculates leverage, position size, SL, and TP.
        """
        # 1. Base Leverage based on regime
        base_leverage = self.get_base_leverage(regime)

        # 2. Confidence-scaled leverage
        # leverage = base_leverage * (confidence - 0.5) * 2
        # Ensure confidence is at least 0.5
        adj_confidence = max(0.0, confidence - 0.5) * 2
        leverage = min(base_leverage * adj_confidence, MAX_LEVERAGE)
        leverage = max(1.0, round(leverage, 1))

        # 3. Kelly sizing
        kelly_size = self.calculate_kelly_size()

        # Drawdown circuit breaker level 1 (15% drop)
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if drawdown >= 0.15:
            kelly_size *= 0.5
            logger.warning(f"Circuit Breaker Level 1: Drawdown {drawdown:.2%}. Reducing size by 50%.")

        # Notional size
        notional_size = max(0.0, self.equity) * kelly_size * leverage

        # 4. SL / TP
        # SL = sl_mult * ATR, TP = tp_mult * ATR
        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr

        # 5. Trailing and Time-based exit parameters
        tsl_activation = sl_mult * atr
        tsl_distance = 1.0 * atr
        time_exit_hours = 24
        time_exit_profit_threshold = 0.5 * atr

        # 6. Min size check
        if notional_size < 5.0:
            logger.info(f"Notional size ${notional_size:.2f} is below Binance minimum $5. Adjusting.")
            notional_size = 5.0 if self.equity >= 5.0 else 0.0

        return {
            'leverage': leverage,
            'kelly_fraction': kelly_size,
            'notional_size': notional_size,
            'sl_dist': sl_dist,
            'tp_dist': tp_dist,
            'tsl_activation': tsl_activation,
            'tsl_distance': tsl_distance,
            'time_exit_hours': time_exit_hours,
            'time_exit_profit_threshold': time_exit_profit_threshold
        }

    def check_funding_entry_delay(self, funding_rate: float, direction: int) -> bool:
        """
        If funding settlement is within next 2 hours AND funding rate > 0.05%
        AND we'd be paying: delay entry.
        direction: 1 for long, -1 for short.
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        # Next settlement hours: 0, 8, 16
        settlement_hours = [0, 8, 16]
        next_settlement = None
        for h in settlement_hours:
            target = now.replace(hour=h, minute=0, second=0, microsecond=0)
            if target > now:
                next_settlement = target
                break

        if not next_settlement: # It's after 16:00, next is 00:00 tomorrow
            next_settlement = now.replace(day=now.day+1, hour=0, minute=0, second=0, microsecond=0)

        time_to_settlement = (next_settlement - now).total_seconds() / 3600

        is_close = time_to_settlement <= 2.0
        is_high = abs(funding_rate) > 0.0005
        is_paying = (direction == 1 and funding_rate > 0) or (direction == -1 and funding_rate < 0)

        if is_close and is_high and is_paying:
            logger.warning(f"Delaying entry: Funding settlement in {time_to_settlement:.1f}h, Rate: {funding_rate:.4%}")
            return True

        return False

    def check_circuit_breakers(self) -> bool:
        """
        Checks daily/weekly loss limits and max drawdown.
        Returns True if trading should be halted.
        """
        # Daily limit -3%
        if self.daily_loss <= -0.03 * self.equity:
            logger.error("Daily loss limit hit!")
            self.is_halted = True
            return True

        # Weekly limit -7%
        if self.weekly_loss <= -0.07 * self.equity:
            logger.error("Weekly loss limit hit!")
            self.is_halted = True
            return True

        # Drawdown Level 2 (25%)
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if drawdown >= 0.25:
            logger.critical("Drawdown circuit breaker level 2 hit! Trading halted.")
            self.is_halted = True
            return True

        return False

    def update_equity(self, pnl: float):
        """Updates equity and peak equity."""
        self.equity += pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # In a real system we'd reset daily/weekly loss at start of period
        self.daily_loss += pnl
        self.weekly_loss += pnl

    def add_trade_result(self, pnl_pct: float):
        """Adds trade P&L percentage to history for Kelly calculation."""
        self.trade_history.append(pnl_pct)
