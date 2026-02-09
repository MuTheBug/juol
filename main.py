import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

import data_pipeline as dp
import features as ft
from model import RegimeFilter, LightGBMModel, XGBoostModel, EnsembleTradingModel
from risk_manager import RiskManager
from live_trader import LiveTrader
from telegram_notifier import TelegramNotifier
from config import (BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_TOKEN,
                    TELEGRAM_CHAT_ID, PRIMARY_ASSET, SECONDARY_ASSET, TIMEFRAME)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")

class TradingBot:
    def __init__(self):
        self.notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        self.trader = LiveTrader(BINANCE_API_KEY, BINANCE_API_SECRET, self.notifier)
        self.risk_manager = RiskManager(initial_equity=0.0) # Will be updated from balance
        self.ensemble: Optional[EnsembleTradingModel] = None
        self.regime_filter = RegimeFilter()
        self.is_live = False

    async def initialize(self):
        await self.trader.connect()
        balance = await self.trader.get_balance()
        self.risk_manager.equity = balance
        self.risk_manager.peak_equity = balance
        logger.info(f"Bot initialized. Balance: ${balance:.2f}")

    async def get_live_data(self):
        """Fetches and processes data for live inference."""
        # Fetch data for primary and secondary assets
        p_klines = await self.trader.fetch_latest_klines(PRIMARY_ASSET, TIMEFRAME)
        s_klines = await self.trader.fetch_latest_klines(SECONDARY_ASSET, TIMEFRAME)

        # Convert to DataFrames
        cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'tbb', 'tbq', 'ignore']
        p_df = pd.DataFrame(p_klines, columns=cols)
        s_df = pd.DataFrame(s_klines, columns=cols)

        for df in [p_df, s_df]:
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('timestamp', inplace=True)
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c])

        # Merge (simplified version of dp.get_merged_data for live)
        s_df = s_df.rename(columns={c: f"market_{c}" for c in s_df.columns})
        df = p_df[['open', 'high', 'low', 'close', 'volume']].join(s_df[['market_close']], how='inner')

        # Add funding (fetching real-time if needed, but for now we'll mock or use last known)
        df['fundingRate'] = 0.0001 # Placeholder

        # Add Features
        df = ft.add_trend_features(df)
        df = ft.add_momentum_features(df)
        df = ft.add_volatility_features(df)
        df = ft.add_volume_features(df)
        df = ft.add_microstructure_features(df)
        df = ft.add_temporal_features(df)
        df = ft.add_meta_features(df)

        # Preprocessing (Rolling Z-Score needs enough history)
        # Assuming we fetched 500 klines, we can apply z-score
        from data_pipeline import apply_rolling_zscore
        from config import EXCLUDE_COLS
        df = apply_rolling_zscore(df, EXCLUDE_COLS)

        return df.dropna()

    async def run_iteration(self):
        logger.info("Starting iteration...")

        # 1. Fetch data
        df = await self.get_live_data()
        if df.empty:
            logger.error("No data available for inference.")
            return

        last_candle = df.iloc[-1]
        from config import EXCLUDE_COLS
        X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns], errors='ignore')
        X = X.tail(1)

        # 2. Regime Filter
        regime = self.regime_filter.fit_predict(df).iloc[-1]
        regime_name = {0: 'TRENDING', 1: 'RANGING', 2: 'VOLATILE'}.get(regime, 'UNKNOWN')

        # 3. Model Inference
        if not self.ensemble:
            logger.error("Model not loaded!")
            return

        signal = self.ensemble.predict(X).iloc[0]
        confidence = self.ensemble.get_confidence(X).iloc[0]

        logger.info(f"Signal: {signal}, Confidence: {confidence:.2%}, Regime: {regime_name}")

        # 4. Check Risk & Execute
        if signal != 0:
            if self.risk_manager.check_circuit_breakers():
                logger.warning("Circuit breaker active. Skipping trade.")
                return

            params = self.risk_manager.calculate_position_parameters(
                price=last_candle['close'],
                atr=last_candle['atr'],
                confidence=confidence,
                regime=regime
            )

            if params['notional_size'] > 0:
                await self.trader.execute_trade(
                    symbol=PRIMARY_ASSET,
                    side=int(signal),
                    notional=params['notional_size'],
                    leverage=int(params['leverage']),
                    sl_price=last_candle['close'] - (signal * params['sl_dist']),
                    tp_price=last_candle['close'] + (signal * params['tp_dist'])
                )

                self.notifier.notify_trade_open(
                    asset=PRIMARY_ASSET,
                    side="LONG" if signal == 1 else "SHORT",
                    price=last_candle['close'],
                    size=params['notional_size'],
                    leverage=params['leverage'],
                    sl=last_candle['close'] - (signal * params['sl_dist']),
                    tp=last_candle['close'] + (signal * params['tp_dist']),
                    confidence=confidence,
                    regime=regime_name
                )

    async def main_loop(self):
        await self.initialize()

        while True:
            now = datetime.utcnow()
            # Run at :01 past the hour
            next_run = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
            wait_seconds = (next_run - now).total_seconds()

            logger.info(f"Waiting {wait_seconds/60:.1f} minutes until next run at {next_run}")
            await asyncio.sleep(wait_seconds)

            try:
                await self.run_iteration()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.notifier.notify_error(str(e))

if __name__ == "__main__":
    import os
    import sys

    bot = TradingBot()

    model_path = "best_model.joblib"
    if os.path.exists(model_path):
        logger.info(f"Loading ensemble model from {model_path}")
        bot.ensemble = EnsembleTradingModel.load(model_path)
    else:
        logger.error(f"Ensemble model not found at {model_path}. Bot cannot start.")
        sys.exit(1)

    try:
        asyncio.run(bot.main_loop())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
