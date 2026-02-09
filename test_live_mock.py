import asyncio
import pandas as pd
import numpy as np
from main import TradingBot
from model import EnsembleTradingModel, LightGBMModel, XGBoostModel

async def test_live_loop_dry_run():
    bot = TradingBot()

    # Mock some components to avoid real API calls
    class MockTrader:
        async def connect(self): pass
        async def get_balance(self): return 1000.0
        async def fetch_latest_klines(self, symbol, interval, limit=500):
            # Return some dummy klines
            return [[0]*12]*100
        async def execute_trade(self, **kwargs):
            print(f"MOCK EXECUTE: {kwargs}")
            return {"status": "FILLED"}

    bot.trader = MockTrader()

    # Mock data processing
    async def mock_get_live_data():
        # Create dummy df with all required columns
        dates = pd.date_range(end=pd.Timestamp.now(), periods=600, freq='h')
        df = pd.DataFrame(index=dates)
        df['close'] = np.random.randn(600).cumsum() + 100
        df['high'] = df['close'] + 1
        df['low'] = df['close'] - 1
        df['open'] = df['close']
        df['volume'] = 1000.0
        df['market_close'] = df['close']
        df['fundingRate'] = 0.0001 + np.random.randn(600) * 0.00001

        import features as ft
        df = ft.add_trend_features(df)
        df = ft.add_momentum_features(df)
        df = ft.add_volatility_features(df)
        df = ft.add_volume_features(df)
        df = ft.add_microstructure_features(df)
        df = ft.add_temporal_features(df)
        df = ft.add_meta_features(df)

        result = df.ffill().dropna()
        return result

    bot.get_live_data = mock_get_live_data

    # Mock Model
    class MockModel:
        def predict(self, X): return pd.Series([1], index=X.index)
        def get_confidence(self, X): return pd.Series([0.7], index=X.index)

    bot.ensemble = MockModel()

    print("Initializing bot...")
    await bot.initialize()

    print("Running dry-run iteration...")
    await bot.run_iteration()
    print("Dry-run iteration complete.")

if __name__ == "__main__":
    asyncio.run(test_live_loop_dry_run())
