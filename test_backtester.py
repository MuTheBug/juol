import pandas as pd
import numpy as np
from backtester import Backtester

def test_backtester_basic():
    # Create 100 rows of dummy data
    dates = pd.date_range("2020-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        'open': [100.0] * 100,
        'high': [101.0] * 100,
        'low': [99.0] * 100,
        'close': [100.0] * 100,
        'volume': [1000] * 100,
        'fundingRate': [0.0001] * 100,
        'atr': [2.0] * 100,
        'regime': [0] * 100,
        'signal': [0] * 100,
        'confidence': [0.7] * 100
    }, index=dates)

    # Induce a long signal
    df.loc[dates[10], 'signal'] = 1
    # Induce a take profit hit at index 15
    df.loc[dates[15], 'high'] = 110.0

    bt = Backtester(df, initial_equity=1000.0)
    report = bt.run()

    print(report['trades'])
    assert len(report['trades']) == 1
    assert report['trades'].iloc[0]['reason'] == 'Take Profit'
    print("Basic Backtester Test Passed")

if __name__ == "__main__":
    test_backtester_basic()
