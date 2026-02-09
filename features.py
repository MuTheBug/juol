import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List
import logging
from config import (EMA_PAIRS, LINREG_LOOKBACKS, ROC_LOOKBACKS, ATR_PERIOD, RSI_PERIOD,
                    MACD_PARAMS, SUPERTREND_PARAMS, BTC_CORR_LOOKBACKS, SHARPE_LOOKBACKS)

logger = logging.getLogger(__name__)

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds trend-based features to the dataframe."""
    logger.info("Adding trend features...")

    # EMA Crosses (using ratios for better model normalization)
    for fast, slow in EMA_PAIRS:
        ema_fast = ta.ema(df['close'], length=fast)
        ema_slow = ta.ema(df['close'], length=slow)
        df[f'ema_{fast}_{slow}_ratio'] = ema_fast / ema_slow
        # Cross signals
        df[f'ema_{fast}_{slow}_cross'] = np.where(ema_fast > ema_slow, 1, -1)

    # Linear Regression Slope
    for period in LINREG_LOOKBACKS:
        # pandas_ta linreg returns multiple values, we want the slope
        lr = ta.linreg(df['close'], length=period)
        # The slope is not directly returned by linreg in some versions,
        # but ta.slope can be used.
        df[f'slope_{period}'] = ta.slope(df['close'], length=period)

    # ADX and Directional Indicators
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df = pd.concat([df, adx], axis=1)

    # Supertrend
    for length, multiplier in SUPERTREND_PARAMS:
        st = ta.supertrend(df['high'], df['low'], df['close'], length=length, multiplier=multiplier)
        # Supertrend returns [ST_L, ST_D, ST_S, ST_X]
        # ST_D is the direction (1 or -1)
        df[f'supertrend_{length}_{multiplier}_dir'] = st[f'SUPERTd_{length}_{multiplier}']
        # Distance to supertrend
        df[f'supertrend_{length}_{multiplier}_dist'] = (df['close'] / st[f'SUPERT_{length}_{multiplier}']) - 1

    # VWAP position (session-based)
    # Since we have 1H candles, ta.vwap with daily anchor (default) works well.
    vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    df['vwap'] = vwap
    df['vwap_dist'] = (df['close'] / df['vwap']) - 1

    return df

def detect_rsi_divergence(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Detects simple RSI divergence.
    Returns 1 for bullish, -1 for bearish, 0 for none.
    This is a simplified version.
    """
    # Placeholder for a more complex divergence detection
    # For now, let's just use the change in price vs change in RSI
    price_delta = df['close'].diff(window)
    rsi_delta = df['rsi'].diff(window)

    bull_div = (price_delta < 0) & (rsi_delta > 0)
    bear_div = (price_delta > 0) & (rsi_delta < 0)

    return np.where(bull_div, 1, np.where(bear_div, -1, 0))

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds momentum-based features to the dataframe."""
    logger.info("Adding momentum features...")

    # RSI
    df['rsi'] = ta.rsi(df['close'], length=RSI_PERIOD)
    df['rsi_divergence'] = detect_rsi_divergence(df, window=14)

    # MACD
    macd = ta.macd(df['close'], fast=MACD_PARAMS[0], slow=MACD_PARAMS[1], signal=MACD_PARAMS[2])
    # Returns [MACD, MACDh, MACDs]
    df['macd_hist'] = macd[f'MACDh_{MACD_PARAMS[0]}_{MACD_PARAMS[1]}_{MACD_PARAMS[2]}']
    df['macd_hist_slope'] = ta.slope(df['macd_hist'], length=3)

    # Stochastic RSI
    stoch_rsi = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)
    # Returns [STOCHRSIk, STOCHRSId]
    df = pd.concat([df, stoch_rsi], axis=1)

    # ROC
    for period in ROC_LOOKBACKS:
        df[f'roc_{period}'] = ta.roc(df['close'], length=period)

    # Williams %R
    df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14)

    return df

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds volatility-based features to the dataframe."""
    logger.info("Adding volatility features...")

    # ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_ratio'] = df['atr'] / df['close']

    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2)
    # Adjust column names based on pandas_ta version
    # It seems to use 'BBB_20_2.0_2.0' in this version
    col_width = [c for c in bb.columns if 'BBB' in c][0]
    col_percent = [c for c in bb.columns if 'BBP' in c][0]
    df['bb_width'] = bb[col_width]
    df['bb_percent'] = bb[col_percent]

    # Keltner Channels
    kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=2)
    # Returns [KCLe_20_2, KCBe_20_2, KCUe_20_2]
    col_upper = [c for c in kc.columns if 'KCU' in c][0]
    col_lower = [c for c in kc.columns if 'KCL' in c][0]
    col_basis = [c for c in kc.columns if 'KCB' in c][0]
    df['kc_width'] = (kc[col_upper] - kc[col_lower]) / kc[col_basis]

    # Parkinson Volatility
    # Parkinson = sqrt(1 / (4 * log(2) * n) * sum(log(high/low)^2))
    # We'll use a rolling version
    window = 14
    high_low_ratio = np.log(df['high'] / df['low']) ** 2
    parkinson = np.sqrt((1 / (4 * np.log(2))) * high_low_ratio.rolling(window).mean())
    df['parkinson_vol'] = parkinson

    # Volatility Regime Classifier
    # low/medium/high/extreme using rolling percentile of ATR
    atr_percentile = df['atr'].rolling(168).rank(pct=True)
    df['volatility_regime'] = pd.cut(atr_percentile,
                                     bins=[0, 0.25, 0.5, 0.75, 1.0],
                                     labels=[0, 1, 2, 3]).astype(float)

    return df

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds volume-based features to the dataframe."""
    logger.info("Adding volume features...")

    # OBV
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['obv_slope'] = ta.slope(df['obv'], length=5)

    # Volume SMA Ratio
    df['volume_sma_ratio'] = df['volume'] / ta.sma(df['volume'], length=20)

    # VWAP deviation (already have vwap_dist from trend features, but let's ensure it's here)
    if 'vwap_dist' not in df.columns:
        vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['vwap_dist'] = (df['close'] / vwap) - 1

    # Volume-Price Trend (VPT)
    # VPT = cumulative_sum(volume * (close - prev_close) / prev_close)
    df['vpt'] = ta.pvt(df['close'], df['volume']) # PVT is similar to VPT

    # Accumulation/Distribution Line
    df['ad_line'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])

    return df

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds market microstructure features to the dataframe."""
    logger.info("Adding microstructure features...")

    # Funding Rate Features
    df['funding_8h_ma'] = df['fundingRate'].rolling(8).mean()
    df['funding_24h_ma'] = df['fundingRate'].rolling(24).mean()
    df['funding_zscore'] = (df['fundingRate'] - df['fundingRate'].rolling(168).mean()) / df['fundingRate'].rolling(168).std()

    # BTC Correlation
    for period in BTC_CORR_LOOKBACKS:
        df[f'btc_corr_{period}h'] = df['close'].rolling(period).corr(df['market_close'])

    # BTC RSI
    df['btc_rsi'] = ta.rsi(df['market_close'], length=14)

    return df

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds temporal and cyclical features to the dataframe."""
    logger.info("Adding temporal features...")

    # Hour of day (cyclical)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    # Day of week (cyclical)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Sessions (One-hot)
    # Asian: 00:00 - 08:00 UTC
    # European: 08:00 - 16:00 UTC
    # US: 16:00 - 24:00 UTC
    df['is_asian'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
    df['is_european'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
    df['is_us'] = ((df.index.hour >= 16) & (df.index.hour < 24)).astype(int)

    # Hours since last local high/low (24H window)
    window = 24
    df['last_24h_high'] = df['high'].rolling(window).max()
    df['last_24h_low'] = df['low'].rolling(window).min()

    # This is slightly tricky to do efficiently for all rows, but we can do it.
    # For each row, we want to know how many hours ago the last high occurred.
    # Using argmax on rolling windows.
    def hours_since_extreme(series):
        return window - 1 - series.argmax()

    df['hours_since_24h_high'] = df['high'].rolling(window).apply(hours_since_extreme, raw=True)
    df['hours_since_24h_low'] = df['low'].rolling(window).apply(hours_since_extreme, raw=True)

    return df

def add_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds meta features to the dataframe."""
    logger.info("Adding meta features...")

    # Rolling Sharpe (24H, 72H, 168H)
    returns = df['close'].pct_change()
    for period in SHARPE_LOOKBACKS:
        # Annualized Sharpe: (mean / std) * sqrt(period_per_year)
        # Here 1H candles, so 365*24 = 8760
        role_mean = returns.rolling(period).mean()
        role_std = returns.rolling(period).std()
        df[f'sharpe_{period}h'] = (role_mean / role_std) * np.sqrt(8760)

    # Distance from 24H high/low as percentage
    df['dist_from_24h_high'] = (df['close'] / df['last_24h_high']) - 1
    df['dist_from_24h_low'] = (df['close'] / df['last_24h_low']) - 1

    # Consecutive candle direction count
    # +1 if close > open, -1 if close < open
    direction = np.sign(df['close'] - df['open'])
    # We want to count consecutive ones.
    # We can use a trick: (direction != direction.shift()).cumsum()
    # But for a simple count:
    consecutive = []
    count = 0
    prev_dir = 0
    for d in direction:
        if d == prev_dir:
            if d > 0: count += 1
            elif d < 0: count -= 1
            else: count = 0
        else:
            count = d
        consecutive.append(count)
        prev_dir = d
    df['consecutive_direction'] = consecutive

    return df
