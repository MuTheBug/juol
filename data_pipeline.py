import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging
from config import (DATA_DIR, PRIMARY_ASSET, SECONDARY_ASSET, BARRIER_ATR_MULT,
                    BARRIER_TIME_LIMIT, ROLLING_Z_LOOKBACK, CORRELATION_THRESHOLD)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_klines(symbol: str) -> pd.DataFrame:
    """Loads 1H klines from CSV."""
    file_path = DATA_DIR / f"{symbol}_1h_klines.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found.")

    df = pd.read_csv(file_path)
    # Binance format: open_time is in ms
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Ensure numeric types
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    return df[cols]

def load_funding_rates(symbol: str) -> pd.DataFrame:
    """Loads 8H funding rates from CSV."""
    file_path = DATA_DIR / f"{symbol}_funding_8h.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found.")

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')

    return df[['fundingRate']]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handles NaN values by forward filling and then dropping leading NaNs."""
    initial_shape = df.shape
    df = df.ffill()
    df = df.dropna()
    final_shape = df.shape

    if initial_shape[0] != final_shape[0]:
        logger.info(f"Cleaned data: {initial_shape[0] - final_shape[0]} rows dropped/filled.")

    return df

def get_merged_data(primary_symbol: str, secondary_symbol: str) -> pd.DataFrame:
    """Loads and merges primary klines, secondary klines, and primary funding rates."""
    logger.info(f"Loading data for {primary_symbol} and {secondary_symbol}...")

    primary_df = load_klines(primary_symbol)
    secondary_df = load_klines(secondary_symbol)
    funding_df = load_funding_rates(primary_symbol)

    # Rename secondary columns to avoid collision
    secondary_df = secondary_df.rename(columns={col: f"market_{col}" for col in secondary_df.columns})

    # Merge klines
    df = primary_df.join(secondary_df, how='inner')

    # Merge funding rates (8H data to 1H data)
    # Use merge_asof or reindex/ffill. Since it's 1H vs 8H, ffill is appropriate.
    df = df.join(funding_df, how='left')
    df['fundingRate'] = df['fundingRate'].ffill()

    # Clean data
    df = clean_data(df)

    logger.info(f"Merged data shape: {df.shape}")
    return df

def apply_triple_barrier_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Triple Barrier Method for labeling.
    Upper barrier: +1.5 * ATR(14)
    Lower barrier: -1.5 * ATR(14)
    Time barrier: 12 candles
    """
    logger.info("Applying Triple Barrier Method labels...")

    # Calculate ATR(14) for barriers
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_barrier'] = atr

    labels = []

    # We need to look forward, so we iterate up to len - time_limit
    # For performance, we can use vectorized approach or a loop if necessary.
    # Given the requirements, we'll implement it carefully.

    closes = df['close'].values
    atrs = df['atr_barrier'].values

    for i in range(len(df)):
        if i + BARRIER_TIME_LIMIT >= len(df):
            labels.append(np.nan)
            continue

        entry_price = closes[i]
        atr_val = atrs[i]

        if np.isnan(atr_val):
            labels.append(np.nan)
            continue

        upper_barrier = entry_price + (BARRIER_ATR_MULT * atr_val)
        lower_barrier = entry_price - (BARRIER_ATR_MULT * atr_val)

        label = 0 # Default to timeout
        for j in range(1, BARRIER_TIME_LIMIT + 1):
            future_high = df['high'].iloc[i+j]
            future_low = df['low'].iloc[i+j]

            # Check if both hit in same candle - conservative: hit SL first
            # But here we use high/low for barrier checks
            hit_upper = future_high >= upper_barrier
            hit_lower = future_low <= lower_barrier

            if hit_upper and hit_lower:
                label = -1 # Conservative
                break
            elif hit_upper:
                label = 1
                break
            elif hit_lower:
                label = -1
                break

        labels.append(label)

    df['triple_barrier_label'] = labels
    return df

def apply_continuous_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates risk-adjusted forward return (12H forward return / current ATR).
    """
    logger.info("Applying continuous target calculation...")

    # 12-hour forward return: (price[t+12] - price[t]) / price[t]
    df['forward_return_12h'] = df['close'].shift(-BARRIER_TIME_LIMIT) / df['close'] - 1

    # Risk-adjusted: forward_return / (ATR / close) or just forward_return / ATR?
    # Prompt says: (12H forward return / current ATR)
    # Usually ATR is in price units, so 12H return should also be in price units for it to make sense?
    # Or return percentage / ATR percentage?
    # (price[t+12] - price[t]) / ATR[t]

    df['forward_price_diff'] = df['close'].shift(-BARRIER_TIME_LIMIT) - df['close']
    df['risk_adj_return'] = df['forward_price_diff'] / df['atr_barrier']

    return df

def apply_rolling_zscore(df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
    """
    Applies rolling z-score standardization to features.
    Standardize features using ROLLING z-scores (lookback=168).
    """
    logger.info(f"Applying rolling z-score (window={ROLLING_Z_LOOKBACK})...")

    features = [col for col in df.columns if col not in exclude_cols]

    # We use a loop for rolling z-score to avoid huge memory usage if using window.apply
    # (x - rolling_mean) / rolling_std
    for col in features:
        # Check if column is numeric
        if not np.issubdtype(df[col].dtype, np.number):
            continue

        rolling_mean = df[col].rolling(window=ROLLING_Z_LOOKBACK).mean()
        rolling_std = df[col].rolling(window=ROLLING_Z_LOOKBACK).std()

        # Handle division by zero
        df[f'{col}_z'] = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)
        df[f'{col}_z'] = df[f'{col}_z'].fillna(0) # If std was 0, z-score is 0 if at mean

        # Optionally replace original or keep both?
        # Requirement says "Standardise features", usually means transform them.
        # But for debugging, keeping both might be better.
        # I'll replace the original ones to keep feature count manageable.
        df[col] = df[f'{col}_z']
        df.drop(columns=[f'{col}_z'], inplace=True)

    return df

def remove_highly_correlated_features(df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
    """
    Removes features with correlation > threshold.
    """
    logger.info(f"Removing highly correlated features (threshold={CORRELATION_THRESHOLD})...")

    features = [col for col in df.columns if col not in exclude_cols]
    corr_matrix = df[features].corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > CORRELATION_THRESHOLD)]

    logger.info(f"Dropped {len(to_drop)} features: {to_drop}")
    df.drop(columns=to_drop, inplace=True)

    return df

if __name__ == "__main__":
    df = get_merged_data(PRIMARY_ASSET, SECONDARY_ASSET)
    df = apply_triple_barrier_labels(df)
    df = apply_continuous_target(df)
    print(df[['close', 'atr_barrier', 'triple_barrier_label', 'risk_adj_return']].tail(20))
    print("\nLabel distribution:")
    print(df['triple_barrier_label'].value_counts())
