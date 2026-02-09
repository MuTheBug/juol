import pandas as pd
from typing import List
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import data_pipeline as dp
import features as ft
from config import PRIMARY_ASSET, SECONDARY_ASSET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_phase1():
    # 1. Load and Merge Data
    df = dp.get_merged_data(PRIMARY_ASSET, SECONDARY_ASSET)

    # 2. Add Features
    df = ft.add_trend_features(df)
    df = ft.add_momentum_features(df)
    df = ft.add_volatility_features(df)
    df = ft.add_volume_features(df)
    df = ft.add_microstructure_features(df)
    df = ft.add_temporal_features(df)
    df = ft.add_meta_features(df)

    # 3. Add Targets
    df = dp.apply_triple_barrier_labels(df)
    df = dp.apply_continuous_target(df)

    # 4. Handle NaNs (before z-score)
    logger.info("Handling NaNs...")
    df = df.ffill().dropna()

    # Define columns to exclude from feature processing (targets and OHLCV if desired)
    target_cols = ['triple_barrier_label', 'forward_return_12h', 'forward_price_diff', 'risk_adj_return']
    raw_cols = ['open', 'high', 'low', 'close', 'volume', 'market_open', 'market_high', 'market_low', 'market_close', 'market_volume', 'fundingRate', 'vwap', 'atr_barrier', 'last_24h_high', 'last_24h_low']
    exclude_cols = target_cols + raw_cols

    # 5. Preprocessing
    df = dp.apply_rolling_zscore(df, exclude_cols)
    df = dp.remove_highly_correlated_features(df, exclude_cols)

    # Final cleanup of NaNs introduced by z-score (at the beginning of the series)
    df = df.dropna()

    logger.info(f"Final feature count: {len(df.columns) - len(target_cols)}")
    logger.info(f"Final sample count: {len(df)}")

    # 6. Feature Importance
    analyze_importance(df, target_cols)

    # 7. Visualization
    plot_correlation(df, exclude_cols)

    return df

def analyze_importance(df: pd.DataFrame, target_cols: List[str]):
    logger.info("Analyzing feature importance...")

    # Use triple_barrier_label, discard class 0
    data = df[df['triple_barrier_label'] != 0].dropna()
    X = data.drop(columns=target_cols + ['open', 'high', 'low', 'close', 'volume', 'market_open', 'market_high', 'market_low', 'market_close', 'market_volume', 'fundingRate', 'vwap', 'atr_barrier', 'last_24h_high', 'last_24h_low'], errors='ignore')
    y = data['triple_barrier_label']

    # Mutual Information
    mi = mutual_info_classif(X, y)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    logger.info(f"Top 10 features by Mutual Information:\n{mi_series.head(10)}")

    # Permutation Importance (using a quick Random Forest)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    perm_importance = permutation_importance(rf, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    perm_series = pd.Series(perm_importance.importances_mean, index=X.columns).sort_values(ascending=False)
    logger.info(f"Top 10 features by Permutation Importance:\n{perm_series.head(10)}")

def plot_correlation(df: pd.DataFrame, exclude_cols: List[str]):
    logger.info("Generating correlation matrix...")
    features = [col for col in df.columns if col not in exclude_cols]
    plt.figure(figsize=(20, 16))
    sns.heatmap(df[features].corr(), cmap='coolwarm', annot=False)
    plt.title("Feature Correlation Matrix")
    plt.savefig("feature_correlation.png")
    logger.info("Correlation matrix saved to feature_correlation.png")

if __name__ == "__main__":
    run_phase1()
