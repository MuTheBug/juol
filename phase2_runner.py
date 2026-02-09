import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import data_pipeline as dp
import features as ft
from model import RegimeFilter
from walk_forward import WalkForwardBacktester
from config import PRIMARY_ASSET, SECONDARY_ASSET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_phase2():
    # 1. Load Data (similar to Phase 1)
    df = dp.get_merged_data(PRIMARY_ASSET, SECONDARY_ASSET)
    df = ft.add_trend_features(df)
    df = ft.add_momentum_features(df)
    df = ft.add_volatility_features(df)
    df = ft.add_volume_features(df)
    df = ft.add_microstructure_features(df)
    df = ft.add_temporal_features(df)
    df = ft.add_meta_features(df)
    df = dp.apply_triple_barrier_labels(df)
    df = dp.apply_continuous_target(df)

    # 2. Preprocessing
    df = df.ffill().dropna()
    target_cols = ['triple_barrier_label', 'forward_return_12h', 'forward_price_diff', 'risk_adj_return']
    raw_cols = ['open', 'high', 'low', 'close', 'volume', 'market_open', 'market_high', 'market_low', 'market_close', 'market_volume', 'fundingRate', 'vwap', 'atr_barrier', 'last_24h_high', 'last_24h_low']
    exclude_cols = target_cols + raw_cols

    df = dp.apply_rolling_zscore(df, exclude_cols)
    df = dp.remove_highly_correlated_features(df, exclude_cols)
    df = df.dropna()

    # 3. Regime Filter
    rf = RegimeFilter()
    df['regime'] = rf.fit_predict(df)

    # 4. Walk-Forward Backtest
    feature_cols = [col for col in df.columns if col not in exclude_cols + ['regime']]
    wfb = WalkForwardBacktester(df, 'triple_barrier_label', feature_cols)

    # For quick verification, we use very few trials
    results_df = wfb.run(n_trials_lgb=5, n_trials_xgb=3)

    # 5. Output Results
    print("\nWalk-Forward Results:")
    print(results_df)
    results_df.to_csv("walk_forward_results.csv", index=False)
    print(f"\nAverage IS Accuracy: {results_df['is_accuracy'].mean():.4f}")
    print(f"Average OOS Accuracy: {results_df['oos_accuracy'].mean():.4f}")
    print(f"Total Signals: {results_df['n_signals'].sum()}")

    # 6. IS vs OOS comparison
    # (We already have OOS results in results_df. For IS, we'd need to track it during run)
    # I'll add a simplified IS check on the last fold's training data.

    # 7. Feature Importance
    avg_importance = wfb.get_avg_importance()
    plt.figure(figsize=(12, 8))
    avg_importance.head(20).plot(kind='barh')
    plt.title("Top 20 Features (Averaged across Folds)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance_folds.png")
    logger.info("Feature importance plot saved to feature_importance_folds.png")

if __name__ == "__main__":
    run_phase2()
