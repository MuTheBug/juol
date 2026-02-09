import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

import data_pipeline as dp
import features as ft
from model import RegimeFilter, LightGBMModel, XGBoostModel, EnsembleTradingModel
from walk_forward import WalkForwardBacktester
from backtester import Backtester
from robustness import run_monte_carlo, plot_monte_carlo, run_parameter_sensitivity, plot_sensitivity_heatmap
from config import PRIMARY_ASSET, SECONDARY_ASSET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_phase4():
    # 1. Load Data
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

    # 4. Get OOS Signals via Walk-Forward
    # To save time in demo, we'll use a smaller train window or fewer folds
    feature_cols = [col for col in df.columns if col not in exclude_cols + ['regime']]

    # Custom Walk-Forward that returns signals
    logger.info("Generating OOS signals via Walk-Forward...")
    # We'll re-implement a bit of walk-forward logic to collect signals across all folds
    all_oos_signals = pd.Series(0.0, index=df.index)
    all_oos_conf = pd.Series(0.0, index=df.index)

    # Just run a few folds for Phase 4 demo
    train_window = 2000
    test_window = 500
    step_size = 500
    purge_gap = 12

    start_idx = 0
    lgb_model = LightGBMModel()
    xgb_model = XGBoostModel()

    fold = 0
    max_folds = 5

    while start_idx + train_window + purge_gap + test_window <= len(df) and fold < max_folds:
        logger.info(f"Fold {fold}...")
        train_df = df.iloc[start_idx : start_idx + train_window]
        test_df = df.iloc[start_idx + train_window + purge_gap : start_idx + train_window + purge_gap + test_window]

        train_clean = train_df[train_df['triple_barrier_label'] != 0].copy()
        X_train = train_clean[feature_cols]
        y_train = train_clean['triple_barrier_label'].map({-1: 0, 1: 1})
        weights = 1.0 / train_clean['atr_barrier']
        weights = weights / weights.mean()

        # Quick fit (no optimization for demo speed)
        lgb_model.fit(X_train, y_train, sample_weight=weights)
        xgb_model.fit(X_train, y_train, sample_weight=weights)

        ensemble = EnsembleTradingModel(lgb_model, xgb_model)
        X_test = test_df[feature_cols]

        all_oos_signals.loc[test_df.index] = ensemble.predict(X_test).values
        all_oos_conf.loc[test_df.index] = ensemble.get_confidence(X_test).values

        start_idx += step_size
        fold += 1

    df['signal'] = all_oos_signals
    df['confidence'] = all_oos_conf

    # Filter only the period where we have OOS signals
    backtest_df = df[df['signal'] != 0].copy() # This might be too restrictive if we want to see flat periods
    # Actually, we want the whole period covered by OOS tests
    backtest_period = df[(df.index >= all_oos_signals[all_oos_signals != 0].index[0]) &
                         (df.index <= all_oos_signals[all_oos_signals != 0].index[-1])].copy()

    # 5. Run Event-Driven Backtester
    bt = Backtester(backtest_period, initial_equity=10000.0)
    report = bt.run()

    # 6. Metrics & Results
    print("\n" + "="*30)
    print("BACKTEST PERFORMANCE METRICS")
    print("="*30)
    for k, v in report['metrics'].items():
        if isinstance(v, float):
            print(f"{k:25}: {v:.4f}")
        else:
            print(f"{k:25}: {v}")

    # 7. Plots
    plot_backtest_results(report, backtest_period)

    # 8. Robustness Tests
    if not report['trades'].empty:
        logger.info("Running Monte Carlo...")
        mc_results = run_monte_carlo(report['trades'], 10000.0)
        plot_monte_carlo(mc_results)

        logger.info("Running Sensitivity Analysis...")
        # Reduce ranges for faster run
        sensitivity = run_parameter_sensitivity(backtest_period,
                                              sl_range=[1.0, 1.5, 2.0],
                                              tp_range=[2.0, 2.5, 3.0])
        plot_sensitivity_heatmap(sensitivity, "SL/TP Sensitivity - Total Return", "sensitivity_heatmap.png")
    else:
        logger.warning("No trades executed. Skipping robustness tests.")

def plot_backtest_results(report: Dict[str, Any], df: pd.DataFrame):
    equity_curve = report['equity_curve']

    plt.figure(figsize=(12, 8))

    # Subplot 1: Equity Curve
    ax1 = plt.subplot(2, 1, 1)
    equity_curve['equity'].plot(ax=ax1, label='Bot Equity', color='blue')

    # Benchmark: Buy & Hold
    initial_price = df['close'].iloc[0]
    benchmark = (df['close'] / initial_price) * 10000.0
    benchmark.plot(ax=ax1, label='Buy & Hold', color='gray', alpha=0.6, linestyle='--')

    plt.title("Equity Curve vs Buy & Hold")
    plt.legend()
    plt.grid(True)

    # Subplot 2: Drawdown
    ax2 = plt.subplot(2, 1, 2)
    rolling_max = equity_curve['equity'].cummax()
    drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
    drawdown.plot(ax=ax2, kind='area', color='red', alpha=0.3)
    plt.title("Drawdown Chart")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("backtest_report.png")
    plt.close()

    # Monthly Heatmap
    daily_returns = equity_curve['equity'].resample('D').last().pct_change()
    monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    if not monthly_returns.empty:
        heatmap_data = monthly_returns.to_frame()
        heatmap_data['year'] = heatmap_data.index.year
        heatmap_data['month'] = heatmap_data.index.month
        pivot_table = heatmap_data.pivot(index='year', columns='month', values='equity')

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".1%", cmap='RdYlGn', center=0)
        plt.title("Monthly Returns Heatmap")
        plt.savefig("monthly_heatmap.png")
        plt.close()

if __name__ == "__main__":
    run_phase4()
