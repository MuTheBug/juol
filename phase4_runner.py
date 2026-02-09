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
    from config import EXCLUDE_COLS

    df = dp.apply_rolling_zscore(df, EXCLUDE_COLS)
    df = dp.remove_highly_correlated_features(df, EXCLUDE_COLS)
    df = df.dropna()

    # 3. Regime Filter
    rf = RegimeFilter()
    df['regime'] = rf.fit_predict(df)

    # 4. Get OOS Signals via Walk-Forward
    # ROUND 1: Automated Feature Selection (Top 20 by Mutual Information)
    from sklearn.feature_selection import mutual_info_classif
    X_selector = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns], errors='ignore')
    y_selector = df['triple_barrier_label'].dropna()
    common_idx = X_selector.index.intersection(y_selector.index)

    # Use only trending/ranging clear labels for selector
    mask = y_selector.loc[common_idx] != 0
    mi = mutual_info_classif(X_selector.loc[common_idx][mask], y_selector.loc[common_idx][mask])
    mi_series = pd.Series(mi, index=X_selector.columns).sort_values(ascending=False)
    feature_cols = mi_series.head(20).index.tolist()
    logger.info(f"Selected top 20 features: {feature_cols}")

    logger.info(f"Generating OOS signals via Walk-Forward using {len(feature_cols)} features...")
    wfb = WalkForwardBacktester(df, 'triple_barrier_label', feature_cols)
    results_list = wfb.run(n_trials_lgb=10, n_trials_xgb=5)

    # Reconstruct OOS Series
    all_oos_signals = pd.Series(0.0, index=df.index)
    all_oos_conf = pd.Series(0.0, index=df.index)
    all_oos_regimes = pd.Series(0.0, index=df.index)

    for res in wfb.results:
        all_oos_signals.loc[res['test_indices']] = res['predictions'].values
        all_oos_conf.loc[res['test_indices']] = res['confidences'].values
        all_oos_regimes.loc[res['test_indices']] = res['regimes'].values

    df['signal'] = all_oos_signals
    df['confidence'] = all_oos_conf
    df['regime'] = all_oos_regimes

    # Filter only the period where we have OOS signals
    oos_period = all_oos_signals[all_oos_signals != 0].index
    if len(oos_period) == 0:
        logger.error("No OOS signals generated. Check model thresholds.")
        return

    backtest_period = df.loc[oos_period[0] : oos_period[-1]].copy()

    # 5. Run Event-Driven Backtester
    # FINAL ROUND: Optimized parameters
    bt = Backtester(backtest_period, initial_equity=10000.0, conf_threshold=0.63)
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

    # Save detailed results to CSV
    report['trades'].to_csv("backtest_trades.csv", index=False)
    report['equity_curve'].to_csv("backtest_equity_curve.csv")
    logger.info("Detailed backtest results saved to backtest_trades.csv and backtest_equity_curve.csv")

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
