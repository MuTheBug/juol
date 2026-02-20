import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Any
from backtester import Backtester

logger = logging.getLogger(__name__)

def run_monte_carlo(trades: pd.DataFrame, initial_equity: float, iterations: int = 1000):
    """
    Randomly reshuffles trade sequence and reports outcome distribution.
    """
    if trades.empty:
        return None

    final_equities = []
    pnl_usd = trades['pnl_usd'].values

    for _ in range(iterations):
        shuffled_pnl = np.random.permutation(pnl_usd)
        final_equity = initial_equity + shuffled_pnl.sum()
        final_equities.append(final_equity)

    final_equities = np.array(final_equities)

    results = {
        'median': np.median(final_equities),
        '5th_percentile': np.quantile(final_equities, 0.05),
        '95th_percentile': np.quantile(final_equities, 0.95),
        'equities': final_equities
    }

    return results

def plot_monte_carlo(results: Dict[str, Any]):
    if not results: return

    if np.max(results['equities']) <= np.min(results['equities']):
        logger.warning("Monte Carlo results have no variance. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    try:
        plt.hist(results['equities'], bins=50, alpha=0.75, color='blue', edgecolor='black')
    except Exception as e:
        logger.warning(f"Could not plot Monte Carlo histogram: {e}")
        plt.close()
        return
    plt.axvline(results['median'], color='red', linestyle='--', label='Median')
    plt.axvline(results['5th_percentile'], color='orange', linestyle=':', label='5th Percentile')
    plt.axvline(results['95th_percentile'], color='green', linestyle=':', label='95th Percentile')
    plt.title("Monte Carlo Simulation - Final Equity Distribution")
    plt.xlabel("Final Equity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("monte_carlo.png")
    plt.close()

def run_parameter_sensitivity(df: pd.DataFrame,
                              sl_range: List[float],
                              tp_range: List[float],
                              initial_equity: float = 10000.0):
    """
    Varies SL/TP multipliers and reports final return.
    """
    results_matrix = np.zeros((len(sl_range), len(tp_range)))

    for i, sl in enumerate(sl_range):
        for j, tp in enumerate(tp_range):
            bt = Backtester(df, initial_equity=initial_equity, sl_mult=sl, tp_mult=tp)
            report = bt.run()
            results_matrix[i, j] = report['metrics'].get('Total Return', 0)

    return pd.DataFrame(results_matrix, index=sl_range, columns=tp_range)

def plot_sensitivity_heatmap(results: pd.DataFrame, title: str, filename: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, annot=True, fmt=".2%", cmap='RdYlGn')
    plt.title(title)
    plt.xlabel("TP Multiplier")
    plt.ylabel("SL Multiplier")
    plt.savefig(filename)
    plt.close()
