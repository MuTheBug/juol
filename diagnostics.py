import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score

import data_pipeline as dp
import features as ft
from model import RegimeFilter, LightGBMModel, XGBoostModel, EnsembleTradingModel
from walk_forward import WalkForwardBacktester
from backtester import Backtester
from config import PRIMARY_ASSET, SECONDARY_ASSET, EXCLUDE_COLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_diagnostics():
    logger.info("Running Strategy Diagnostics...")

    # 1. Load Data & Generate Signals
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
    df = df.ffill().dropna()

    df = dp.apply_rolling_zscore(df, EXCLUDE_COLS)
    df = dp.remove_highly_correlated_features(df, EXCLUDE_COLS)
    df = df.dropna()

    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]

    # 2. Run Walk-Forward to get OOS predictions
    wfb = WalkForwardBacktester(df, 'triple_barrier_label', feature_cols)
    wfb.run(n_trials_lgb=1, n_trials_xgb=1) # Minimal trials for baseline

    # 3. Aggregate Data for Diagnostics
    all_preds = []
    all_actuals = []
    all_confidences = []
    all_detailed_probas = []
    all_regimes = []
    all_X_test = []

    for res in wfb.results:
        all_preds.append(res['predictions'])
        all_actuals.append(res['actuals'])
        all_confidences.append(res['confidences'])
        all_detailed_probas.append(res['detailed_probas'])
        all_regimes.append(res['regimes'])
        all_X_test.append(res['X_test'])

    oos_preds = pd.concat(all_preds)
    oos_actuals = pd.concat(all_actuals)
    oos_conf = pd.concat(all_confidences)
    oos_probas = pd.concat(all_detailed_probas)
    oos_regimes = pd.concat(all_regimes)
    oos_X = pd.concat(all_X_test)

    df['signal'] = oos_preds
    df['confidence'] = oos_conf
    df['regime'] = oos_regimes

    # 4. Run Backtester on OOS period
    backtest_period = df.loc[oos_preds.index].copy()
    bt = Backtester(backtest_period)
    report = bt.run()
    trades = report['trades']

    # --- DIAGNOSTIC 1: Signal Quality Audit ---
    diag_signal_quality(oos_preds, oos_actuals, oos_probas, trades)

    # --- DIAGNOSTIC 2: Stop Loss Analysis ---
    diag_stop_loss(trades, backtest_period)

    # --- DIAGNOSTIC 3: Entry Timing ---
    diag_entry_timing(trades)

    # --- DIAGNOSTIC 4: Regime Analysis ---
    diag_regime(trades)

    # --- DIAGNOSTIC 5: Feature Importance ---
    diag_feature_importance(wfb, trades, oos_X)

    # --- DIAGNOSTIC 6: Walk-Forward Fold Analysis ---
    diag_fold_analysis(wfb)

    # --- DIAGNOSTIC 7: Position Sizing ---
    diag_position_sizing(trades, report['equity_curve'])

def diag_signal_quality(preds, actuals, probas, trades):
    print("\n--- DIAGNOSTIC 1: SIGNAL QUALITY AUDIT ---")

    # Filter only samples where the model actually made a prediction (!= 0)
    valid_mask = preds != 0
    if not valid_mask.any():
        print("No signals generated.")
        return

    v_preds = preds[valid_mask]
    v_actuals = actuals[valid_mask]
    v_probas = probas.loc[valid_mask]

    # TBM outcomes are -1, 0, 1. We mapped ML labels to 0 (-1) and 1 (1).
    # Correct if pred == actual.
    # Note: actuals 0 (timeout) are treated as wrong.
    acc = accuracy_score(v_actuals, v_preds)
    print(f"Model Accuracy vs Actual TBM Labels: {acc:.2%}")
    print(f"Edge vs Random (50%): {acc - 0.5:.2%}")

    # Agreement
    # Model agreed if lgb_dir == xgb_dir
    lgb_dir = np.where(v_probas['lgb_pos'] > v_probas['lgb_neg'], 1, -1)
    xgb_dir = np.where(v_probas['xgb_pos'] > v_probas['xgb_neg'], 1, -1)
    agreement_rate = (lgb_dir == xgb_dir).mean()
    print(f"Model Directional Agreement: {agreement_rate:.2%}")

    # Probability Distribution Plot
    if not trades.empty:
        plt.figure(figsize=(10, 6))
        winners = trades[trades['pnl_usd'] > 0]['confidence']
        losers = trades[trades['pnl_usd'] <= 0]['confidence']
        sns.kdeplot(winners, label='Winners', fill=True)
        sns.kdeplot(losers, label='Losers', fill=True)
        plt.title("Confidence Distribution: Winners vs Losers")
        plt.legend()
        plt.savefig("diag1_prob_dist.png")
        plt.close()

def diag_stop_loss(trades, df):
    print("\n--- DIAGNOSTIC 2: STOP LOSS ANALYSIS ---")
    if trades.empty: return

    sl_trades = trades[trades['reason'].str.contains('Stop Loss', na=False)].copy()
    if sl_trades.empty:
        print("No stop loss exits found.")
        return

    print(f"Stop Loss Exit Rate: {len(sl_trades)/len(trades):.2%}")

    saved_rates = {}
    for mult in [1.5, 2.0, 2.5, 3.0]:
        saved_count = 0
        for _, t in sl_trades.iterrows():
            # Check if it would have hit TP before a wider SL
            # This is complex to calculate exactly without rerunning backtest
            # but we can check MFE. If MFE > wider TP? No, wider stop saves the current trade.
            # If MAE < wider stop, it would have stayed alive.
            if abs(t['mae']) < mult * (t['entry_atr'] / t['entry_price']):
                # If it stayed alive, did it hit TP?
                if t['mfe'] >= 2.5 * (t['entry_atr'] / t['entry_price']):
                    saved_count += 1
        saved_rates[mult] = saved_count / len(sl_trades)

    print("Percentage of stopped-out trades that would have hit TP with wider stops:")
    for m, r in saved_rates.items():
        print(f"{m}x ATR Stop: {r:.2%}")

def diag_entry_timing(trades):
    print("\n--- DIAGNOSTIC 3: ENTRY TIMING ANALYSIS ---")
    if trades.empty: return

    plt.figure(figsize=(10, 8))
    plt.scatter(trades['mae'], trades['mfe'], alpha=0.5, c=(trades['pnl_usd'] > 0), cmap='RdYlGn')
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.xlabel("MAE (Max Adverse Excursion)")
    plt.ylabel("MFE (Max Favorable Excursion)")
    plt.title("MAE vs MFE Scatter Plot")
    plt.savefig("diag3_mae_mfe.png")
    plt.close()

    avg_init_dd = trades['initial_drawdown'].mean()
    print(f"Average immediate drawdown (first 3 candles): {avg_init_dd:.4%}")

def diag_regime(trades):
    print("\n--- DIAGNOSTIC 4: REGIME ANALYSIS ---")
    if trades.empty: return

    # 0: TRENDING, 1: RANGING, 2: VOLATILE
    regime_map = {0.0: 'TRENDING', 1.0: 'RANGING', 2.0: 'VOLATILE'}
    stats = []
    for r_val, r_name in regime_map.items():
        r_trades = trades[trades['regime'] == r_val]
        if r_trades.empty: continue

        wr = len(r_trades[r_trades['pnl_usd'] > 0]) / len(r_trades)
        pf = r_trades[r_trades['pnl_usd'] > 0]['pnl_usd'].sum() / abs(r_trades[r_trades['pnl_usd'] <= 0]['pnl_usd'].sum())
        stats.append({'Regime': r_name, 'Trades': len(r_trades), 'Win Rate': wr, 'Profit Factor': pf})

    print(pd.DataFrame(stats))

def diag_feature_importance(wfb, trades, X):
    print("\n--- DIAGNOSTIC 5: FEATURE IMPORTANCE REALITY CHECK ---")
    avg_imp = wfb.get_avg_importance()
    top_10 = avg_imp.head(10)
    print("Top 10 features by average fold importance:")
    print(top_10)

    if trades.empty: return

    # Show distribution for top feature
    top_feat = top_10.index[0]
    # We need to map trades back to X index
    plt.figure(figsize=(10, 6))
    # This is a bit tricky since X has duplicates if folds overlap, but let's take unique timestamps
    # Actually trades have entry_timestamp
    trade_indices = trades['entry_timestamp']
    winners = X.loc[trades[trades['pnl_usd'] > 0]['entry_timestamp'], top_feat]
    losers = X.loc[trades[trades['pnl_usd'] <= 0]['entry_timestamp'], top_feat]

    sns.kdeplot(winners, label='Winning Trades', fill=True)
    sns.kdeplot(losers, label='Losing Trades', fill=True)
    plt.title(f"Value Distribution for Top Feature: {top_feat}")
    plt.legend()
    plt.savefig("diag5_top_feat_dist.png")
    plt.close()

def diag_fold_analysis(wfb):
    print("\n--- DIAGNOSTIC 6: WALK-FORWARD FOLD ANALYSIS ---")
    results = pd.DataFrame(wfb.results).drop(columns=['predictions', 'confidences', 'detailed_probas', 'regimes', 'actuals', 'X_test'], errors='ignore')
    print(results)

    avg_gap = (results['is_accuracy'] - results['oos_accuracy']).mean()
    print(f"Average IS-OOS Accuracy Gap: {avg_gap:.2%}")
    if avg_gap > 0.15:
        print("CRITICAL: Severe Overfitting Detected!")

def diag_position_sizing(trades, equity_curve):
    print("\n--- DIAGNOSTIC 7: POSITION SIZING ANOMALY ---")
    if trades.empty: return

    # trades['notional_size'] is not currently in backtest_trades,
    # wait I added it in the fix to backtester but didn't run it yet.
    # Ah, I added it to the trade_record.

    # I'll check consistency of USD PnL vs PnL %
    # pnl_usd = notional * pnl_pct
    # so notional = pnl_usd / pnl_pct
    notionals = trades['pnl_usd'] / trades['pnl_pct']
    print(f"Position Size Stats: Mean: ${notionals.mean():.2f}, Std: ${notionals.std():.2f}, Max: ${notionals.max():.2f}, Min: ${notionals.min():.2f}")

    if notionals.std() > notionals.mean():
        print("WARNING: Position sizing is wildly inconsistent!")

if __name__ == "__main__":
    run_diagnostics()
