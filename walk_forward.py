import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from model import LightGBMModel, XGBoostModel, EnsembleTradingModel, RegimeFilter
from config import TRAIN_WINDOW, TEST_WINDOW, STEP_SIZE, PURGE_GAP
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

class WalkForwardBacktester:
    def __init__(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]):
        self.df = df
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.results = []
        self.importances = []

    def run(self, n_trials_lgb=50, n_trials_xgb=30):
        logger.info("Starting Walk-Forward Backtest...")

        n_samples = len(self.df)
        start_idx = 0
        fold = 0

        lgb_model = LightGBMModel()
        xgb_model = XGBoostModel()
        regime_filter = RegimeFilter()

        # Limit folds for Phase 2 demo if necessary, but here we'll try to run a decent amount
        max_folds = 20 # Limit for verification

        while start_idx + TRAIN_WINDOW + PURGE_GAP + TEST_WINDOW <= n_samples and fold < max_folds:
            logger.info(f"Processing Fold {fold}...")

            # Windows
            train_start = start_idx
            train_end = start_idx + TRAIN_WINDOW
            test_start = train_end + PURGE_GAP
            test_end = test_start + TEST_WINDOW

            # Data split
            train_df = self.df.iloc[train_start:train_end]
            test_df = self.df.iloc[test_start:test_end]

            # Fit regime filter on training data ONLY
            regime_filter.fit(train_df)
            test_regimes = regime_filter.predict(test_df)

            # Discard class 0 (neutral) from training as per Phase 2 instructions
            train_clean = train_df[train_df[self.target_col] != 0].copy()

            X_train = train_clean[self.feature_cols]
            # Map labels for XGBoost/LightGBM: -1 -> 0, 1 -> 1
            y_train = train_clean[self.target_col].map({-1: 0, 1: 1})

            # Sample weights: inverse ATR
            # We need raw ATR, not z-scored for this weighting if possible,
            # or just use the ATR barrier column.
            # phase1_runner excludes 'atr_barrier' from z-scoring so it should be raw.
            weights = 1.0 / train_clean['atr_barrier']
            weights = weights / weights.mean() # Normalize

            # Re-optimize every 4th fold
            if fold % 4 == 0:
                lgb_model.optimize(X_train, y_train, sample_weight=weights, n_trials=n_trials_lgb)
                xgb_model.optimize(X_train, y_train, sample_weight=weights, n_trials=n_trials_xgb)
            else:
                lgb_model.fit(X_train, y_train, sample_weight=weights)
                xgb_model.fit(X_train, y_train, sample_weight=weights)

            # Ensemble
            ensemble = EnsembleTradingModel(lgb_model, xgb_model)

            # Inference on test set (including class 0 for realistic backtest)
            X_test = test_df[self.feature_cols]
            # Map y_test same way for scoring if we exclude 0s,
            # but ensemble.predict returns -1, 0, 1.
            y_test = test_df[self.target_col]

            predictions = ensemble.predict(X_test)
            confidence = ensemble.get_confidence(X_test)

            # In-Sample (IS) performance
            is_preds = ensemble.predict(X_train)
            is_valid = is_preds != 0
            # Map y_train back to -1, 1 for consistent comparison
            y_train_orig = y_train.map({0: -1, 1: 1})
            is_acc = accuracy_score(y_train_orig[is_valid], is_preds[is_valid]) if is_valid.any() else 0

            # Metrics (only on signals where prediction != 0)
            valid_idx = predictions != 0
            if valid_idx.any():
                acc = accuracy_score(y_test[valid_idx], predictions[valid_idx])
                prec = precision_score(y_test[valid_idx], predictions[valid_idx], average='weighted', zero_division=0)
            else:
                acc, prec = 0, 0

            fold_results = {
                'fold': fold,
                'test_start': test_df.index[0],
                'test_end': test_df.index[-1],
                'is_accuracy': is_acc,
                'oos_accuracy': acc,
                'precision': prec,
                'n_signals': valid_idx.sum(),
                'test_indices': test_df.index,
                'predictions': predictions,
                'confidences': confidence,
                'regimes': test_regimes
            }
            self.results.append(fold_results)

            # Store feature importance (LightGBM)
            importance = pd.Series(lgb_model.model.feature_importances_, index=self.feature_cols)
            self.importances.append(importance)

            logger.info(f"Fold {fold} Accuracy: {acc:.4f}, Signals: {valid_idx.sum()}")

            # Step forward
            start_idx += STEP_SIZE
            fold += 1

        return pd.DataFrame(self.results)

    def get_avg_importance(self):
        return pd.concat(self.importances, axis=1).mean(axis=1).sort_values(ascending=False)
