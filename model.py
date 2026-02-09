import pandas as pd
import numpy as np
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import optuna
from typing import Tuple, Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class RegimeFilter:
    """
    Classifies market regimes using Gaussian Mixture Model.
    Regimes: TRENDING, RANGING, VOLATILE/CHAOTIC
    """
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.scaler = StandardScaler()
        self.regime_map = {0: 'TRENDING', 1: 'RANGING', 2: 'VOLATILE'} # Initial guess, will be refined

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Fits GMM and returns regime labels.
        Features: ATR percentile, ADX, BTC correlation, volume ratio, funding rate
        """
        logger.info("Fitting Regime Filter (GMM)...")

        # Prepare features for GMM
        features = pd.DataFrame(index=df.index)
        features['atr_pct'] = df['atr'].rolling(168).rank(pct=True)
        features['adx'] = df['ADX_14']
        features['btc_corr'] = df['btc_corr_24h']
        features['vol_ratio'] = df['volume_sma_ratio']
        features['funding'] = df['fundingRate'].abs()

        features = features.ffill().dropna()

        scaled_features = self.scaler.fit_transform(features)
        regimes = self.gmm.fit_predict(scaled_features)

        # Map regimes based on characteristics
        # TRENDING: High ADX
        # RANGING: Low ADX, Low ATR
        # VOLATILE: High ATR, High Vol ratio

        # For simplicity in Phase 2, we return the cluster labels and log stats
        regime_series = pd.Series(regimes, index=features.index)

        for i in range(self.n_components):
            cluster_data = features[regime_series == i]
            logger.info(f"Regime {i} mean ADX: {cluster_data['adx'].mean():.2f}, mean ATR Pct: {cluster_data['atr_pct'].mean():.2f}")

        return regime_series.reindex(df.index).ffill()

class BaseTradingModel:
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.params = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

class LightGBMModel(BaseTradingModel):
    def __init__(self):
        super().__init__("LightGBM")

    def optimize(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None, n_trials: int = 100):
        logger.info(f"Optimizing {self.name}...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'verbosity': -1,
                'random_state': 42
            }

            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import f1_score
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
                y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
                w_t = sample_weight.iloc[train_idx] if sample_weight is not None else None

                clf = lgb.LGBMClassifier(**params)
                clf.fit(X_t, y_t, sample_weight=w_t)
                preds = clf.predict(X_v)
                scores.append(f1_score(y_v, preds, average='weighted'))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        self.params = study.best_params
        logger.info(f"Best params for {self.name}: {self.params}")
        self.model = lgb.LGBMClassifier(**self.params, verbosity=-1)
        self.model.fit(X, y, sample_weight=sample_weight)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        if not self.params:
            # Default params if not optimized
            self.model = lgb.LGBMClassifier(verbosity=-1)
        else:
            self.model = lgb.LGBMClassifier(**self.params, verbosity=-1)
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

class XGBoostModel(BaseTradingModel):
    def __init__(self):
        super().__init__("XGBoost")

    def optimize(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None, n_trials: int = 50):
        logger.info(f"Optimizing {self.name}...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42
            }

            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import f1_score
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
                y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
                w_t = sample_weight.iloc[train_idx] if sample_weight is not None else None

                clf = xgb.XGBClassifier(**params)
                clf.fit(X_t, y_t, sample_weight=w_t)
                preds = clf.predict(X_v)
                scores.append(f1_score(y_v, preds, average='weighted'))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        self.params = study.best_params
        logger.info(f"Best params for {self.name}: {self.params}")
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        if not self.params:
            self.model = xgb.XGBClassifier()
        else:
            self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

class EnsembleTradingModel:
    def __init__(self, lgb_model: LightGBMModel, xgb_model: XGBoostModel, confidence_threshold: float = 0.58):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.confidence_threshold = confidence_threshold

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Only trade when BOTH models agree on direction.
        Confidence = average of both models' predicted probabilities.
        """
        lgb_probs = self.lgb_model.predict_proba(X)
        xgb_probs = self.xgb_model.predict_proba(X)

        # Proba for class 1 (+1) is at index 1
        # Proba for class 0 (-1) is at index 0 (assuming labels were mapped to 0, 1)
        # We need to handle mapping if classes are -1, 1

        classes = self.lgb_model.model.classes_
        # Ensure mapping: 0 corresponds to -1 signal, 1 corresponds to +1 signal
        idx_neg = np.where(classes == 0)[0][0]
        idx_pos = np.where(classes == 1)[0][0]

        lgb_neg = lgb_probs[:, idx_neg]
        lgb_pos = lgb_probs[:, idx_pos]
        xgb_neg = xgb_probs[:, idx_neg]
        xgb_pos = xgb_probs[:, idx_pos]

        # Average probabilities
        avg_neg = (lgb_neg + xgb_neg) / 2
        avg_pos = (lgb_pos + xgb_pos) / 2

        signals = np.zeros(len(X))

        # Agreement and confidence check
        lgb_dir = np.where(lgb_pos > lgb_neg, 1, -1)
        xgb_dir = np.where(xgb_pos > xgb_neg, 1, -1)

        agreement = (lgb_dir == xgb_dir)

        long_signal = agreement & (lgb_dir == 1) & (avg_pos >= self.confidence_threshold)
        short_signal = agreement & (lgb_dir == -1) & (avg_neg >= self.confidence_threshold)

        signals[long_signal] = 1
        signals[short_signal] = -1

        return pd.Series(signals, index=X.index)

    def get_confidence(self, X: pd.DataFrame) -> pd.Series:
        lgb_probs = self.lgb_model.predict_proba(X)
        xgb_probs = self.xgb_model.predict_proba(X)

        classes = self.lgb_model.model.classes_
        idx_neg = np.where(classes == 0)[0][0]
        idx_pos = np.where(classes == 1)[0][0]

        avg_neg = (lgb_probs[:, idx_neg] + xgb_probs[:, idx_neg]) / 2
        avg_pos = (lgb_probs[:, idx_pos] + xgb_probs[:, idx_pos]) / 2

        return pd.Series(np.maximum(avg_neg, avg_pos), index=X.index)
