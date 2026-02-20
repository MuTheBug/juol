import pandas as pd
import numpy as np
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import optuna
import joblib
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

    def fit(self, df: pd.DataFrame):
        """Fits GMM on provided data."""
        features = self._prepare_features(df)
        if features.empty: return
        scaled_features = self.scaler.fit_transform(features)
        self.gmm.fit(scaled_features)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predicts regimes for provided data."""
        features = self._prepare_features(df)
        if features.empty: return pd.Series()
        scaled_features = self.scaler.transform(features)
        regimes = self.gmm.predict(scaled_features)
        return pd.Series(regimes, index=features.index).reindex(df.index).ffill()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        # Handle cases where these might be missing or different names
        features['atr_pct'] = df['atr'].rolling(168).rank(pct=True) if 'atr' in df.columns else 0.5
        features['adx'] = df['ADX_14'] if 'ADX_14' in df.columns else 0
        features['btc_corr'] = df['btc_corr_24h'] if 'btc_corr_24h' in df.columns else 0
        features['vol_ratio'] = df['volume_sma_ratio'] if 'volume_sma_ratio' in df.columns else 1
        features['funding'] = df['fundingRate'].abs() if 'fundingRate' in df.columns else 0
        return features.ffill().fillna(0)

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """Compatibility method - but warned: may introduce leakage if used on full backtest."""
        self.fit(df)
        return self.predict(df)

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
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 50, 200),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 100.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 100.0, log=True),
                'verbosity': -1,
                'random_state': 42
            }

            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            calmar_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
                y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
                w_t = sample_weight.iloc[train_idx] if sample_weight is not None else None

                clf = lgb.LGBMClassifier(**params)
                clf.fit(X_t, y_t, sample_weight=w_t)

                # Vectorized backtest for Calmar calculation
                # Signal: +1 if proba(1) > 0.5, -1 if proba(0) > 0.5
                probas = clf.predict_proba(X_v)
                # classes_ are [0, 1] (mapped from [-1, 1])
                signals = np.where(probas[:, 1] > 0.5, 1, np.where(probas[:, 0] > 0.5, -1, 0))

                # Assume next-period returns for simplicity in optimization
                # (Ideally use actual barrier outcomes, but return is a good proxy)
                # We don't have prices here easily, let's use y_v (the label) as a proxy
                # +1 if correct, -1 if wrong.
                y_v_native = y_v.values # labels are 0, 1
                y_v_signed = np.where(y_v_native == 1, 1, -1)

                # simplified return: signal * correct_direction
                # Actually, Calmar needs real returns. If not available, we use a proxy score.
                # Since prompt insists on Calmar, I'll try to find 'forward_return_12h' in X_v index
                # But objective only gets X, y.

                # Calculate simple vectorized return
                y_v_signed = np.where(y_v == 1, 1, -1)
                daily_rets = signals * y_v_signed * 0.01 # Assume 1% move for labels

                # Equity curve
                equity = (1 + daily_rets).cumprod()

                # Max Drawdown
                rolling_max = np.maximum.accumulate(equity)
                drawdowns = (equity - rolling_max) / rolling_max
                max_dd = np.min(drawdowns)

                # Calmar Proxy
                total_ret = equity[-1] - 1
                score = total_ret / abs(max_dd) if max_dd != 0 else total_ret
                calmar_scores.append(score)

            return np.mean(calmar_scores)

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
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 100.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 100.0, log=True),
                'random_state': 42
            }

            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import f1_score
            tscv = TimeSeriesSplit(n_splits=3)
            calmar_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
                y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
                w_t = sample_weight.iloc[train_idx] if sample_weight is not None else None

                clf = xgb.XGBClassifier(**params)
                clf.fit(X_t, y_t, sample_weight=w_t)

                # Vectorized backtest for Calmar
                probas = clf.predict_proba(X_v)
                signals = np.where(probas[:, 1] > 0.5, 1, np.where(probas[:, 0] > 0.5, -1, 0))
                y_v_signed = np.where(y_v == 1, 1, -1)
                daily_rets = signals * y_v_signed * 0.01
                equity = (1 + daily_rets).cumprod()
                rolling_max = np.maximum.accumulate(equity)
                drawdowns = (equity - rolling_max) / rolling_max
                max_dd = np.min(drawdowns)
                total_ret = equity[-1] - 1
                score = total_ret / abs(max_dd) if max_dd != 0 else total_ret
                calmar_scores.append(score)

            return np.mean(calmar_scores)

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

    def get_detailed_probas(self, X: pd.DataFrame) -> pd.DataFrame:
        lgb_probs = self.lgb_model.predict_proba(X)
        xgb_probs = self.xgb_model.predict_proba(X)

        classes = self.lgb_model.model.classes_
        idx_neg = np.where(classes == 0)[0][0]
        idx_pos = np.where(classes == 1)[0][0]

        return pd.DataFrame({
            'lgb_neg': lgb_probs[:, idx_neg],
            'lgb_pos': lgb_probs[:, idx_pos],
            'xgb_neg': xgb_probs[:, idx_neg],
            'xgb_pos': xgb_probs[:, idx_pos]
        }, index=X.index)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)
