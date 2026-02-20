import unittest
import pandas as pd
import numpy as np
from model import RegimeFilter, LightGBMModel, EnsembleTradingModel

class TestModels(unittest.TestCase):
    def test_regime_filter(self):
        df = pd.DataFrame({
            'atr': np.random.rand(200),
            'ADX_14': np.random.rand(200),
            'btc_corr_24h': np.random.rand(200),
            'volume_sma_ratio': np.random.rand(200),
            'fundingRate': np.random.rand(200)
        })
        rf = RegimeFilter(n_components=3)
        regimes = rf.fit_predict(df)
        self.assertEqual(len(regimes), 200)
        self.assertTrue(regimes.nunique() <= 3)

    def test_ensemble_agreement(self):
        class MockModel:
            def __init__(self, probas, classes):
                self.probas = probas
                self.classes_ = classes
            def predict_proba(self, X):
                return self.probas

        # Proba: [neg, pos]
        # Label 0 means neg (-1), 1 means pos (1)
        lgb_probas = np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
        xgb_probas = np.array([[0.7, 0.3], [0.3, 0.7], [0.5, 0.5]])

        lgb = LightGBMModel()
        lgb.model = MockModel(lgb_probas, np.array([0, 1]))

        from model import XGBoostModel
        xgb_mod = XGBoostModel()
        xgb_mod.model = MockModel(xgb_probas, np.array([0, 1]))

        ensemble = EnsembleTradingModel(lgb, xgb_mod, confidence_threshold=0.6)
        preds = ensemble.predict(pd.DataFrame(np.zeros((3, 1))))

        # 1st sample: both agree neg, confidence (0.8+0.7)/2 = 0.75 > 0.6. Result: -1
        # 2nd sample: both agree pos, confidence (0.8+0.7)/2 = 0.75 > 0.6. Result: 1
        # 3rd sample: both agree neutral/low confidence. Result: 0

        self.assertEqual(preds.iloc[0], -1)
        self.assertEqual(preds.iloc[1], 1)
        self.assertEqual(preds.iloc[2], 0)

if __name__ == '__main__':
    unittest.main()
