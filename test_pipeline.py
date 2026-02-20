import unittest
import pandas as pd
import numpy as np
import data_pipeline as dp

class TestDataPipeline(unittest.TestCase):
    def test_triple_barrier_labeling(self):
        # Create dummy data with enough rows for ATR(14)
        n = 50
        data = {
            'high': [100.0] * n,
            'low': [99.0] * n,
            'close': [99.5] * n,
        }
        df = pd.DataFrame(data)
        # Induce a hit
        df.loc[20, 'high'] = 110.0 # Upper hit for index 19 (barrier is small because ATR is small)
        df.loc[25, 'low'] = 80.0   # Lower hit for index 24

        labeled_df = dp.apply_triple_barrier_labels(df)

        # At index 19, should have a label (ATR should be valid after 14+ periods)
        self.assertFalse(np.isnan(labeled_df['triple_barrier_label'].iloc[19]))
        self.assertEqual(labeled_df['triple_barrier_label'].iloc[19], 1.0)
        self.assertEqual(labeled_df['triple_barrier_label'].iloc[24], -1.0)

if __name__ == '__main__':
    unittest.main()
