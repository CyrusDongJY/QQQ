import unittest

import pandas as pd

from qqq_options_system.research.features import (
    build_daily_features,
    rolling_percentile,
    signal_dates_for_trigger,
)


class FeatureTests(unittest.TestCase):
    def test_return_signal_uses_feature_date_only(self):
        chains = {
            "2024-01-02": pd.DataFrame(
                [{"dte": 30, "mid_iv": 0.2, "strike": 100, "underlying_price": 100.0}]
            ),
            "2024-01-03": pd.DataFrame(
                [{"dte": 30, "mid_iv": 0.21, "strike": 100, "underlying_price": 98.0}]
            ),
        }
        features = build_daily_features(chains)
        signals = signal_dates_for_trigger(features, "return", -0.01)
        self.assertEqual(signals, ["2024-01-03"])

    def test_rolling_iv_percentile_uses_only_past_and_current_values(self):
        values = pd.Series([0.10, 0.20, 0.15, 0.50])
        percentiles = rolling_percentile(values, lookback=3, min_periods=2)
        self.assertTrue(pd.isna(percentiles.iloc[0]))
        self.assertEqual(percentiles.iloc[1], 100.0)
        self.assertAlmostEqual(percentiles.iloc[2], 100.0 * 2 / 3)
        self.assertEqual(percentiles.iloc[3], 100.0)


if __name__ == "__main__":
    unittest.main()
