import math
import unittest

import pandas as pd

from qqq_options_system.research.entry_signals import (
    build_ablation_masks,
    build_skew25_features,
    calibrate_training_thresholds,
    close_location_features,
    select_daily_skew25,
    standardized_shock,
)


def option_quote(expiry, dte, right, delta, strike, mid_iv, bid=1.0, mid=1.1, ask=1.2):
    return {
        "expiry": expiry,
        "dte": dte,
        "right": right,
        "delta": delta,
        "strike": strike,
        "mid_iv": mid_iv,
        "bid": bid,
        "mid": mid,
        "ask": ask,
        "open_interest": 100,
        "volume": 10,
    }


class EntrySignalFeatureTests(unittest.TestCase):
    def test_standardized_shock_uses_only_prior_twenty_returns(self):
        history = [0.01, -0.01] * 10
        returns = pd.Series([*history, -0.02, 9.0])
        result = standardized_shock(returns)
        expected_vol = pd.Series(history).std(ddof=1)
        self.assertAlmostEqual(result.loc[20, "prior_return_vol20"], expected_vol)
        self.assertAlmostEqual(
            result.loc[20, "standardized_shock"], -0.02 / expected_vol
        )

        changed = returns.copy()
        changed.iloc[20] = -0.50
        changed.iloc[21] = -99.0
        changed_result = standardized_shock(changed)
        self.assertAlmostEqual(
            changed_result.loc[20, "prior_return_vol20"],
            result.loc[20, "prior_return_vol20"],
        )

    def test_skew_selects_same_tradeable_expiry_and_nearest_deltas(self):
        chain = pd.DataFrame(
            [
                option_quote("2024-02-16", 44, "P", -0.25, 95, 0.30),
                option_quote(
                    "2024-02-16", 44, "C", 0.25, 105, 0.20, bid=1.2, mid=1.1, ask=1.0
                ),
                option_quote("2024-02-23", 47, "P", -0.10, 90, 0.25),
                option_quote("2024-02-23", 47, "P", -0.24, 95, 0.31),
                option_quote("2024-02-23", 47, "C", 0.40, 100, 0.23),
                option_quote("2024-02-23", 47, "C", 0.26, 105, 0.21),
                option_quote("2024-03-01", 54, "P", -0.25, 95, 0.50),
                option_quote("2024-03-01", 54, "C", 0.25, 105, 0.10),
            ]
        )
        result = select_daily_skew25(chain)
        self.assertEqual(result["skew_expiry"], "2024-02-23")
        self.assertEqual(result["put25_delta"], -0.24)
        self.assertEqual(result["call25_delta"], 0.26)
        self.assertAlmostEqual(result["skew25"], 0.10)

    def test_missing_nearest_expiry_iv_stays_nan_without_fallback_or_fill(self):
        missing = pd.DataFrame(
            [
                option_quote("2024-02-23", 45, "P", -0.25, 95, None),
                option_quote("2024-02-23", 45, "C", 0.25, 105, 0.20),
                option_quote("2024-03-01", 52, "P", -0.25, 95, 0.50),
                option_quote("2024-03-01", 52, "C", 0.25, 105, 0.10),
            ]
        )
        selected = select_daily_skew25(missing)
        self.assertEqual(selected["skew_expiry"], "2024-02-23")
        self.assertTrue(math.isnan(selected["skew25"]))

        valid = pd.DataFrame(
            [
                option_quote("2024-02-23", 45, "P", -0.25, 95, 0.30),
                option_quote("2024-02-23", 45, "C", 0.25, 105, 0.20),
            ]
        )
        chains = {f"2024-01-0{day}": valid for day in range(1, 6)}
        chains["2024-01-06"] = missing
        features = build_skew25_features(chains)
        self.assertTrue(pd.isna(features.loc[5, "skew25"]))
        self.assertTrue(pd.isna(features.loc[5, "skew25_5d_change"]))

    def test_close_location_and_zero_range_boundary(self):
        ohlc = pd.DataFrame(
            {
                "Date": ["2024-01-02", "2024-01-03"],
                "High": [110.0, 100.0],
                "Low": [90.0, 100.0],
                "Close": [105.0, 100.0],
            }
        )
        result = close_location_features(ohlc)
        self.assertAlmostEqual(result.loc[0, "close_location"], 0.75)
        self.assertTrue(pd.isna(result.loc[1, "close_location"]))

    def test_holdout_changes_do_not_change_training_calibration(self):
        features = pd.DataFrame(
            {
                "date": [
                    "2016-01-03",
                    "2016-01-04",
                    "2016-01-05",
                    "2016-01-06",
                    "2016-01-07",
                    "2016-01-08",
                    "2016-01-11",
                    "2018-01-04",
                    "2022-01-04",
                ],
                "gate_no_stress": [True, True, True, True, True, True, True, True, True],
                "underlying_return": [-0.20, -0.02, -0.015, -0.005, 0.0, 0.01, 0.02, 0.5, -0.9],
                "standardized_shock": [None, -2.0, -1.8, -0.5, 0.0, 0.5, 1.0, 99.0, -99.0],
                "skew25_5d_change": [-0.10, -0.01, -0.02, 0.01, 0.0, 0.0, 0.0, 9.0, -9.0],
                "close_location": [0.9, 0.7, 0.5, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0],
            }
        )
        calibration = calibrate_training_thresholds(
            features, "2007-01-01", "2016-12-31"
        )
        self.assertAlmostEqual(calibration.baseline_event_rate_q, 2.0 / 6.0)
        self.assertEqual(calibration.training_gate_days, 7)
        self.assertEqual(calibration.training_shock_observations, 6)
        self.assertAlmostEqual(calibration.close_location_threshold, 0.6)

        masks = build_ablation_masks(features, calibration)
        self.assertEqual(
            list(masks.columns),
            [
                "date",
                "baseline",
                "standardized",
                "standardized_skew_nonaccelerating",
                "standardized_skew_close_confirmation",
            ],
        )
        self.assertTrue(
            (
                masks["standardized_skew_close_confirmation"]
                <= masks["standardized_skew_nonaccelerating"]
            ).all()
        )
        self.assertTrue(
            (
                masks["standardized_skew_nonaccelerating"]
                <= masks["standardized"]
            ).all()
        )

        changed = features.copy()
        holdout = pd.to_datetime(changed["date"]).dt.year > 2016
        changed.loc[holdout, "underlying_return"] = [-100.0, 100.0]
        changed.loc[holdout, "standardized_shock"] = [-1000.0, 1000.0]
        changed.loc[holdout, "close_location"] = [0.01, 0.99]
        recalibrated = calibrate_training_thresholds(
            changed, "2007-01-01", "2016-12-31"
        )
        self.assertEqual(
            recalibrated.standardized_shock_threshold,
            calibration.standardized_shock_threshold,
        )
        self.assertEqual(
            recalibrated.close_location_threshold,
            calibration.close_location_threshold,
        )


if __name__ == "__main__":
    unittest.main()
