import unittest

import pandas as pd

from qqq_options_system.research.regimes import (
    build_regime_features,
    classify_regime,
    constant_maturity_atm_iv,
    vix_bucket,
)


def option_chain(spot=100.0, iv20=0.20, iv40=0.40, iv90=0.30):
    rows = []
    for dte, iv in ((20, iv20), (40, iv40), (90, iv90)):
        for right in ("C", "P"):
            rows.append(
                {
                    "dte": dte,
                    "strike": spot,
                    "mid_iv": iv,
                    "underlying_price": spot,
                    "right": right,
                    "bid": 1.0,
                    "mid": 1.1,
                    "ask": 1.2,
                    "delta": 0.5 if right == "C" else -0.5,
                    "open_interest": 10,
                    "volume": 1,
                }
            )
    return pd.DataFrame(rows)


class RegimeFeatureTests(unittest.TestCase):
    def test_external_close_keeps_calendar_continuous_when_chain_is_missing(self):
        dates = pd.date_range("2024-01-02", periods=12, freq="B")
        close = pd.DataFrame({"date": dates, "close": range(100, 112)})
        chains = {
            date.strftime("%Y-%m-%d"): option_chain(spot=float(price))
            for date, price in zip(dates, range(100, 112))
            if date != dates[5]
        }
        vix = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "vix": 15.0,
                "vxn": 18.0,
            }
        )
        result = build_regime_features(
            chains,
            vix,
            {"ma20_min_periods": 2, "ma50_min_periods": 2},
            underlying_close=close,
        )
        self.assertEqual(len(result), len(close))
        missing_day = result.loc[
            result["date"] == dates[5].strftime("%Y-%m-%d")
        ].iloc[0]
        self.assertEqual(missing_day["qqq_close"], 105.0)
        self.assertTrue(pd.isna(missing_day["iv30"]))

    def test_vix_bucket_boundaries(self):
        self.assertEqual(vix_bucket(14.99), "<15")
        self.assertEqual(vix_bucket(15.0), "[15,20)")
        self.assertEqual(vix_bucket(20.0), "[20,25)")
        self.assertEqual(vix_bucket(25.0), "[25,30)")
        self.assertEqual(vix_bucket(30.0), ">=30")

    def test_iv_interpolation_uses_same_day_chain_only(self):
        day1 = option_chain(iv20=0.20, iv40=0.40)
        day2 = option_chain(iv20=0.90, iv40=0.90)
        self.assertAlmostEqual(constant_maturity_atm_iv(day1, 30), 0.30)
        self.assertAlmostEqual(constant_maturity_atm_iv(day2, 30), 0.90)

    def test_rolling_features_do_not_use_future_values(self):
        chains = {
            "2024-01-01": option_chain(spot=100.0, iv20=0.20, iv40=0.30),
            "2024-01-02": option_chain(spot=110.0, iv20=0.80, iv40=0.90),
            "2024-01-03": option_chain(spot=90.0, iv20=0.10, iv40=0.20),
        }
        vix = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "vix": [10.0, 50.0, 12.0],
                "vxn": [20.0, 25.0, 20.0],
            }
        )
        features = build_regime_features(
            chains,
            vix,
            {
                "iv_percentile_lookback": 2,
                "iv_percentile_min_periods": 1,
                "vix_percentile_lookback": 2,
                "vix_percentile_min_periods": 1,
                "rv_min_periods": 1,
                "rv63_min_periods": 1,
                "ma20_min_periods": 1,
                "ma50_min_periods": 1,
                "ma200_min_periods": 1,
            },
        )
        self.assertEqual(features.loc[0, "vix_percentile"], 100.0)
        self.assertEqual(features.loc[1, "vix_percentile"], 100.0)
        self.assertEqual(features.loc[2, "vix_percentile"], 50.0)

    def test_vxn_pct_change_does_not_forward_fill_missing_values(self):
        dates = pd.date_range("2024-01-01", periods=7, freq="D").strftime("%Y-%m-%d")
        chains = {date: option_chain(spot=100.0 + index) for index, date in enumerate(dates)}
        vix = pd.DataFrame(
            {
                "date": dates,
                "vix": [15.0] * 7,
                "vxn": [20.0, None, 21.0, 22.0, 23.0, 24.0, 25.0],
            }
        )
        features = build_regime_features(
            chains,
            vix,
            {
                "iv_percentile_min_periods": 1,
                "vix_percentile_min_periods": 1,
                "rv_min_periods": 1,
                "rv63_min_periods": 1,
                "ma20_min_periods": 1,
                "ma50_min_periods": 1,
                "ma200_min_periods": 1,
            },
        )
        self.assertTrue(pd.isna(features.loc[6, "vxn_5d_pct_change"]))

    def test_state_priority_is_mutually_exclusive(self):
        row = pd.Series(
            {
                "is_high_vix": True,
                "is_vxn_rising": True,
                "is_term_inverted": True,
                "is_iv_elevated": True,
                "is_downtrend": True,
                "is_vxn_falling": False,
            }
        )
        regime, _ = classify_regime(row)
        self.assertEqual(regime, "high_rising")

    def test_normalized_requires_term_and_rv_normalization(self):
        base = {
            "is_vxn_rising": False,
            "is_term_normal": True,
            "is_rv_normalizing": True,
            "is_iv_rv_positive_normal": True,
            "is_trend_recovered": True,
        }
        self.assertEqual(classify_regime(pd.Series(base))[0], "normalized")
        base["is_term_normal"] = False
        self.assertEqual(classify_regime(pd.Series(base))[0], "transition_no_trade")


if __name__ == "__main__":
    unittest.main()
