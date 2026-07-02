import unittest

import pandas as pd

from qqq_options_system.backtest.selection import select_by_delta_dte, select_put_wing_by_width
from qqq_options_system.data.normalize import QuoteFilterConfig


class SelectionTests(unittest.TestCase):
    def test_normalized_scoring_prefers_reasonable_delta_and_dte(self):
        chain = pd.DataFrame(
            [
                {
                    "right": "P",
                    "dte": 30,
                    "delta": -0.05,
                    "strike": 100.0,
                    "bid": 1.0,
                    "mid": 1.1,
                    "ask": 1.2,
                    "open_interest": 10,
                    "volume": 1,
                },
                {
                    "right": "P",
                    "dte": 35,
                    "delta": -0.25,
                    "strike": 95.0,
                    "bid": 1.0,
                    "mid": 1.1,
                    "ask": 1.2,
                    "open_interest": 10,
                    "volume": 1,
                },
            ]
        )
        selected = select_by_delta_dte(chain, "P", -0.25, 30)
        self.assertEqual(selected["strike"], 95.0)

    def test_selection_filters_illiquid_wing(self):
        chain = pd.DataFrame(
            [
                {
                    "right": "P",
                    "expiry": "2024-02-16",
                    "strike": 95.0,
                    "bid": 1.0,
                    "mid": 1.1,
                    "ask": 1.2,
                    "open_interest": 10,
                    "volume": 1,
                },
                {
                    "right": "P",
                    "expiry": "2024-02-16",
                    "strike": 90.0,
                    "bid": 0.1,
                    "mid": 0.2,
                    "ask": 2.0,
                    "open_interest": 10,
                    "volume": 1,
                },
                {
                    "right": "P",
                    "expiry": "2024-02-16",
                    "strike": 85.0,
                    "bid": 0.2,
                    "mid": 0.25,
                    "ask": 0.3,
                    "open_interest": 10,
                    "volume": 1,
                },
            ]
        )
        short_put = pd.Series(
            {
                "expiry": "2024-02-16",
                "strike": 95.0,
                "underlying_price": 100.0,
            }
        )
        wing = select_put_wing_by_width(
            chain,
            short_put,
            0.05,
            quote_filter_config=QuoteFilterConfig(max_relative_spread=0.5),
        )
        self.assertEqual(wing["strike"], 85.0)


if __name__ == "__main__":
    unittest.main()
