import unittest

import pandas as pd

from qqq_options_system.backtest.selection import (
    select_by_delta_dte,
    select_call_within_premium_budget,
    select_put_wing_by_width,
)
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


def _call_chain():
    return pd.DataFrame(
        [
            {
                "right": "C",
                "expiry": "2024-03-15",
                "dte": 45,
                "delta": 0.60,
                "strike": 98.0,
                "bid": 4.0,
                "mid": 4.1,
                "ask": 4.2,
                "open_interest": 50,
                "volume": 5,
            },
            {
                "right": "C",
                "expiry": "2024-03-15",
                "dte": 45,
                "delta": 0.40,
                "strike": 104.0,
                "bid": 1.4,
                "mid": 1.5,
                "ask": 1.6,
                "open_interest": 50,
                "volume": 5,
            },
            {
                "right": "C",
                "expiry": "2024-03-15",
                "dte": 45,
                "delta": 0.20,
                "strike": 110.0,
                "bid": 0.4,
                "mid": 0.5,
                "ask": 0.6,
                "open_interest": 50,
                "volume": 5,
            },
        ]
    )


class CallBudgetSelectionTests(unittest.TestCase):
    def test_picks_most_expensive_affordable_call(self):
        # Budget 2.0 rules out the 4.2 ask (delta 0.60); best affordable is
        # the 1.6 ask call (delta 0.40), not the cheaper 0.6 one.
        selected = select_call_within_premium_budget(
            _call_chain(), call_dte_target=45, premium_budget=2.0
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["strike"], 104.0)
        self.assertAlmostEqual(selected["delta"], 0.40)

    def test_returns_none_when_no_call_affordable(self):
        selected = select_call_within_premium_budget(
            _call_chain(), call_dte_target=45, premium_budget=0.3
        )
        self.assertIsNone(selected)

    def test_midpoint_cost_column_used(self):
        # With a 1.5 budget on mid pricing, the 1.5-mid call is exactly
        # affordable while the 4.1-mid call is not.
        selected = select_call_within_premium_budget(
            _call_chain(),
            call_dte_target=45,
            premium_budget=1.5,
            entry_cost_column="mid",
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["strike"], 104.0)

    def test_required_expiry_excludes_adjacent_expiration(self):
        chain = _call_chain()
        adjacent = chain.iloc[0].copy()
        adjacent["expiry"] = "2024-03-22"
        adjacent["dte"] = 52
        adjacent["ask"] = 1.9
        adjacent["mid"] = 1.8
        adjacent["bid"] = 1.7
        chain = pd.concat([chain, adjacent.to_frame().T], ignore_index=True)
        selected = select_call_within_premium_budget(
            chain,
            call_dte_target=52,
            premium_budget=2.0,
            required_expiry="2024-03-15",
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["expiry"], "2024-03-15")
        self.assertEqual(selected["strike"], 104.0)


if __name__ == "__main__":
    unittest.main()
