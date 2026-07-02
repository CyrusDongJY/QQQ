import unittest
import math

from qqq_options_system.backtest.execution import fill_price


class ExecutionTests(unittest.TestCase):
    def test_bid_ask_execution(self):
        self.assertEqual(fill_price(1.0, 1.5, 2.0, "buy", "bid_ask"), 2.0)
        self.assertEqual(fill_price(1.0, 1.5, 2.0, "sell", "bid_ask"), 1.0)

    def test_price_sensitivity_modes(self):
        self.assertEqual(fill_price(1.0, 1.5, 2.0, "buy", "midpoint"), 1.5)
        self.assertEqual(fill_price(1.0, 1.5, 2.0, "sell", "midpoint"), 1.5)
        self.assertEqual(fill_price(1.0, 1.5, 2.0, "buy", "half_spread"), 1.75)
        self.assertEqual(fill_price(1.0, 1.5, 2.0, "sell", "half_spread"), 1.25)

    def test_invalid_quotes_raise(self):
        invalid_quotes = [
            (math.nan, 1.5, 2.0),
            (1.0, math.nan, 2.0),
            (1.0, 1.5, math.inf),
            (-1.0, 1.5, 2.0),
            (2.0, 1.5, 1.0),
            (0.0, 0.0, 0.0),
        ]
        for bid, mid, ask in invalid_quotes:
            with self.subTest(bid=bid, mid=mid, ask=ask):
                with self.assertRaises(ValueError):
                    fill_price(bid, mid, ask, "buy")


if __name__ == "__main__":
    unittest.main()
