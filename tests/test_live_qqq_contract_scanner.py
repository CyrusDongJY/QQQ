import math
import unittest
from datetime import datetime, timezone

from qqq_options_system.live.qqq_contract_scanner import (
    LiveOptionQuote,
    build_downtrend_combo_candidate,
    build_uptrend_funded_call_candidate,
    build_wall_debit_spread_candidate,
    format_contract_scan_section,
)


QUOTE_TIME = datetime(2026, 7, 22, 19, 35, tzinfo=timezone.utc)
EXPIRY = "2026-09-04"


def quote(right, strike, bid, ask, delta, expiry=EXPIRY, oi=100, volume=20):
    return LiveOptionQuote(
        underlying="QQQ",
        expiry=expiry,
        right=right,
        strike=strike,
        bid=bid,
        ask=ask,
        delta=delta,
        iv=0.25,
        volume=volume,
        open_interest=oi,
        market_data_type=1,
    )


class LiveQQQContractScannerTests(unittest.TestCase):
    def test_quote_quality_rejects_nan_crossed_and_illiquid_quotes(self):
        self.assertFalse(quote("P", 100, math.nan, 2.0, -0.4).is_tradeable())
        self.assertFalse(quote("P", 100, 2.1, 2.0, -0.4).is_tradeable())
        self.assertFalse(quote("P", 100, 1.0, 2.0, -0.4).is_tradeable())
        self.assertFalse(quote("P", 100, 1.9, 2.0, -0.4, oi=9).is_tradeable())
        self.assertTrue(quote("P", 100, 1.9, 2.0, -0.4, oi=10).is_tradeable())

    def test_uptrend_selects_delta_put_and_most_expensive_affordable_same_expiry_call(self):
        puts = [
            quote("P", 98, 4.0, 4.2, -0.30),
            quote("P", 100, 5.0, 5.2, -0.40),
        ]
        calls = [
            quote("C", 104, 5.0, 5.10, 0.55),
            quote("C", 105, 4.7, 4.90, 0.50),
            quote("C", 106, 4.5, 4.80, 0.45),
            quote("C", 105, 4.8, 4.95, 0.50, expiry="2026-09-11"),
        ]
        result = build_uptrend_funded_call_candidate(
            puts, calls, 100.0, QUOTE_TIME
        )
        self.assertTrue(result.succeeded)
        self.assertEqual(result.legs[0].quote.strike, 100)
        self.assertEqual(result.legs[0].fill_price, 5.0)
        self.assertEqual(result.legs[1].quote.strike, 105)
        self.assertEqual(result.legs[1].quote.expiry, EXPIRY)
        self.assertEqual(result.legs[1].fill_price, 4.9)
        self.assertAlmostEqual(result.net_entry_credit, 0.1)
        self.assertAlmostEqual(result.max_loss, 9991.34)

    def test_uptrend_records_failure_when_no_call_fits_executable_budget(self):
        result = build_uptrend_funded_call_candidate(
            [quote("P", 100, 5.0, 5.2, -0.40)],
            [quote("C", 100, 5.1, 5.3, 0.55)],
            100.0,
            QUOTE_TIME,
        )
        self.assertFalse(result.succeeded)
        self.assertIn("ask不超过", result.failure_reason)

    def test_downtrend_three_legs_use_bid_ask_budget_and_defined_max_loss(self):
        puts = [
            quote("P", 100, 8.0, 8.2, -0.50),
            quote("P", 90, 1.8, 2.0, -0.15),
            quote("P", 89, 1.5, 1.7, -0.12),
        ]
        calls = [
            quote("C", 101, 5.7, 5.9, 0.48),
            quote("C", 102, 5.5, 5.8, 0.44),
        ]
        result = build_downtrend_combo_candidate(
            puts, calls, 100.0, QUOTE_TIME
        )
        self.assertTrue(result.succeeded)
        self.assertEqual([leg.quote.strike for leg in result.legs], [100, 90, 101])
        self.assertEqual([leg.fill_price for leg in result.legs], [8.0, 2.0, 5.9])
        self.assertLessEqual(result.legs[2].quote.ask, 8.0 - 2.0)
        self.assertAlmostEqual(result.net_entry_credit, 0.1)
        self.assertAlmostEqual(result.max_loss, 992.01)
        self.assertAlmostEqual(result.capital_required, result.max_loss)

    def test_wall_call_spread_uses_wall_and_outer_strikes(self):
        calls = [
            quote("C", 105, 2.8, 3.0, 0.50, expiry="2026-08-21"),
            quote("C", 110, 1.0, 1.2, 0.30, expiry="2026-08-21"),
            quote("C", 111, 0.8, 1.0, 0.25, expiry="2026-08-21"),
        ]
        result = build_wall_debit_spread_candidate(
            calls,
            spot=100.0,
            quote_time=QUOTE_TIME,
            expiry="2026-08-21",
            right="C",
            wall_strike=105.0,
        )
        self.assertTrue(result.succeeded)
        self.assertEqual([leg.quote.strike for leg in result.legs], [105, 110])
        self.assertAlmostEqual(result.net_entry_credit, -2.0)
        self.assertAlmostEqual(result.max_loss, 201.34)

    def test_formatter_lists_contract_quote_liquidity_and_risk(self):
        result = build_uptrend_funded_call_candidate(
            [quote("P", 100, 5.0, 5.2, -0.40)],
            [quote("C", 105, 4.7, 4.9, 0.50)],
            100.0,
            QUOTE_TIME,
        )
        text = format_contract_scan_section([result])
        self.assertIn("QQQ 2026-09-04 $100P", text)
        self.assertIn("bid $5.00 / ask $5.20", text)
        self.assertIn("OI 100", text)
        self.assertIn("组合净信用", text)
        self.assertIn("估算最大损失", text)
        self.assertIn("不创建或发送订单", text)


if __name__ == "__main__":
    unittest.main()
