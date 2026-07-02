import unittest

import pandas as pd

from qqq_options_system.backtest.engine import BacktestConfig, StrategySpec, run_strategy
from qqq_options_system.backtest.execution import ExecutionCostConfig


def chain(date, put_bid=1.0, put_mid=1.1, put_ask=1.2, call_bid=2.0, call_mid=2.1, call_ask=2.2):
    return pd.DataFrame(
        [
            {
                "trade_date": date,
                "underlying": "QQQ",
                "expiry": "2024-02-16",
                "dte": 30,
                "right": "P",
                "strike": 95.0,
                "contract_id": "QQQ_20240216_P_00095000",
                "bid": put_bid,
                "mid": put_mid,
                "ask": put_ask,
                "delta": -0.25,
                "underlying_price": 100.0,
            },
            {
                "trade_date": date,
                "underlying": "QQQ",
                "expiry": "2024-02-16",
                "dte": 30,
                "right": "P",
                "strike": 90.0,
                "contract_id": "QQQ_20240216_P_00090000",
                "bid": 0.4,
                "mid": 0.5,
                "ask": 0.6,
                "delta": -0.10,
                "underlying_price": 100.0,
            },
            {
                "trade_date": date,
                "underlying": "QQQ",
                "expiry": "2024-02-16",
                "dte": 30,
                "right": "C",
                "strike": 105.0,
                "contract_id": "QQQ_20240216_C_00105000",
                "bid": call_bid,
                "mid": call_mid,
                "ask": call_ask,
                "delta": 0.55,
                "underlying_price": 100.0,
            },
        ]
    )


class BacktestEngineTests(unittest.TestCase):
    def test_entry_lag_uses_next_trading_day(self):
        chains = {
            "2024-01-02": chain("2024-01-02", put_bid=9.0),
            "2024-01-03": chain("2024-01-03", put_bid=1.0),
            "2024-01-04": chain("2024-01-04", put_bid=0.4, put_mid=0.45, put_ask=0.5),
        }
        spec = StrategySpec(
            strategy="short_put",
            dte_target=30,
            hold_days=1,
            price_mode="bid_ask",
            short_put_delta=-0.25,
        )
        trades = run_strategy(chains, ["2024-01-02"], spec, entry_lag_days=1)
        self.assertEqual(trades["entry_date"].iloc[0], "2024-01-03")
        self.assertAlmostEqual(trades["gross_pnl"].iloc[0], 50.0)
        self.assertAlmostEqual(trades["pnl"].iloc[0], 48.66)

    def test_put_spread_reports_defined_risk_capital(self):
        chains = {
            "2024-01-03": chain("2024-01-03"),
            "2024-01-04": chain("2024-01-04", put_bid=0.4, put_mid=0.45, put_ask=0.5),
        }
        spec = StrategySpec(
            strategy="put_spread",
            dte_target=30,
            hold_days=1,
            price_mode="bid_ask",
            short_put_delta=-0.25,
            put_spread_width_pct=0.05,
        )
        trades = run_strategy(chains, ["2024-01-03"], spec, entry_lag_days=0)
        self.assertEqual(len(trades), 1)
        self.assertLess(trades["capital_required"].iloc[0], 500.0)

    def test_failed_no_exit_quote_is_recorded(self):
        exit_chain = chain("2024-01-04")
        exit_chain = exit_chain.loc[exit_chain["strike"] != 95.0].copy()
        chains = {
            "2024-01-03": chain("2024-01-03"),
            "2024-01-04": exit_chain,
        }
        spec = StrategySpec(
            strategy="short_put",
            dte_target=30,
            hold_days=1,
            price_mode="bid_ask",
            short_put_delta=-0.25,
        )
        trades = run_strategy(chains, ["2024-01-03"], spec, entry_lag_days=0)
        self.assertEqual(trades["status"].iloc[0], "failed_no_exit_quote")
        self.assertIn("Missing exit quote", trades["failure_reason"].iloc[0])

    def test_costs_lower_net_pnl(self):
        chains = {
            "2024-01-03": chain("2024-01-03"),
            "2024-01-04": chain("2024-01-04", put_bid=0.4, put_mid=0.45, put_ask=0.5),
        }
        spec = StrategySpec(
            strategy="short_put",
            dte_target=30,
            hold_days=1,
            price_mode="bid_ask",
            short_put_delta=-0.25,
        )
        zero_cost = BacktestConfig(
            cost_config=ExecutionCostConfig(
                commission_per_contract=0.0,
                exchange_fee_per_contract=0.0,
            )
        )
        with_cost = run_strategy(chains, ["2024-01-03"], spec, entry_lag_days=0)
        no_cost = run_strategy(chains, ["2024-01-03"], spec, entry_lag_days=0, config=zero_cost)
        self.assertLess(with_cost["pnl"].iloc[0], no_cost["pnl"].iloc[0])
        self.assertGreater(with_cost["total_costs"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
