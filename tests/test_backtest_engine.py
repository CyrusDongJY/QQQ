import ast
import unittest

import pandas as pd

from qqq_options_system.backtest.engine import BacktestConfig, StrategySpec, run_strategy
from qqq_options_system.backtest.execution import ExecutionCostConfig
from qqq_options_system.data.normalize import QuoteFilterConfig


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


def rr_chain(
    date,
    put_bid=3.0,
    put_mid=3.1,
    put_ask=3.2,
    call100_bid=2.8,
    call100_mid=2.9,
    call100_ask=3.0,
):
    """Chain for risk reversal: ATM put (45 DTE) plus calls at 45 and 60 DTE."""
    return pd.DataFrame(
        [
            {
                "trade_date": date,
                "underlying": "QQQ",
                "expiry": "2024-02-17",
                "dte": 45,
                "right": "P",
                "strike": 100.0,
                "contract_id": "QQQ_20240217_P_00100000",
                "bid": put_bid,
                "mid": put_mid,
                "ask": put_ask,
                "delta": -0.50,
                "underlying_price": 100.0,
                "open_interest": 100,
                "volume": 10,
            },
            {
                "trade_date": date,
                "underlying": "QQQ",
                "expiry": "2024-02-17",
                "dte": 45,
                "right": "C",
                "strike": 100.0,
                "contract_id": "QQQ_20240217_C_00100000",
                "bid": call100_bid,
                "mid": call100_mid,
                "ask": call100_ask,
                "delta": 0.52,
                "underlying_price": 100.0,
                "open_interest": 100,
                "volume": 10,
            },
            {
                "trade_date": date,
                "underlying": "QQQ",
                "expiry": "2024-03-03",
                "dte": 60,
                "right": "C",
                "strike": 105.0,
                "contract_id": "QQQ_20240303_C_00105000",
                "bid": 2.0,
                "mid": 2.1,
                "ask": 2.2,
                "delta": 0.45,
                "underlying_price": 100.0,
                "open_interest": 100,
                "volume": 10,
            },
        ]
    )


def combo_chain(
    date,
    short_bid=5.0,
    short_mid=5.1,
    short_ask=5.2,
    wing_bid=0.8,
    wing_mid=0.9,
    wing_ask=1.0,
    selected_call_bid=3.6,
    selected_call_mid=3.7,
    selected_call_ask=3.8,
):
    common = {
        "trade_date": date,
        "underlying": "QQQ",
        "expiry": "2024-03-15",
        "dte": 45,
        "underlying_price": 100.0,
        "open_interest": 100,
        "volume": 10,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "right": "P",
                "strike": 100.0,
                "contract_id": "QQQ_20240315_P_00100000",
                "bid": short_bid,
                "mid": short_mid,
                "ask": short_ask,
                "delta": -0.50,
            },
            {
                **common,
                "right": "P",
                "strike": 95.0,
                "contract_id": "QQQ_20240315_P_00095000",
                "bid": wing_bid,
                "mid": wing_mid,
                "ask": wing_ask,
                "delta": -0.25,
            },
            {
                **common,
                "right": "C",
                "strike": 100.0,
                "contract_id": "QQQ_20240315_C_00100000",
                "bid": 4.3,
                "mid": 4.4,
                "ask": 4.5,
                "delta": 0.52,
            },
            {
                **common,
                "right": "C",
                "strike": 105.0,
                "contract_id": "QQQ_20240315_C_00105000",
                "bid": selected_call_bid,
                "mid": selected_call_mid,
                "ask": selected_call_ask,
                "delta": 0.40,
            },
            {
                **common,
                "right": "C",
                "strike": 110.0,
                "contract_id": "QQQ_20240315_C_00110000",
                "bid": 1.8,
                "mid": 1.9,
                "ask": 2.0,
                "delta": 0.25,
            },
        ]
    )


class BacktestEngineTests(unittest.TestCase):
    def test_explicit_market_calendar_does_not_jump_over_missing_entry_chain(self):
        chains = {
            "2024-01-02": chain("2024-01-02"),
            "2024-01-04": chain("2024-01-04"),
        }
        spec = StrategySpec(
            strategy="short_put",
            dte_target=30,
            hold_days=1,
            price_mode="bid_ask",
            short_put_delta=-0.25,
        )
        trades = run_strategy(
            chains,
            ["2024-01-02"],
            spec,
            entry_lag_days=1,
            calendar_dates=["2024-01-02", "2024-01-03", "2024-01-04"],
        )
        self.assertEqual(trades["entry_date"].iloc[0], "2024-01-03")
        self.assertEqual(trades["status"].iloc[0], "failed_missing_entry_chain")

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

    def test_risk_reversal_builds_two_legs_and_combines_pnl(self):
        # Entry (bid_ask): sell 100 put @bid 3.0, buy 100 call @ask 3.0.
        # Exit on a rally: buy back short put @ask 1.2, sell long call @bid 5.0.
        # Short put leg: +3.0 - 1.2 = +1.8 (*100 = 180)
        # Long call leg: -3.0 + 5.0 = +2.0 (*100 = 200)
        # Combined gross = 380.
        exit_chain = rr_chain(
            "2024-01-04",
            put_bid=1.0,
            put_mid=1.1,
            put_ask=1.2,
            call100_bid=5.0,
            call100_mid=5.1,
            call100_ask=5.2,
        )
        chains = {
            "2024-01-03": rr_chain("2024-01-03"),
            "2024-01-04": exit_chain,
        }
        spec = StrategySpec(
            strategy="risk_reversal",
            dte_target=45,
            hold_days=1,
            price_mode="bid_ask",
            short_put_delta=-0.50,
            call_dte_target=45,
        )
        zero_cost = BacktestConfig(
            cost_config=ExecutionCostConfig(
                commission_per_contract=0.0,
                exchange_fee_per_contract=0.0,
            )
        )
        trades = run_strategy(chains, ["2024-01-03"], spec, entry_lag_days=0, config=zero_cost)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades["status"].iloc[0], "closed")
        self.assertEqual(trades["legs"].iloc[0], 2)
        self.assertAlmostEqual(trades["gross_pnl"].iloc[0], 380.0)
        # Capital driven by the short put margin (100 strike * 100 * 1.0).
        self.assertAlmostEqual(trades["capital_required"].iloc[0], 10000.0)

    def test_risk_reversal_records_failure_when_no_affordable_call(self):
        # Put credit only 0.4; even the cheapest call (ask 2.2) is unaffordable,
        # so no combo is opened for this signal.
        entry = rr_chain("2024-01-03", put_bid=0.4, put_mid=0.45, put_ask=0.5)
        chains = {
            "2024-01-03": entry,
            "2024-01-04": rr_chain("2024-01-04"),
        }
        spec = StrategySpec(
            strategy="risk_reversal",
            dte_target=45,
            hold_days=1,
            price_mode="bid_ask",
            short_put_delta=-0.50,
            call_dte_target=45,
        )
        trades = run_strategy(chains, ["2024-01-03"], spec, entry_lag_days=0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades["status"].iloc[0], "failed_no_entry_selection")
        self.assertIn("premium-budget", trades["failure_reason"].iloc[0])

    def test_dataset_boundary_does_not_force_early_exit(self):
        dates = pd.bdate_range("2024-01-02", periods=10).strftime("%Y-%m-%d").tolist()
        chains = {date: chain(date) for date in dates}
        spec = StrategySpec(
            strategy="short_put",
            dte_target=30,
            hold_days=21,
            price_mode="bid_ask",
            short_put_delta=-0.25,
        )
        trades = run_strategy(
            chains,
            [dates[0]],
            spec,
            entry_lag_days=0,
            calendar_dates=dates,
        )
        self.assertEqual(trades["status"].iloc[0], "failed_no_valid_exit_date")

    def test_defined_risk_bullish_combo_three_legs_budget_pnl_and_t_plus_one(self):
        dates = pd.bdate_range("2024-01-02", periods=23).strftime("%Y-%m-%d").tolist()
        chains = {date: combo_chain(date) for date in dates}
        chains[dates[-1]] = combo_chain(
            dates[-1],
            short_bid=1.8,
            short_mid=1.9,
            short_ask=2.0,
            wing_bid=0.4,
            wing_mid=0.5,
            wing_ask=0.6,
            selected_call_bid=6.0,
            selected_call_mid=6.1,
            selected_call_ask=6.2,
        )
        spec = StrategySpec(
            strategy="defined_risk_bullish_combo",
            dte_target=45,
            hold_days=21,
            price_mode="bid_ask",
            short_put_delta=-0.50,
            put_spread_width_pct=0.05,
            call_dte_target=45,
        )
        trades = run_strategy(chains, [dates[0]], spec, entry_lag_days=1)
        trade = trades.iloc[0]
        self.assertEqual(trade["entry_date"], dates[1])
        self.assertEqual(trade["exit_date"], dates[-1])
        self.assertEqual(trade["legs"], 3)
        legs = ast.literal_eval(trade["leg_details"])
        short_put = next(leg for leg in legs if leg["right"] == "P" and leg["qty"] < 0)
        long_put = next(leg for leg in legs if leg["right"] == "P" and leg["qty"] > 0)
        long_call = next(leg for leg in legs if leg["right"] == "C")
        put_spread_credit = short_put["entry_price"] - long_put["entry_price"]
        self.assertEqual(long_call["entry_strike"], 105.0)
        self.assertLessEqual(long_call["entry_price"], put_spread_credit)
        self.assertAlmostEqual(trade["gross_pnl"], 460.0)
        self.assertAlmostEqual(trade["total_costs"], 4.02)
        self.assertAlmostEqual(trade["pnl"], 455.98)
        self.assertAlmostEqual(trade["capital_required"], 484.02)
        self.assertAlmostEqual(trade["max_loss"], 484.02)
        self.assertAlmostEqual(trade["return_on_capital"], 455.98 / 484.02)
        self.assertEqual(
            trade["capital_model"],
            "defined_risk_width_minus_residual_credit_plus_round_trip_costs",
        )

    def test_defined_risk_bullish_combo_records_no_affordable_call_without_mid(self):
        entry = combo_chain(
            "2024-01-03",
            short_bid=1.0,
            short_mid=4.0,
            short_ask=4.2,
            wing_bid=0.6,
            wing_mid=0.7,
            wing_ask=0.8,
        )
        chains = {
            "2024-01-03": entry,
            "2024-01-04": combo_chain("2024-01-04"),
        }
        spec = StrategySpec(
            strategy="defined_risk_bullish_combo",
            dte_target=45,
            hold_days=1,
            price_mode="bid_ask",
            short_put_delta=-0.50,
            put_spread_width_pct=0.05,
            call_dte_target=45,
        )
        cfg = BacktestConfig(
            quote_filter_config=QuoteFilterConfig(max_relative_spread=None)
        )
        trades = run_strategy(
            chains, ["2024-01-03"], spec, entry_lag_days=0, config=cfg
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["status"], "failed_no_affordable_call")
        self.assertIn("put-spread credit 0.2", trades.iloc[0]["failure_reason"])

    def test_defined_risk_bullish_combo_records_nonpositive_credit(self):
        entry = combo_chain(
            "2024-01-03",
            short_bid=0.8,
            short_mid=0.9,
            short_ask=1.0,
            wing_bid=0.8,
            wing_mid=0.9,
            wing_ask=1.0,
        )
        spec = StrategySpec(
            strategy="defined_risk_bullish_combo",
            dte_target=45,
            hold_days=1,
            price_mode="bid_ask",
            short_put_delta=-0.50,
            put_spread_width_pct=0.05,
            call_dte_target=45,
        )
        trades = run_strategy(
            {"2024-01-03": entry, "2024-01-04": combo_chain("2024-01-04")},
            ["2024-01-03"],
            spec,
            entry_lag_days=0,
        )
        self.assertEqual(trades.iloc[0]["status"], "failed_nonpositive_put_spread_credit")


if __name__ == "__main__":
    unittest.main()
