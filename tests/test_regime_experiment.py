import unittest

import pandas as pd

from qqq_options_system.backtest.engine import StrategySpec, run_strategy
from qqq_options_system.research.regime_experiment import (
    build_conditional_trades,
    iter_regime_strategy_specs,
    portfolio_summary,
    simulate_minimal_portfolio,
    walk_forward_regime_candidates,
)


def chain(date, put_bid=1.0, put_mid=1.1, put_ask=1.2):
    return pd.DataFrame(
        [
            {
                "trade_date": date,
                "underlying": "QQQ",
                "expiry": "2024-06-21",
                "dte": 60,
                "right": "P",
                "strike": 95.0,
                "contract_id": "QQQ_20240621_P_00095000",
                "bid": put_bid,
                "mid": put_mid,
                "ask": put_ask,
                "delta": -0.25,
                "underlying_price": 100.0,
                "mid_iv": 0.2,
                "open_interest": 10,
                "volume": 1,
            },
            {
                "trade_date": date,
                "underlying": "QQQ",
                "expiry": "2024-06-21",
                "dte": 60,
                "right": "P",
                "strike": 90.0,
                "contract_id": "QQQ_20240621_P_00090000",
                "bid": 0.4,
                "mid": 0.5,
                "ask": 0.6,
                "delta": -0.10,
                "underlying_price": 100.0,
                "mid_iv": 0.2,
                "open_interest": 10,
                "volume": 1,
            },
        ]
    )


def candidate_trade(
    strategy_id,
    signal_date,
    pnl,
    *,
    price_mode="bid_ask",
    non_overlap_eligible=True,
):
    return {
        "regime": "normalized",
        "strategy": "put_spread",
        "strategy_id": strategy_id,
        "signal_date": signal_date,
        "entry_date": signal_date,
        "exit_date": signal_date,
        "price_mode": price_mode,
        "non_overlap_eligible": non_overlap_eligible,
        "status": "closed",
        "pnl": float(pnl),
        "gross_pnl": float(pnl),
        "total_costs": 0.0,
        "return_on_capital": float(pnl) / 100.0,
        "capital_required": 100.0,
        "dte_target": 30,
        "hold_days": 5,
        "short_put_delta": -0.25,
        "put_spread_width_pct": 0.05,
    }


class RegimeExperimentTests(unittest.TestCase):
    def test_t_signal_enters_t_plus_one(self):
        regimes = pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "regime": "transition_no_trade",
                    "vix_bucket": "<15",
                    "vix_percentile_bucket": "0-20",
                    "allow_new_short_gamma": False,
                }
            ]
        )
        chains = {
            "2024-01-02": chain("2024-01-02"),
            "2024-01-03": chain("2024-01-03"),
            "2024-01-04": chain("2024-01-04"),
        }
        trades = build_conditional_trades(
            chains,
            regimes,
            {
                "entry_lag_days": 1,
                "baseline_hold_days": [1],
                "long_option_dtes": [],
                "price_modes": ["bid_ask"],
            },
        )
        self.assertEqual(trades.iloc[0]["entry_date"], "2024-01-03")

    def test_high_rising_and_vol_rising_do_not_open_short_premium(self):
        for regime in ("high_rising", "vol_rising"):
            specs = list(iter_regime_strategy_specs({"price_modes": ["bid_ask"]}, regime))
            self.assertNotIn("put_spread", {spec.strategy for spec in specs})

    def test_vix_only_and_joint_state_fields_are_separate(self):
        regimes = pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "regime": "low_calm",
                    "vix_bucket": "<15",
                    "vix_percentile_bucket": "0-20",
                    "allow_new_short_gamma": False,
                }
            ]
        )
        trades = build_conditional_trades(
            {"2024-01-02": chain("2024-01-02"), "2024-01-03": chain("2024-01-03")},
            regimes,
            {
                "entry_lag_days": 1,
                "baseline_hold_days": [0],
                "long_option_dtes": [],
                "price_modes": ["bid_ask"],
            },
        )
        self.assertIn("vix_bucket", trades.columns)
        self.assertIn("regime", trades.columns)

    def test_long_put_and_spread_pnl_and_max_loss(self):
        chains = {
            "2024-01-02": chain("2024-01-02"),
            "2024-01-03": chain("2024-01-03", put_bid=1.5, put_mid=1.6, put_ask=1.7),
        }
        long_put = StrategySpec(
            strategy="long_put",
            dte_target=60,
            hold_days=1,
            price_mode="bid_ask",
            long_put_delta=-0.25,
        )
        trades = run_strategy(chains, ["2024-01-02"], long_put, entry_lag_days=0)
        self.assertGreater(trades.iloc[0]["max_loss"], 0)
        self.assertAlmostEqual(trades.iloc[0]["net_pnl"], trades.iloc[0]["pnl"])

        spread = StrategySpec(
            strategy="long_put_spread",
            dte_target=60,
            hold_days=1,
            price_mode="bid_ask",
            long_put_delta=-0.25,
            put_spread_width_pct=0.05,
        )
        spread_trades = run_strategy(chains, ["2024-01-02"], spread, entry_lag_days=0)
        self.assertGreater(spread_trades.iloc[0]["capital_required"], 0)
        self.assertAlmostEqual(
            spread_trades.iloc[0]["pnl_per_1000_risk"],
            spread_trades.iloc[0]["return_on_risk"] * 1000,
        )

    def test_holdout_changes_cannot_change_training_rank(self):
        rows = [
            candidate_trade("training_best", "2010-01-04", 10.0),
            candidate_trade("training_best", "2010-01-05", 9.0),
            candidate_trade("training_second", "2010-01-04", 8.0),
            candidate_trade("training_second", "2010-01-05", 7.0),
            candidate_trade("training_best", "2018-01-04", -100.0),
            candidate_trade("training_second", "2018-01-04", 1000.0),
            candidate_trade("training_best", "2022-01-04", -100.0),
            candidate_trade("training_second", "2022-01-04", 1000.0),
            candidate_trade("training_second", "2010-01-06", 10000.0, price_mode="midpoint"),
            candidate_trade("training_second", "2010-01-07", 10000.0, non_overlap_eligible=False),
        ]
        config = {
            "selection_metric": "total_pnl",
            "top_n_candidates": 2,
            "min_training_trades": 2,
            "periods": {
                "training": {"start": "2007-01-01", "end": "2016-12-31"},
                "validation": {"start": "2017-01-01", "end": "2020-12-31"},
                "test": {"start": "2021-01-01", "end": "2024-12-31"},
            },
        }
        first = walk_forward_regime_candidates(pd.DataFrame(rows), config)
        self.assertEqual(first["strategy_id"].tolist(), ["training_best", "training_second"])
        self.assertEqual(first.iloc[0]["training_total_pnl"], 19.0)
        self.assertIn("validation_total_pnl", first.columns)
        self.assertIn("test_total_pnl", first.columns)

        changed = pd.DataFrame(rows)
        holdout = pd.to_datetime(changed["signal_date"]).dt.year >= 2017
        changed.loc[holdout & (changed["strategy_id"] == "training_best"), "pnl"] = 1_000_000.0
        changed.loc[holdout & (changed["strategy_id"] == "training_second"), "pnl"] = -1_000_000.0
        changed.loc[holdout, "return_on_capital"] = changed.loc[holdout, "pnl"] / 100.0
        second = walk_forward_regime_candidates(changed, config)
        self.assertEqual(second["strategy_id"].tolist(), first["strategy_id"].tolist())
        self.assertEqual(second["training_total_pnl"].tolist(), first["training_total_pnl"].tolist())

    def test_portfolio_outputs_declare_equity_allocation_only_scope(self):
        regimes = pd.DataFrame(
            [
                {"date": "2024-01-02", "qqq_close": 100.0, "vix": 14.0, "regime": "low_calm"},
                {"date": "2024-01-03", "qqq_close": 101.0, "vix": 16.0, "regime": "normalized"},
            ]
        )
        daily = simulate_minimal_portfolio(regimes, {"initial_nav": 100000.0})
        metrics = portfolio_summary(daily)
        self.assertEqual(set(daily["portfolio_scope"]), {"equity_allocation_only"})
        self.assertEqual(set(metrics["portfolio_scope"]), {"equity_allocation_only"})

    def test_portfolio_target_enters_on_next_close_without_earning_prior_return(self):
        regimes = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                "qqq_close": [100.0, 110.0, 121.0, 133.1],
                "vix": [30.0, 10.0, 10.0, 10.0],
                "regime": ["high_rising", "normalized", "normalized", "normalized"],
            }
        )
        daily = simulate_minimal_portfolio(
            regimes,
            {
                "initial_nav": 100000.0,
                "entry_lag_days": 1,
                "portfolio_targets": {"high_rising": 0.5, "normalized": 1.0},
            },
        )

        self.assertEqual(daily["joint_effective_target"].tolist(), [1.0, 1.0, 0.5, 1.0])
        self.assertEqual(set(daily["target_effective_lag_rows"]), {2})
        self.assertAlmostEqual(daily.loc[1, "joint_nav"], 110000.0)
        self.assertAlmostEqual(daily.loc[2, "joint_nav"], 115500.0)

    def test_walk_forward_allows_pilot_with_no_training_trades(self):
        trades = pd.DataFrame([candidate_trade("pilot", "2024-01-04", 10.0)])
        candidates = walk_forward_regime_candidates(
            trades,
            {
                "selection_metric": "sharpe",
                "min_training_trades": 1,
                "periods": {
                    "training": {"start": "2007-01-01", "end": "2016-12-31"},
                    "validation": {"start": "2017-01-01", "end": "2020-12-31"},
                    "test": {"start": "2021-01-01", "end": "2024-12-31"},
                },
            },
        )
        self.assertTrue(candidates.empty)
        self.assertIn("training_sharpe", candidates.columns)


if __name__ == "__main__":
    unittest.main()
