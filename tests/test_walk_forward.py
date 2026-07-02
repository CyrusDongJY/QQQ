import unittest

import pandas as pd

from qqq_options_system.research.walk_forward import (
    rank_walk_forward_candidates,
    validate_walk_forward_config,
)


class WalkForwardTests(unittest.TestCase):
    def base_config(self):
        return {
            "grid_search_scope": "training_only",
            "selection_metric": "sharpe",
            "training_period": {"start": "2007-01-01", "end": "2016-12-31"},
            "validation_period": {"start": "2017-01-01", "end": "2020-12-31"},
            "test_period": {"start": "2021-01-01", "end": "2024-12-31"},
            "entry_lag_days": 1,
        }

    def test_ranking_uses_training_metric_only(self):
        candidates = pd.DataFrame(
            [
                {
                    "strategy_id": "train_best_holdout_bad",
                    "training_sharpe": 2.0,
                    "validation_sharpe": -5.0,
                    "test_sharpe": -5.0,
                    "training_closed_trades": 30,
                },
                {
                    "strategy_id": "train_second_holdout_great",
                    "training_sharpe": 1.0,
                    "validation_sharpe": 99.0,
                    "test_sharpe": 99.0,
                    "training_closed_trades": 30,
                },
            ]
        )
        ranked = rank_walk_forward_candidates(
            candidates,
            selection_metric="sharpe",
            top_n=2,
            min_training_trades=1,
        )
        self.assertEqual(ranked.iloc[0]["strategy_id"], "train_best_holdout_bad")
        self.assertEqual(ranked.iloc[1]["strategy_id"], "train_second_holdout_great")

    def test_validation_test_performance_cannot_improve_rank(self):
        candidates = pd.DataFrame(
            [
                {
                    "strategy_id": "poor_holdout_but_training_top",
                    "training_capital_return_sum": 5.0,
                    "validation_capital_return_sum": -100.0,
                    "test_capital_return_sum": -100.0,
                    "training_closed_trades": 50,
                },
                {
                    "strategy_id": "excellent_holdout_training_second",
                    "training_capital_return_sum": 4.0,
                    "validation_capital_return_sum": 1000.0,
                    "test_capital_return_sum": 1000.0,
                    "training_closed_trades": 50,
                },
            ]
        )
        ranked = rank_walk_forward_candidates(
            candidates,
            selection_metric="capital_return_sum",
            top_n=2,
            min_training_trades=1,
        )
        self.assertEqual(ranked.iloc[0]["strategy_id"], "poor_holdout_but_training_top")

    def test_candidate_output_has_period_columns(self):
        candidates = pd.DataFrame(
            [
                {
                    "strategy_id": "candidate",
                    "training_sharpe": 1.2,
                    "validation_sharpe": 0.8,
                    "test_sharpe": 0.6,
                    "training_total_pnl": 100.0,
                    "validation_total_pnl": 50.0,
                    "test_total_pnl": 25.0,
                    "training_closed_trades": 20,
                }
            ]
        )
        ranked = rank_walk_forward_candidates(candidates, top_n=1, min_training_trades=1)
        for column in (
            "training_sharpe",
            "validation_sharpe",
            "test_sharpe",
            "training_total_pnl",
            "validation_total_pnl",
            "test_total_pnl",
        ):
            self.assertIn(column, ranked.columns)

    def test_validate_rejects_unknown_scope(self):
        config = self.base_config()
        config["grid_search_scope"] = "all_periods"
        with self.assertRaises(ValueError):
            validate_walk_forward_config(config)

    def test_validate_rejects_missing_periods(self):
        config = self.base_config()
        del config["test_period"]
        with self.assertRaises(ValueError):
            validate_walk_forward_config(config)


if __name__ == "__main__":
    unittest.main()
