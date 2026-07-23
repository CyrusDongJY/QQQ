import unittest
from pathlib import Path

from qqq_options_system.research.experiments import iter_strategy_specs, run_experiment


class ExperimentTests(unittest.TestCase):
    def test_run_experiment_rejects_same_day_eod_entry(self):
        config = {
            "data_start": "2024-01-01",
            "data_end": "2024-01-02",
            "entry_lag_days": 0,
            "underlying": "QQQ",
        }
        with self.assertRaises(ValueError):
            run_experiment(config, Path("unused-output-root"))

    def test_defined_risk_bullish_combo_narrow_grid_and_labels(self):
        specs = list(
            iter_strategy_specs(
                {
                    "strategies": ["defined_risk_bullish_combo"],
                    "price_modes": ["bid_ask"],
                    "dte_targets": [45],
                    "hold_days": [21],
                    "short_put_deltas": [-0.50],
                    "put_spread_width_pcts": [0.05, 0.10],
                    "call_dte_targets": [45],
                }
            )
        )
        self.assertEqual(len(specs), 2)
        self.assertEqual({spec.short_put_delta for spec in specs}, {-0.50})
        self.assertEqual({spec.put_spread_width_pct for spec in specs}, {0.05, 0.10})
        self.assertEqual({spec.call_dte_target for spec in specs}, {45})
        self.assertEqual(
            {spec.label for spec in specs},
            {
                "defined_risk_bullish_combo_dte45_hold21_bid_ask_spdm0p5_wing0p05_cdte45",
                "defined_risk_bullish_combo_dte45_hold21_bid_ask_spdm0p5_wing0p1_cdte45",
            },
        )


if __name__ == "__main__":
    unittest.main()
