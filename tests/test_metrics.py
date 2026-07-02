import unittest

import pandas as pd

from qqq_options_system.backtest.metrics import summarize_trades


class MetricsTests(unittest.TestCase):
    def test_risk_free_rate_lowers_sharpe(self):
        trades = pd.DataFrame(
            [
                {
                    "status": "closed",
                    "exit_date": "2024-01-02",
                    "pnl": 10.0,
                    "gross_pnl": 10.0,
                    "total_costs": 0.0,
                    "return_on_capital": 0.01,
                    "capital_required": 1000.0,
                },
                {
                    "status": "closed",
                    "exit_date": "2024-01-03",
                    "pnl": 8.0,
                    "gross_pnl": 8.0,
                    "total_costs": 0.0,
                    "return_on_capital": 0.008,
                    "capital_required": 1000.0,
                },
                {
                    "status": "closed",
                    "exit_date": "2024-01-04",
                    "pnl": -4.0,
                    "gross_pnl": -4.0,
                    "total_costs": 0.0,
                    "return_on_capital": -0.004,
                    "capital_required": 1000.0,
                },
            ]
        )
        no_rf = summarize_trades(trades, risk_free_rate=0.0)
        with_rf = summarize_trades(trades, risk_free_rate=0.05)
        self.assertLess(with_rf["sharpe"], no_rf["sharpe"])


if __name__ == "__main__":
    unittest.main()
