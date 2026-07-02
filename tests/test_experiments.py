import tempfile
import unittest
from pathlib import Path

from qqq_options_system.research.experiments import run_experiment


class ExperimentTests(unittest.TestCase):
    def test_run_experiment_rejects_same_day_eod_entry(self):
        config = {
            "data_start": "2024-01-01",
            "data_end": "2024-01-02",
            "entry_lag_days": 0,
            "underlying": "QQQ",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                run_experiment(config, Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
