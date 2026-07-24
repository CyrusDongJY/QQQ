import datetime
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
CLOUD_DIR = PROJECT_DIR / "cloud_integration"
if not CLOUD_DIR.exists():
    # Development workspace keeps cloud_integration beside qqq_options_system;
    # the GitHub distribution keeps it inside the repository root.
    CLOUD_DIR = PROJECT_DIR.parent / "cloud_integration"


def load_cloud_builder_module():
    fake_yfinance = types.ModuleType("yfinance")
    fake_calendar_module = types.ModuleType("pandas_market_calendars")
    fake_calendar_module.get_calendar = lambda name: None
    fake_ib = types.ModuleType("ib_insync")
    fake_supabase = types.ModuleType("supabase")
    fake_supabase.create_client = lambda *args, **kwargs: None
    fake_supabase.Client = object
    fake_config = types.ModuleType("market_config")

    module_path = CLOUD_DIR / "multi_tenor_builder.py"
    spec = importlib.util.spec_from_file_location(
        "cloud_multi_tenor_builder_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "yfinance": fake_yfinance,
        "pandas_market_calendars": fake_calendar_module,
        "ib_insync": fake_ib,
        "supabase": fake_supabase,
        "market_config": fake_config,
    }
    sys.path.insert(0, str(CLOUD_DIR))
    try:
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(CLOUD_DIR))
    return module


class FakeCalendar:
    def __init__(self, sessions, closes=None):
        self.sessions = [pd.Timestamp(value).date() for value in sessions]
        self.closes = closes or {}

    def schedule(self, start_date, end_date):
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()
        selected = [day for day in self.sessions if start <= day <= end]
        close_values = [
            self.closes.get(
                day,
                pd.Timestamp(f"{day.isoformat()} 21:00:00", tz="UTC"),
            )
            for day in selected
        ]
        return pd.DataFrame(
            {"market_close": close_values},
            index=pd.DatetimeIndex(selected),
        )


class CloudSignalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_cloud_builder_module()
        cls.builder = cls.module.UnifiedLedgerBuilder.__new__(
            cls.module.UnifiedLedgerBuilder
        )

    def test_regime_extraction_uses_only_frozen_schema_columns(self):
        extract = self.builder._extract_five_factor_regime
        self.assertEqual(
            extract({"five_factor_regime": "normalized", "vol_regime": "HIGH"}),
            "normalized",
        )
        self.assertEqual(
            extract({"joint_regime": "low_calm", "regime": "stress"}),
            "low_calm",
        )
        self.assertIsNone(extract({"regime": "normalized", "vol_regime": "HIGH"}))

    def test_monthly_window_counts_six_sessions_after_signal_through_expiry(self):
        calendar = FakeCalendar(
            [
                "2024-06-10",
                "2024-06-11",
                "2024-06-12",
                "2024-06-13",
                "2024-06-14",
                "2024-06-17",
                "2024-06-18",
                "2024-06-20",
                "2024-06-21",
            ]
        )
        expiry, sessions = self.builder._next_standard_monthly_expiry_window(
            datetime.date(2024, 6, 12), calendar=calendar
        )
        self.assertEqual(expiry, datetime.date(2024, 6, 21))
        self.assertEqual(sessions, 6)

    def test_half_day_bar_uses_actual_market_close(self):
        half_day = datetime.date(2024, 11, 29)
        calendar = FakeCalendar(
            ["2024-11-26", "2024-11-27", "2024-11-29"],
            closes={
                half_day: pd.Timestamp("2024-11-29 18:00:00", tz="UTC")
            },
        )
        history = pd.DataFrame(
            {
                "Open": [501.0, 504.0, 508.0],
                "High": [505.0, 507.0, 510.0],
                "Low": [500.0, 503.0, 507.0],
                "Close": [503.0, 506.0, 509.0],
            },
            index=pd.to_datetime(["2024-11-26", "2024-11-27", "2024-11-29"]),
        )
        after_publication = self.module.NY_TZ.localize(
            datetime.datetime(2024, 11, 29, 13, 15)
        )
        bars = self.builder._load_completed_qqq_bars(
            history=history,
            now=after_publication,
            calendar=calendar,
        )
        self.assertEqual(bars.index[-1], half_day)
        self.assertEqual(list(bars.columns), ["Open", "High", "Low", "Close"])

        before_publication = self.module.NY_TZ.localize(
            datetime.datetime(2024, 11, 29, 13, 5)
        )
        bars = self.builder._load_completed_qqq_bars(
            history=history,
            now=before_publication,
            calendar=calendar,
        )
        self.assertEqual(bars.index[-1], datetime.date(2024, 11, 27))

    def test_entry_staleness_uses_actual_session_close(self):
        entry_date = datetime.date(2024, 11, 29)
        calendar = FakeCalendar(
            [entry_date],
            closes={
                entry_date: pd.Timestamp("2024-11-29 18:00:00", tz="UTC")
            },
        )
        before_close = self.module.NY_TZ.localize(
            datetime.datetime(2024, 11, 29, 12, 59)
        )
        after_close = self.module.NY_TZ.localize(
            datetime.datetime(2024, 11, 29, 13, 1)
        )
        self.assertIsNone(
            self.builder._entry_stale_reason(
                entry_date, now=before_close, calendar=calendar
            )
        )
        self.assertIn(
            "已于13:00",
            self.builder._entry_stale_reason(
                entry_date, now=after_close, calendar=calendar
            ),
        )

    def test_wall_expiry_requires_explicit_matching_metadata(self):
        extract = self.builder._extract_wall_expiry
        self.assertEqual(
            extract({"monthly_wall_expiry": "2024-06-21"}),
            datetime.date(2024, 6, 21),
        )
        self.assertEqual(
            extract({"wall_expiry": "2024-06-21"}),
            datetime.date(2024, 6, 21),
        )
        self.assertEqual(
            extract(
                {
                    "put_wall_expiry": "20240621",
                    "call_wall_expiry": "20240621",
                }
            ),
            datetime.date(2024, 6, 21),
        )
        self.assertIsNone(
            extract(
                {
                    "put_wall_expiry": "20240621",
                    "call_wall_expiry": "20240628",
                }
            )
        )
        self.assertEqual(
            self.builder._extract_wall_source_date(
                {"monthly_wall_oi_source_date": "2024-06-12"}
            ),
            datetime.date(2024, 6, 12),
        )

    def test_signal_scan_runs_before_broker_connection_failure(self):
        builder = self.module.UnifiedLedgerBuilder.__new__(
            self.module.UnifiedLedgerBuilder
        )
        calls = []
        builder.action_logs = []
        builder.logger = SimpleNamespace(error=lambda message: calls.append("log_error"))
        builder.is_trading_day = lambda: True
        builder.scan_qqq_research_signals = lambda: calls.append("scan")
        builder.scan_qqq_option_contract_candidates = lambda: calls.append(
            "scan_contracts"
        )

        def fail_connect():
            calls.append("connect")
            raise RuntimeError("broker unavailable")

        builder.connect = fail_connect
        builder.log_action = lambda message: builder.action_logs.append(message)
        builder.send_report = lambda subject, body: calls.append("send_report")
        builder.disconnect = lambda: calls.append("disconnect")

        builder.run()

        self.assertEqual(calls[:2], ["scan", "connect"])
        self.assertNotIn("scan_contracts", calls)
        self.assertIn("send_report", calls)

    def test_contract_candidates_scan_only_after_broker_connects(self):
        builder = self.module.UnifiedLedgerBuilder.__new__(
            self.module.UnifiedLedgerBuilder
        )
        calls = []
        builder.action_logs = []
        builder.logger = SimpleNamespace(error=lambda message: calls.append("log_error"))
        builder.is_trading_day = lambda: True
        builder.scan_qqq_research_signals = lambda: calls.append("scan")
        builder.connect = lambda: calls.append("connect")
        builder.scan_qqq_option_contract_candidates = lambda: calls.append(
            "scan_contracts"
        )
        builder.reconcile_executions = lambda: calls.append("reconcile")
        builder.execute_maintenance = lambda: calls.append("maintenance")
        builder.execute_laddering = lambda: calls.append("laddering")
        builder.print_current_inventory = lambda: calls.append("inventory")
        builder.send_report = lambda subject, body: calls.append("send_report")
        builder.disconnect = lambda: calls.append("disconnect")

        builder.run()

        self.assertEqual(
            calls[:4], ["scan", "connect", "scan_contracts", "reconcile"]
        )

    def test_candidate_strike_sampling_covers_spot_and_protective_wing(self):
        selected = self.builder._nearest_option_strikes(
            range(70, 111), centers=[100.0, 90.0], count_per_center=5
        )
        self.assertIn(100.0, selected)
        self.assertIn(90.0, selected)
        self.assertLess(len(selected), 15)

    def test_frozen_candidate_expiry_uses_closest_dte_not_monthly_preference(self):
        selected = self.builder._closest_option_expiry(
            ["20260821", "20260904", "20260918"],
            target_days=45,
            today=datetime.date(2026, 7, 22),
        )
        self.assertEqual(selected, "20260904")


if __name__ == "__main__":
    unittest.main()
