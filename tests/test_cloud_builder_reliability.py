import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from test_cloud_signal_integration import load_cloud_builder_module


class BuilderReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_cloud_builder_module()

    def setUp(self):
        self.builder = self.module.UnifiedLedgerBuilder.__new__(
            self.module.UnifiedLedgerBuilder)
        self.builder.ib = MagicMock()
        self.builder.host = '127.0.0.1'
        self.builder.port = 4001
        self.builder.client_id = 889
        self.builder.logger = MagicMock()
        self.builder.action_logs = []
        self.builder.log_action = self.builder.action_logs.append

    def mock_workflow(self):
        self.builder.is_trading_day = MagicMock(return_value=True)
        self.builder.scan_qqq_research_signals = MagicMock(return_value='signal')
        self.builder.scan_qqq_option_contract_candidates = MagicMock()
        self.builder.reconcile_executions = MagicMock()
        self.builder.execute_maintenance = MagicMock()
        self.builder.execute_laddering = MagicMock()
        self.builder.print_current_inventory = MagicMock()
        self.builder.send_report = MagicMock(return_value=True)
        self.builder.qqq_strategy_signals = (SimpleNamespace(status='WATCH'),)

    def test_connection_preserves_readonly_and_sets_twenty_second_timeout(self):
        self.builder.connect()
        self.builder.ib.connect.assert_called_once_with(
            '127.0.0.1', 4001, clientId=889, readonly=True, timeout=20.0)
        self.builder.ib.reqMarketDataType.assert_called_once_with(1)

    def test_transient_sync_timeouts_retry_with_cleanup_and_backoff(self):
        self.builder.ib.connect.side_effect = [TimeoutError(), TimeoutError(), None]
        with patch.object(self.module.time, 'sleep') as sleep:
            self.builder.connect()
        self.assertEqual(self.builder.ib.connect.call_count, 3)
        self.assertEqual(self.builder.ib.disconnect.call_count, 2)
        self.assertEqual(sleep.call_args_list, [call(3), call(6)])
        self.builder.ib.reqMarketDataType.assert_called_once_with(1)
        self.assertIn('TimeoutError()', '\n'.join(self.builder.action_logs))

    def test_exhausted_retries_raise_original_timeout_and_never_request_quotes(self):
        error = TimeoutError()
        self.builder.ib.connect.side_effect = error
        with patch.object(self.module.time, 'sleep') as sleep:
            with self.assertRaises(TimeoutError) as raised:
                self.builder.connect()
        self.assertIs(raised.exception, error)
        self.assertEqual(self.builder.ib.connect.call_count, 3)
        self.assertEqual(self.builder.ib.disconnect.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(3), call(6)])
        self.builder.ib.reqMarketDataType.assert_not_called()
        self.assertEqual(self.builder.logger.exception.call_count, 3)

    def test_false_connected_state_is_not_accepted(self):
        self.builder.ib.isConnected.return_value = False
        with patch.object(self.module.time, 'sleep'):
            with self.assertRaises(ConnectionError):
                self.builder.connect()
        self.builder.ib.reqMarketDataType.assert_not_called()

    def test_failed_cleanup_stops_retrying_on_the_same_connection(self):
        self.builder.ib.connect.side_effect = TimeoutError()
        self.builder.ib.disconnect.side_effect = RuntimeError('cleanup failed')
        with patch.object(self.module.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                self.builder.connect()
        self.builder.ib.connect.assert_called_once()
        sleep.assert_not_called()

    def test_ib_error_logs_remain_visible_without_info_account_details(self):
        logger = logging.getLogger('ib_insync')
        self.assertTrue(logger.isEnabledFor(logging.ERROR))
        self.assertFalse(logger.isEnabledFor(logging.INFO))

    def test_full_run_recovers_before_running_each_business_stage_once(self):
        self.mock_workflow()
        self.builder.ib.connect.side_effect = [TimeoutError(), None]
        with patch.object(self.module.time, 'sleep'):
            self.assertTrue(self.builder.run())
        self.assertEqual(self.builder.ib.connect.call_count, 2)
        self.builder.scan_qqq_research_signals.assert_called_once()
        self.builder.scan_qqq_option_contract_candidates.assert_called_once()
        self.builder.reconcile_executions.assert_called_once()
        self.builder.execute_maintenance.assert_called_once()
        self.builder.execute_laddering.assert_called_once()
        self.builder.print_current_inventory.assert_called_once()
        self.builder.send_report.assert_called_once()

    def test_failed_connection_sends_one_alert_and_skips_ledger_work(self):
        self.mock_workflow()
        self.builder.ib.connect.side_effect = TimeoutError()
        with patch.object(self.module.time, 'sleep'):
            self.assertFalse(self.builder.run())
        self.builder.scan_qqq_option_contract_candidates.assert_called_once()
        self.builder.reconcile_executions.assert_not_called()
        self.builder.execute_maintenance.assert_not_called()
        self.builder.execute_laddering.assert_not_called()
        self.builder.print_current_inventory.assert_not_called()
        self.builder.send_report.assert_called_once()
        subject, body = self.builder.send_report.call_args.args
        self.assertIn('崩溃警报', subject)
        self.assertIn('TimeoutError()', body)
        self.assertIn('Traceback', body)

    def test_full_run_mail_failure_returns_failed_status(self):
        self.mock_workflow()
        self.builder.send_report.return_value = False
        self.assertFalse(self.builder.run())

    def test_full_run_cleanup_failure_is_logged_and_returns_failed_status(self):
        self.mock_workflow()
        self.builder.disconnect = MagicMock(side_effect=RuntimeError('cleanup failed'))
        self.assertFalse(self.builder.run())
        self.builder.send_report.assert_called_once()
        self.builder.logger.exception.assert_called()

    def test_open_scan_connection_failure_still_reports_without_ledger_work(self):
        self.mock_workflow()
        self.builder.ib.connect.side_effect = TimeoutError()
        with patch.object(self.module.time, 'sleep'):
            self.assertFalse(self.builder.run_qqq_live_scan())
        self.builder.scan_qqq_option_contract_candidates.assert_called_once()
        self.builder.send_report.assert_called_once()
        self.builder.reconcile_executions.assert_not_called()
        self.builder.execute_maintenance.assert_not_called()
        self.builder.execute_laddering.assert_not_called()

    def test_open_scan_missing_signal_section_returns_failure_and_sends_alert(self):
        self.mock_workflow()
        self.builder.scan_qqq_research_signals.return_value = None
        self.assertFalse(self.builder.run_qqq_live_scan())
        self.builder.ib.connect.assert_not_called()
        self.builder.send_report.assert_called_once()

    def test_no_active_signal_and_holiday_keep_quiet_success_exit_codes(self):
        self.mock_workflow()
        self.builder.qqq_strategy_signals = ()
        with patch.object(self.module, 'UnifiedLedgerBuilder', return_value=self.builder):
            self.assertEqual(self.module.main(['--mode', 'qqq-live-scan']), 0)
            self.builder.is_trading_day.return_value = False
            self.assertEqual(self.module.main([]), 0)
        self.builder.ib.connect.assert_not_called()
        self.builder.send_report.assert_not_called()

    def test_cli_returns_one_for_failed_full_or_open_run(self):
        builder = MagicMock()
        builder.run.return_value = False
        builder.run_qqq_live_scan.return_value = False
        with patch.object(self.module, 'UnifiedLedgerBuilder', return_value=builder):
            self.assertEqual(self.module.main([]), 1)
            self.assertEqual(self.module.main(['--mode', 'qqq-live-scan']), 1)


class BuilderMailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_cloud_builder_module()

    def setUp(self):
        self.builder = self.module.UnifiedLedgerBuilder.__new__(
            self.module.UnifiedLedgerBuilder)
        self.builder.logger = MagicMock()
        config = SimpleNamespace(
            SENDER_EMAIL='sender@example.invalid',
            QQ_RECEIVER_EMAIL='recipient@example.invalid',
            APP_PASSWORD='test-only')
        config_patch = patch.object(self.module, 'cfg', config)
        config_patch.start()
        self.addCleanup(config_patch.stop)
        self.smtp = MagicMock()
        self.smtp.sendmail.return_value = {}
        smtp_patch = patch.object(self.module.smtplib, 'SMTP_SSL', return_value=self.smtp)
        self.smtp_factory = smtp_patch.start()
        self.addCleanup(smtp_patch.stop)

    def test_smtp_acceptance_is_logged_and_returns_true(self):
        self.assertTrue(self.builder.send_report('subject', 'body'))
        self.smtp_factory.assert_called_once_with('smtp.gmail.com', 465, timeout=10)
        self.smtp.sendmail.assert_called_once()
        self.builder.logger.info.assert_called_once()

    def test_authentication_error_returns_false_and_does_not_claim_acceptance(self):
        self.smtp.login.side_effect = RuntimeError('authentication unavailable')
        self.assertFalse(self.builder.send_report('subject', 'body'))
        self.smtp.sendmail.assert_not_called()
        self.builder.logger.info.assert_not_called()
        self.builder.logger.exception.assert_called_once()

    def test_recipient_refusal_returns_false(self):
        self.smtp.sendmail.return_value = {'recipient@example.invalid': (550, b'no')}
        self.assertFalse(self.builder.send_report('subject', 'body'))
        self.builder.logger.info.assert_not_called()

    def test_quit_failure_does_not_undo_smtp_acceptance_or_send_again(self):
        self.smtp.quit.side_effect = TimeoutError()
        self.assertTrue(self.builder.send_report('subject', 'body'))
        self.smtp.sendmail.assert_called_once()
        self.smtp.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
