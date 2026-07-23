import unittest
from datetime import date

from qqq_options_system.live.qqq_strategy_email import (
    QQQSignalContext,
    evaluate_qqq_signals,
    format_signal_email_section,
)


def context(**overrides):
    values = {
        "signal_date": date(2026, 7, 20),
        "entry_date": date(2026, 7, 21),
        "prior_close": 600.0,
        "high": 601.0,
        "close": 600.0,
        "put_wall": 580.0,
        "call_wall": 620.0,
        "wall_expiry": date(2026, 7, 31),
        "wall_data_expiry": date(2026, 7, 31),
        "wall_data_source_date": date(2026, 7, 20),
        "wall_sessions_to_expiry": 8,
        "five_factor_regime": "normalized",
    }
    values.update(overrides)
    return QQQSignalContext(**values)


class LiveQQQStrategyEmailTests(unittest.TestCase):
    def test_uptrend_uses_frozen_two_percent_intraday_rule(self):
        ctx = context(high=612.0, close=606.0)
        signals = evaluate_qqq_signals(ctx)
        uptrend = signals[0]
        self.assertEqual(uptrend.status, "TRIGGERED")
        self.assertIn("-0.40 Delta", " ".join(uptrend.legs))
        self.assertIn("45D", " ".join(uptrend.legs))
        self.assertIn("2026-07-21", uptrend.entry_rule)
        self.assertIn("all-signals", " ".join(uptrend.caveats))

    def test_downtrend_combo_is_fixed_and_stress_regime_blocks_it(self):
        allowed = evaluate_qqq_signals(
            context(close=594.0, five_factor_regime="normalized")
        )[1]
        self.assertEqual(allowed.status, "TRIGGERED")
        details = " ".join(allowed.legs)
        self.assertIn("-0.50 Delta", details)
        self.assertIn("10%", details)
        self.assertIn("45DTE", details)

        blocked = evaluate_qqq_signals(
            context(close=594.0, five_factor_regime="high_rising")
        )[1]
        self.assertEqual(blocked.status, "BLOCKED")

    def test_downtrend_without_regime_is_watch_not_unqualified_trade(self):
        signal = evaluate_qqq_signals(
            context(close=594.0, five_factor_regime=None)
        )[1]
        self.assertEqual(signal.status, "WATCH")
        self.assertIn("人工复核", " ".join(signal.caveats))

    def test_unknown_regime_cannot_bypass_downtrend_gate(self):
        for regime in ("HIGH", "elevated", "stress", "typo_state"):
            with self.subTest(regime=regime):
                signal = evaluate_qqq_signals(
                    context(close=594.0, five_factor_regime=regime)
                )[1]
                self.assertEqual(signal.status, "WATCH")
                self.assertIn("不在冻结词表", " ".join(signal.caveats))

    def test_uptrend_vol_rising_is_watch_with_explicit_risk_note(self):
        signal = evaluate_qqq_signals(
            context(high=612.0, close=606.0, five_factor_regime="vol_rising")
        )[0]
        self.assertEqual(signal.status, "WATCH")
        self.assertIn("Short Gamma", " ".join(signal.caveats))

        high_rising = evaluate_qqq_signals(
            context(high=612.0, close=606.0, five_factor_regime="high_rising")
        )[0]
        self.assertEqual(high_rising.status, "TRIGGERED")
        self.assertIn("左尾风险", " ".join(high_rising.caveats))
        self.assertIn("high_rising", " ".join(high_rising.evidence))

    def test_monthly_call_wall_break_requires_six_sessions_and_three_percent(self):
        triggered = evaluate_qqq_signals(
            context(
                close=625.0,
                high=625.0,
                wall_expiry=date(2026, 7, 31),
                wall_sessions_to_expiry=6,
            )
        )[2]
        self.assertEqual(triggered.status, "TRIGGERED")
        self.assertIn("Call Debit Spread", " ".join(triggered.legs))

        wrong_timing = evaluate_qqq_signals(
            context(close=625.0, high=625.0, wall_sessions_to_expiry=7)
        )[2]
        self.assertEqual(wrong_timing.status, "NO_SIGNAL")

        too_far = evaluate_qqq_signals(
            context(close=640.0, high=640.0, wall_sessions_to_expiry=6)
        )[2]
        self.assertEqual(too_far.status, "NO_SIGNAL")

    def test_wall_requires_matching_expiry_metadata(self):
        missing = evaluate_qqq_signals(
            context(
                close=625.0,
                high=625.0,
                wall_sessions_to_expiry=6,
                wall_data_expiry=None,
            )
        )[2]
        self.assertEqual(missing.status, "WATCH")
        self.assertIn("wall_expiry", " ".join(missing.caveats))

        mismatch = evaluate_qqq_signals(
            context(
                close=625.0,
                high=625.0,
                wall_sessions_to_expiry=6,
                wall_data_expiry=date(2026, 8, 21),
            )
        )[2]
        self.assertEqual(mismatch.status, "WATCH")
        self.assertIn("不一致", " ".join(mismatch.caveats))

        wrong_source = evaluate_qqq_signals(
            context(
                close=625.0,
                high=625.0,
                wall_sessions_to_expiry=6,
                wall_data_source_date=date(2026, 7, 17),
            )
        )[2]
        self.assertEqual(wrong_source.status, "WATCH")
        self.assertIn("OI 来源日", " ".join(wrong_source.caveats))

    def test_put_wall_inversion_and_three_percent_boundary(self):
        put_break = evaluate_qqq_signals(
            context(close=570.0, high=600.0, wall_sessions_to_expiry=6)
        )[2]
        self.assertEqual(put_break.status, "TRIGGERED")
        self.assertIn("Put Debit Spread", " ".join(put_break.legs))

        inverted = evaluate_qqq_signals(
            context(
                put_wall=625.0,
                call_wall=620.0,
                wall_sessions_to_expiry=6,
            )
        )[2]
        self.assertEqual(inverted.status, "WATCH")
        self.assertIn("倒挂", " ".join(inverted.caveats))

        boundary_close = 620.0 / (1.0 - 0.03)
        boundary = evaluate_qqq_signals(
            context(
                close=boundary_close,
                high=boundary_close,
                wall_sessions_to_expiry=6,
            )
        )[2]
        self.assertEqual(boundary.status, "TRIGGERED")

    def test_actionable_signal_is_marked_stale_after_entry_window(self):
        signal = evaluate_qqq_signals(
            context(
                close=594.0,
                five_factor_regime="normalized",
                entry_stale_reason="T+1交易日2026-07-21已收盘",
            )
        )[1]
        self.assertEqual(signal.status, "STALE")
        self.assertIn("不入场", signal.entry_rule)
        self.assertIn("不得按过期价格追单", " ".join(signal.caveats))

    def test_email_section_contains_trigger_evidence_structure_and_risk(self):
        ctx = context(high=612.0, close=606.0, iv_rank=48.0, skew_25d=3.2)
        text = format_signal_email_section(ctx, evaluate_qqq_signals(ctx))
        self.assertIn("QQQ 冻结策略信号监控", text)
        self.assertIn("触发条件", text)
        self.assertIn("结构", text)
        self.assertIn("风险", text)
        self.assertIn("仅作研究提示，不自动下单", text)


if __name__ == "__main__":
    unittest.main()
