import unittest

from qqq_options_system.market.contracts import canonical_contract_id, calendar_dte


class ContractTests(unittest.TestCase):
    def test_contract_id_from_canonical_fields(self):
        self.assertEqual(
            canonical_contract_id("qqq", "6/2/2026", "c", "495.00"),
            "QQQ_20260602_C_00495000",
        )

    def test_calendar_dte(self):
        self.assertEqual(calendar_dte("2026-06-02", "6/18/2026"), 16)


if __name__ == "__main__":
    unittest.main()
