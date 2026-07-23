import tempfile
import unittest
import warnings
from pathlib import Path

from qqq_options_system.data.orats_files import discover_files


class OratsFileDiscoveryTests(unittest.TestCase):
    def test_unreadable_and_empty_csv_files_do_not_abort_year(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            year = Path(tmpdir) / "2022"
            year.mkdir()
            (year / "ORATS_SMV_Strikes_20220103.csv").write_text(
                "ticker,stkPx,trade_date\nQQQ,400,2022-01-03\n",
                encoding="utf-8",
            )
            (year / "ORATS_SMV_Strikes_20220104.csv").write_bytes(b"\xe0\x27\x18\x9d")
            (year / "ORATS_SMV_Strikes_20220105.csv").write_bytes(b"")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                files = discover_files(Path(tmpdir))

            self.assertEqual([item.file_date.isoformat() for item in files], ["2022-01-03"])
            self.assertEqual(len(caught), 2)
            self.assertTrue(all("Skipping unreadable ORATS CSV" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
