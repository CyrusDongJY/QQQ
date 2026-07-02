import unittest

import pandas as pd

from qqq_options_system.data.normalize import (
    QuoteFilterConfig,
    filter_quote_quality,
    normalize_qqq_wide_to_long,
)


class NormalizeTests(unittest.TestCase):
    def test_wide_call_put_rows_become_long_rows(self):
        frame = pd.DataFrame(
            [
                {
                    "ticker": "QQQ",
                    "cOpra": "QQQ260602C00495000",
                    "pOpra": "QQQ260602P00495000",
                    "stkPx": "745.63",
                    "expirDate": "6/2/2026",
                    "yte": "0.00000",
                    "strike": "495.00",
                    "cVolu": "35",
                    "cOi": "0",
                    "pVolu": "13",
                    "pOi": "667",
                    "cBidPx": "248.47",
                    "cValue": "250.63",
                    "cAskPx": "252.15",
                    "pBidPx": "0.00",
                    "pValue": "0.00",
                    "pAskPx": "0.01",
                    "cBidIv": "0",
                    "cMidIv": "0",
                    "cAskIv": "0",
                    "smoothSmvVol": "0.139",
                    "pBidIv": "0",
                    "pMidIv": "0",
                    "pAskIv": "0",
                    "delta": "1.0",
                    "gamma": "0",
                    "theta": "0",
                    "vega": "0.00001",
                    "rho": "0",
                    "extCTheo": "250.63",
                    "extPTheo": "0",
                    "spot_px": "0.00",
                    "trade_date": "6/2/2026",
                }
            ]
        )
        out = normalize_qqq_wide_to_long(frame, "sample.csv", "abcdef123456")
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out["right"]), {"C", "P"})
        self.assertEqual(out.loc[out["right"] == "C", "bid"].iloc[0], 248.47)
        self.assertEqual(out.loc[out["right"] == "P", "open_interest"].iloc[0], 667)
        self.assertEqual(out.loc[out["right"] == "C", "underlying_price"].iloc[0], 745.63)

    def test_missing_opra_schema_drift_still_builds_ids(self):
        frame = pd.DataFrame(
            [
                {
                    "ticker": "QQQ",
                    "stkPx": "100.0",
                    "expirDate": "1/19/2007",
                    "yte": "0.04384",
                    "strike": "90.00",
                    "cBidPx": "10.00",
                    "cValue": "10.10",
                    "cAskPx": "10.20",
                    "pBidPx": "0.10",
                    "pValue": "0.20",
                    "pAskPx": "0.30",
                    "trade_date": "1/3/2007",
                }
            ]
        )
        out = normalize_qqq_wide_to_long(frame, "old.csv", "oldschema")
        self.assertEqual(len(out), 2)
        self.assertTrue(out["opra_alias"].isna().all())
        self.assertEqual(
            out.loc[out["right"] == "P", "contract_id"].iloc[0],
            "QQQ_20070119_P_00090000",
        )

    def test_quote_filter_removes_invalid_and_illiquid_rows(self):
        frame = pd.DataFrame(
            [
                {"bid": 1.0, "mid": 1.1, "ask": 1.2, "open_interest": 10, "volume": 1, "mid_iv": 0.2},
                {"bid": -1.0, "mid": 1.1, "ask": 1.2, "open_interest": 10, "volume": 1, "mid_iv": 0.2},
                {"bid": 1.0, "mid": 1.1, "ask": 0.9, "open_interest": 10, "volume": 1, "mid_iv": 0.2},
                {"bid": 1.0, "mid": 1.1, "ask": 2.0, "open_interest": 10, "volume": 1, "mid_iv": 0.2},
                {"bid": 1.0, "mid": 1.1, "ask": 1.2, "open_interest": 0, "volume": 1, "mid_iv": 0.2},
            ]
        )
        filtered = filter_quote_quality(
            frame,
            QuoteFilterConfig(max_relative_spread=0.25, min_open_interest=1),
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["bid"], 1.0)


if __name__ == "__main__":
    unittest.main()
