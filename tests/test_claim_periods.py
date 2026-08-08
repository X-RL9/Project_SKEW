import unittest

import pandas as pd

from skew_pipeline import _claim_window


class ClaimPeriodTests(unittest.TestCase):
    def test_defaults_to_latest_five_years(self):
        start, end, supplied = _claim_window(
            "UK imports have increased", pd.Timestamp("2025-12-01")
        )
        self.assertEqual(start, pd.Timestamp("2020-12-01"))
        self.assertEqual(end, pd.Timestamp("2025-12-01"))
        self.assertFalse(supplied)

    def test_uses_explicit_since_year(self):
        start, end, supplied = _claim_window(
            "UK inflation has fallen since 2023", pd.Timestamp("2026-01-01")
        )
        self.assertEqual(start, pd.Timestamp("2023-01-01"))
        self.assertEqual(end, pd.Timestamp("2026-01-01"))
        self.assertTrue(supplied)

    def test_uses_explicit_year_range(self):
        start, end, supplied = _claim_window(
            "GDP increased between 2020 and 2024", pd.Timestamp("2026-01-01")
        )
        self.assertEqual(start, pd.Timestamp("2020-01-01"))
        self.assertEqual(end, pd.Timestamp("2024-12-31"))
        self.assertTrue(supplied)


if __name__ == "__main__":
    unittest.main()
