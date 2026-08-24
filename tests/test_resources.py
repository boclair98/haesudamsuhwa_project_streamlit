import tempfile
import unittest
from pathlib import Path

import pandas as pd

from desalination.resources import NUMERIC_COLUMNS, ResourceError, load_history


class ResourceTests(unittest.TestCase):
    def _row(self, timestamp, value=1.0):
        return {"timestamp": timestamp, **{column: value for column in NUMERIC_COLUMNS}}

    def test_load_history_drops_invalid_and_duplicate_records(self):
        frame = pd.DataFrame(
            [
                self._row("2021-01-01 00:00", 1.0),
                self._row("2021-01-01 00:00", 2.0),
                self._row("not-a-date", 3.0),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            frame.to_csv(path, index=False, encoding="utf-8")
            clean, profile = load_history(path)

        self.assertEqual(len(clean), 1)
        self.assertEqual(clean.loc[0, "temperature_c"], 2.0)
        self.assertEqual(profile.invalid_timestamp_rows, 1)
        self.assertEqual(profile.duplicate_rows_removed, 1)

    def test_load_history_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            pd.DataFrame({"timestamp": ["2021-01-01"]}).to_csv(path, index=False)
            with self.assertRaises(ResourceError):
                load_history(path)


if __name__ == "__main__":
    unittest.main()
