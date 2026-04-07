import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_data, validate_required_columns


class TestDataLoading(unittest.TestCase):
    def test_load_data_success_and_extra_columns(self):
        required_columns = [
            "N", "P", "K", "pH", "organic_matter", "rainfall",
            "temp_min", "temp_max", "fertilizer_usage", "crop_type", "crop_yield"
        ]

        row = {
            "N": 50,
            "P": 20,
            "K": 30,
            "pH": 6.7,
            "organic_matter": 2.5,
            "rainfall": 180,
            "temp_min": 16,
            "temp_max": 30,
            "fertilizer_usage": 100,
            "crop_type": "rice",
            "crop_yield": 120,
            "extra_col": 999,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "sample.csv"
            pd.DataFrame([row]).to_csv(csv_path, index=False)

            df = load_data(str(csv_path))
            self.assertEqual(df.shape[0], 1)
            self.assertIn("extra_col", df.columns)
            self.assertTrue(validate_required_columns(df, required_columns))

    def test_load_data_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_data("non_existent_file.csv")


if __name__ == "__main__":
    unittest.main()
