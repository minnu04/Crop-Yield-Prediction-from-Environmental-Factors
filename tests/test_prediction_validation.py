import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict import predict_crop_yield


class DummyModel:
    def predict(self, df):
        return np.array([123.45] * len(df))


class TestPredictionValidation(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.valid_input = {
            "N": 90,
            "P": 40,
            "K": 40,
            "pH": 6.5,
            "organic_matter": 2.8,
            "rainfall": 220,
            "temp_min": 18,
            "temp_max": 31,
            "fertilizer_usage": 120,
            "crop_type": "rice",
        }

    def test_predict_crop_yield_valid_input(self):
        prediction = predict_crop_yield(self.model, self.valid_input, feature_names=[])
        self.assertIsInstance(float(prediction), float)
        self.assertAlmostEqual(prediction, 123.45, places=2)

    def test_predict_crop_yield_missing_fields_raises(self):
        bad_input = {
            "N": 90,
            "P": 40,
            "K": 40,
            "pH": 6.5,
        }

        with self.assertRaises(ValueError):
            predict_crop_yield(self.model, bad_input, feature_names=[])


if __name__ == "__main__":
    unittest.main()
