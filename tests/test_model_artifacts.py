import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict import predict_crop_yield
from utils import load_model


class TestModelArtifacts(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.model_path = self.project_root / "models" / "crop_yield_rf_pipeline.joblib"
        self.outputs_dir = self.project_root / "outputs"

    def test_expected_artifact_files_exist(self):
        expected_files = [
            self.model_path,
            self.outputs_dir / "feature_importance.csv",
            self.outputs_dir / "results_summary.json",
            self.outputs_dir / "correlation_heatmap.png",
            self.outputs_dir / "feature_vs_yield_scatter.png",
            self.outputs_dir / "yield_by_crop_type_boxplot.png",
            self.outputs_dir / "yield_distribution.png",
            self.outputs_dir / "feature_importance_TOP15.png",
        ]

        for file_path in expected_files:
            self.assertTrue(file_path.exists(), f"Missing artifact: {file_path}")

    def test_loaded_model_can_predict(self):
        self.assertTrue(self.model_path.exists(), "Model artifact is missing")
        model = load_model(str(self.model_path))

        sample_input = {
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

        prediction = predict_crop_yield(model, sample_input, feature_names=[])
        self.assertGreater(float(prediction), 0.0)


if __name__ == "__main__":
    unittest.main()
