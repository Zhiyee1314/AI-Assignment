import unittest
from pathlib import Path

import joblib
import pandas as pd

from prediction_service import FEATURES, predict_all_models, preprocess_patient_data


REPO_ROOT = Path(__file__).resolve().parents[1]


class PredictionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.imputer = joblib.load(REPO_ROOT / "models" / "imputer.pkl")
        cls.scaler = joblib.load(REPO_ROOT / "models" / "scaler.pkl")
        cls.models = {
            "ANN": joblib.load(REPO_ROOT / "models" / "ann_model.pkl"),
            "SVM": joblib.load(REPO_ROOT / "models" / "svm_model.pkl"),
            "KNN": joblib.load(REPO_ROOT / "models" / "knn_model.pkl"),
        }
        cls.patients = pd.DataFrame(
            [
                [3, 162, 80, 20, 70, 30.01, 0.51, 35],
                [1, 95, 66, 18, 85, 25.4, 0.25, 27],
                [20, 300, 200, 100, 900, 70.0, 3.0, 120],
            ],
            columns=FEATURES,
        )

    def test_backend_values_are_between_zero_and_one(self):
        normalized = preprocess_patient_data(self.patients, self.imputer, self.scaler)
        self.assertTrue(((normalized >= 0.0) & (normalized <= 1.0)).all().all())

    def test_batch_prediction_runs_all_three_models(self):
        results, _ = predict_all_models(
            self.patients,
            self.models,
            self.imputer,
            self.scaler,
        )
        self.assertEqual(len(results), 3)
        for model_name in ["ANN", "SVM", "KNN"]:
            self.assertIn(f"{model_name}_Prediction", results.columns)
            self.assertIn(f"{model_name}_Probability_Percent", results.columns)
            self.assertTrue(results[f"{model_name}_Probability_Percent"].between(0, 100).all())

    def test_missing_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            preprocess_patient_data(
                self.patients.drop(columns=["BMI"]),
                self.imputer,
                self.scaler,
            )


if __name__ == "__main__":
    unittest.main()
