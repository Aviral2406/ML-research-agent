import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api import create_app
from aitraining import MLExperimentAnalyst, build_demo_dataset


class AnalystApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        analyst = MLExperimentAnalyst(build_demo_dataset(), target="sale_price")
        cls.client = TestClient(create_app(analyst))

    def test_health(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_profile(self) -> None:
        response = self.client.get("/profile")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["rows"], 10)
        self.assertIn("sale_price", response.json()["columns"])

    def test_single_prediction(self) -> None:
        response = self.client.post(
            "/predict",
            json={
                "features": {
                    "area_sqft": 2000,
                    "bedrooms": 4,
                    "neighborhood": "West",
                }
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["target"], "sale_price")
        self.assertIsInstance(response.json()["prediction"], float)

    def test_batch_prediction(self) -> None:
        response = self.client.post(
            "/predict-batch",
            json={
                "rows": [
                    {"area_sqft": 900, "bedrooms": 2, "neighborhood": "North"},
                    {"area_sqft": 2400, "bedrooms": 4, "neighborhood": "West"},
                ]
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["predictions"]), 2)

    def test_missing_feature_returns_bad_request(self) -> None:
        response = self.client.post(
            "/predict",
            json={"features": {"area_sqft": 2000}},
        )

        self.assertEqual(response.status_code, 400)

    def test_create_app_uses_environment_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "custom_dataset.csv"
            build_demo_dataset().to_csv(csv_path, index=False)

            previous_dataset = os.environ.get("MODEL_DATASET_PATH")
            previous_target = os.environ.get("MODEL_TARGET")
            os.environ["MODEL_DATASET_PATH"] = str(csv_path)
            os.environ["MODEL_TARGET"] = "sale_price"

            try:
                client = TestClient(create_app())
                response = client.get("/profile")
            finally:
                if previous_dataset is None:
                    os.environ.pop("MODEL_DATASET_PATH", None)
                else:
                    os.environ["MODEL_DATASET_PATH"] = previous_dataset
                if previous_target is None:
                    os.environ.pop("MODEL_TARGET", None)
                else:
                    os.environ["MODEL_TARGET"] = previous_target

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["rows"], 10)

    def test_create_app_rejects_invalid_environment_configuration(self) -> None:
        previous_dataset = os.environ.get("MODEL_DATASET_PATH")
        previous_target = os.environ.get("MODEL_TARGET")
        os.environ["MODEL_DATASET_PATH"] = "missing_dataset.csv"
        os.environ["MODEL_TARGET"] = "sale_price"

        try:
            with self.assertRaises((FileNotFoundError, ValueError)):
                create_app()
        finally:
            if previous_dataset is None:
                os.environ.pop("MODEL_DATASET_PATH", None)
            else:
                os.environ["MODEL_DATASET_PATH"] = previous_dataset
            if previous_target is None:
                os.environ.pop("MODEL_TARGET", None)
            else:
                os.environ["MODEL_TARGET"] = previous_target


if __name__ == "__main__":
    unittest.main()