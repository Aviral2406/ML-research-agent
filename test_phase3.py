import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aitraining import ExperimentConfig, MLExperimentAnalyst, build_demo_dataset


class Phase3Tests(unittest.TestCase):
    def test_tuning_is_opt_in_and_reported(self):
        config = ExperimentConfig(
            tune=True,
            n_iter=1,
            tuning_cv=2,
            cv_folds=2,
            models=("ridge_regression",),
        )
        report = MLExperimentAnalyst(
            build_demo_dataset(), "sale_price", config=config
        ).run_experiment()

        self.assertTrue(report["hyperparameter_optimization"]["enabled"])
        self.assertTrue(report["model_selection"]["tuned"])
        self.assertIn("best_params", report["hyperparameter_optimization"]["results"]["ridge_regression"])
        json.dumps(report)

    def test_error_analysis_and_importance_are_serializable(self):
        analyst = MLExperimentAnalyst(build_demo_dataset(), "sale_price")
        report = analyst.run_experiment(folds=2)

        self.assertEqual(report["error_analysis"]["task"], "regression")
        self.assertEqual(len(report["error_analysis"]["rows"]), report["split"]["test_rows"])
        self.assertTrue(report["feature_importance"]["available"])
        self.assertTrue(report["feature_importance"]["features"])
        json.dumps(report)

    def test_classification_error_analysis_has_confusion_matrix(self):
        data = pd.DataFrame({
            "feature": list(range(24)),
            "target": ["yes", "no"] * 12,
        })
        config = ExperimentConfig(
            task="classification",
            tune=True,
            n_iter=1,
            tuning_cv=2,
            cv_folds=2,
            models=("logistic_regression",),
        )
        report = MLExperimentAnalyst(data, "target", config=config).run_experiment()

        analysis = report["error_analysis"]
        self.assertEqual(analysis["task"], "classification")
        self.assertIn("confusion_matrix", analysis)
        self.assertIn("per_class_metrics", analysis)
        self.assertTrue(report["model_selection"]["tuned"])

    def test_tuning_artifact_is_written(self):
        config = ExperimentConfig(
            tune=True,
            n_iter=1,
            tuning_cv=2,
            cv_folds=2,
            models=("ridge_regression",),
        )
        analyst = MLExperimentAnalyst(build_demo_dataset(), "sale_price", config=config)
        with tempfile.TemporaryDirectory() as directory:
            analyst.save_model_artifact(directory)
            search_results = Path(directory) / analyst.experiment_id / "search_results.json"
            self.assertTrue(search_results.exists())
            self.assertTrue(json.loads(search_results.read_text(encoding="utf-8"))["enabled"])


if __name__ == "__main__":
    unittest.main()
