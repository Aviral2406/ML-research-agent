import json
import unittest

import pandas as pd

from aitraining import (
    DEFAULT_CLASSIFICATION_MODEL_NAMES,
    ExperimentConfig,
    FeatureEngineeringTransformer,
    MLExperimentAnalyst,
    ModelRegistry,
    TaskDetector,
)


class Phase2Tests(unittest.TestCase):
    def classification_data(self, rows=60):
        return pd.DataFrame({
            "amount": [value + 1 for value in range(rows)],
            "count": [value % 4 for value in range(rows)],
            "segment": ["a", "b", "c"] * (rows // 3),
            "recorded_at": pd.date_range("2024-01-01", periods=rows),
            "label": ["yes" if value % 2 else "no" for value in range(rows)],
        })

    def test_feature_engineering_is_deterministic_and_modular(self):
        data = pd.DataFrame({
            "amount": [1, 2, 3, 4, 1000, 1200],
            "count": [1, 2, 1, 2, 3, 4],
            "when": pd.date_range("2024-01-01", periods=6),
        })
        transformer = FeatureEngineeringTransformer(enabled=True, max_interaction_features=2)
        first = transformer.fit_transform(data)
        second = transformer.fit_transform(data)

        self.assertEqual(first.columns.tolist(), second.columns.tolist())
        self.assertEqual(first.shape, second.shape)
        self.assertIn("when__year", first.columns)
        self.assertIn("amount__x__count", first.columns)
        self.assertIn("amount__div__count", first.columns)
        self.assertGreaterEqual(len(first.columns), len(data.columns))

    def test_feature_engineering_fits_decisions_on_training_rows(self):
        train = pd.DataFrame({"value": [0, 1, 2, 3], "other": [1, 1, 2, 2]})
        test = pd.DataFrame({"value": [1000000], "other": [3]})
        transformer = FeatureEngineeringTransformer(enabled=True)
        transformer.fit(train)
        transformed = transformer.transform(test)

        self.assertNotIn("value", transformer.log_columns_)
        self.assertIn("value__x__other", transformed.columns)
        self.assertTrue((transformed["value"] == 1000000).all())

    def test_registry_exposes_task_specific_baselines(self):
        self.assertEqual(ModelRegistry("classification").names(), DEFAULT_CLASSIFICATION_MODEL_NAMES)
        self.assertEqual(set(ModelRegistry("classification").models()), set(DEFAULT_CLASSIFICATION_MODEL_NAMES))
        self.assertEqual(len(ModelRegistry("regression").models()), 5)

    def test_binary_classification_runs_all_baselines_and_selects_by_f1(self):
        analyst = MLExperimentAnalyst(
            self.classification_data(),
            "label",
            config=ExperimentConfig(task="classification", cv_folds=3),
        )
        report = analyst.run_experiment()

        self.assertEqual(report["task"]["task"], "binary_classification")
        self.assertEqual(set(report["candidate_models"]), set(DEFAULT_CLASSIFICATION_MODEL_NAMES))
        self.assertEqual(report["model_selection"]["selection_metric"], "cv_f1")
        self.assertIn("f1", report["final_test_results"])
        self.assertIn("accuracy", report["final_test_results"])
        self.assertEqual(report["classification"]["number_of_classes"], 2)
        self.assertTrue(all(item["training_status"] == "completed" for item in report["model_comparison"].values()))

    def test_classification_uses_stratified_validation(self):
        analyst = MLExperimentAnalyst(
            self.classification_data(),
            "label",
            config=ExperimentConfig(task="classification", cv_folds=4),
        )
        strategy = analyst.profile()["validation_strategy"]

        self.assertEqual(strategy["strategy"], "stratified_kfold")
        self.assertTrue(strategy["shuffle"])

    def test_small_class_counts_reduce_folds_with_warning(self):
        data = self.classification_data(18)
        data["label"] = ["yes"] * 3 + ["no"] * 15
        analyst = MLExperimentAnalyst(
            data,
            "label",
            config=ExperimentConfig(task="classification", cv_folds=5),
        )
        report = analyst.run_experiment()

        self.assertEqual(report["cross_validation"]["folds"], 3)
        self.assertTrue(any("reduced" in warning for warning in report["warnings"]))

    def test_phase2_report_is_json_serializable_and_preserves_contract(self):
        analyst = MLExperimentAnalyst(
            self.classification_data(),
            "label",
            config=ExperimentConfig(task="classification", cv_folds=3, enable_feature_engineering=True),
        )
        report = analyst.experiment_report(folds=3)
        json.dumps(report)

        for key in (
            "experiment_id", "dataset", "cross_validation", "model_selection",
            "final_test_results", "interpretation", "split", "configuration", "limitations",
            "task", "profiler", "validation_strategy", "leakage_analysis",
            "feature_engineering_plan", "model_comparison", "warnings", "research_summary",
        ):
            self.assertIn(key, report)
        self.assertTrue(report["feature_engineering_plan"]["enabled"])
        self.assertTrue(report["feature_engineering_plan"]["safeguards"]["fit_inside_pipeline"])

    def test_same_seed_produces_same_comparison_and_metrics(self):
        config = ExperimentConfig(task="classification", cv_folds=3, random_state=11)
        first = MLExperimentAnalyst(self.classification_data(), "label", config=config).run_experiment()
        second = MLExperimentAnalyst(self.classification_data(), "label", config=config).run_experiment()

        self.assertEqual(first["model_selection"], second["model_selection"])
        self.assertEqual(first["cross_validation"], second["cross_validation"])
        self.assertEqual(first["final_test_results"], second["final_test_results"])


if __name__ == "__main__":
    unittest.main()
