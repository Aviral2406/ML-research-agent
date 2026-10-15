import json
import unittest

import pandas as pd

from aitraining import (
    ExperimentConfig,
    FeatureEngineeringTransformer,
    FeatureTransformationRegistry,
    LeakageAuditor,
    MLExperimentAnalyst,
    ModelRegistry,
    TaskDetector,
    ValidationStrategySelector,
)


class Phase25Tests(unittest.TestCase):
    def test_transformation_registry_is_explicit(self):
        descriptions = FeatureTransformationRegistry.describe(["log1p", "datetime_components"])
        self.assertEqual([item["name"] for item in descriptions], ["log1p", "datetime_components"])
        self.assertEqual(descriptions[0]["group"], "numeric")

    def test_feature_names_and_lineage_are_recoverable(self):
        data = pd.DataFrame({
            "age": [20, 30, 40, 50, 60, 70],
            "income": [10, 20, 30, 40, 50, 60],
            "when": pd.date_range("2024-01-01", periods=6),
        })
        transformer = FeatureEngineeringTransformer(enabled=True, max_interaction_features=2).fit(data)
        names = transformer.get_feature_names_out().tolist()
        self.assertIn("when__year", names)
        self.assertIn("age__x__income", names)

        target = pd.Series([1, 2, 3, 4, 5, 6], name="target")
        analyst = MLExperimentAnalyst(pd.concat([data, target], axis=1), "target")
        report = analyst._feature_engineering_report()
        self.assertEqual(report["feature_name_lineage"]["age"], "age")

    def test_task_detection_rationales_cover_discrete_numeric_targets(self):
        binary = pd.DataFrame({"x": range(20), "target": [0, 1] * 10})
        multiclass = pd.DataFrame({"x": range(30), "target": [0, 1, 2] * 10})
        continuous = pd.DataFrame({"x": range(30), "target": [value + 0.25 for value in range(30)]})

        self.assertEqual(TaskDetector.detect(binary, "target")["task"], "binary_classification")
        self.assertEqual(TaskDetector.detect(multiclass, "target")["task"], "multiclass_classification")
        self.assertEqual(TaskDetector.detect(continuous, "target")["task"], "regression")
        self.assertIn("reason", TaskDetector.detect(binary, "target"))

    def test_leakage_report_separates_confirmed_and_review_findings(self):
        data = pd.DataFrame({
            "id": range(6),
            "feature": [1, 2, 3, 4, 5, 6],
            "target": [10, 20, 30, 40, 50, 60],
            "target_copy": [10, 20, 30, 40, 50, 60],
        })
        report = LeakageAuditor(data, "target").audit_report()
        self.assertTrue(report["errors"])
        self.assertTrue(any(item["category"] == "id_like_column" for item in report["informational"]))
        self.assertFalse(report["preprocessing_outside_cv"])
        self.assertFalse(report["target_encoding_outside_cv"])

    def test_validation_records_fold_fallback_reason(self):
        data = pd.DataFrame({"x": range(12), "target": [0] * 3 + [1] * 9})
        strategy = ValidationStrategySelector(data, "target").select(cv_folds=5)
        self.assertEqual(strategy["strategy"], "stratified_kfold")
        self.assertEqual(strategy["n_splits"], 3)
        self.assertIn("reduced", strategy["fallback_reason"])

    def test_invalid_configuration_fails_before_training(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(test_size=0)
        with self.assertRaises(ValueError):
            ExperimentConfig(cv_folds=1)
        with self.assertRaises(ValueError):
            ExperimentConfig(task="classification", cv_strategy="timeseries")
        with self.assertRaises(ValueError):
            ExperimentConfig(feature_engineering_mode="unknown")

    def test_model_registry_is_task_isolated_and_structured(self):
        regression = ModelRegistry("regression")
        classification = ModelRegistry("classification")
        self.assertTrue(all("classifier" not in name for name in regression.names()))
        self.assertTrue(all("regressor" not in name for name in classification.names()))
        specification = classification.specifications()[0]
        self.assertEqual(specification["task"], "classification")
        self.assertIn("default_parameters", specification)
        self.assertIn("supported_hyperparameters", specification)

    def test_single_class_classification_fails_clearly(self):
        data = pd.DataFrame({"x": range(8), "target": ["only"] * 8})
        with self.assertRaisesRegex(ValueError, "at least two target classes"):
            MLExperimentAnalyst(data, "target", config=ExperimentConfig(task="classification"))

    def test_report_is_serializable_and_contains_hardening_metadata(self):
        data = pd.DataFrame({
            "x": range(30),
            "category": ["a", "b", "c"] * 10,
            "target": [value * 1.5 for value in range(30)],
        })
        config = ExperimentConfig(enable_feature_engineering=True, cv_folds=3)
        report = MLExperimentAnalyst(data, "target", config=config).run_experiment()
        json.dumps(report)
        self.assertIn("registry", report["parameters"])
        self.assertIn("leakage_analysis", report)
        self.assertIn("feature_name_lineage", report["feature_engineering_plan"])
        self.assertIn("random_state", report["configuration"])


if __name__ == "__main__":
    unittest.main()
