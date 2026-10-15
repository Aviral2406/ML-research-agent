import json
import math
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from aitraining import (
    DatasetProfiler,
    ExperimentConfig,
    LeakageAuditor,
    MLExperimentAnalyst,
    TaskDetector,
    ValidationStrategySelector,
    build_demo_dataset,
)


class MLExperimentAnalystTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyst = MLExperimentAnalyst(
            build_demo_dataset(), target="sale_price"
        )

    def test_profile_describes_dataset(self) -> None:
        profile = self.analyst.profile()

        self.assertEqual(profile["rows"], 10)
        self.assertIn("sale_price", profile["columns"])
        self.assertEqual(profile["missing_values"]["sale_price"], 0)
        self.assertEqual(profile["feature_count"], 3)
        self.assertEqual(profile["duplicate_rows"], 0)
        self.assertIn("neighborhood", profile["categorical_columns"])
        self.assertIn("area_sqft", profile["numerical_columns"])
        self.assertEqual(profile["suspicious_target_values"]["negative_count"], 0)

    def test_baselines_return_all_candidate_models(self) -> None:
        results = self.analyst.run_baselines()

        self.assertEqual(
            {result.model_name for result in results},
            {
                "dummy_regressor",
                "linear_regression",
                "ridge_regression",
                "random_forest_regressor",
                "gradient_boosting_regressor",
            },
        )
        self.assertTrue(all(result.mean_absolute_error >= 0 for result in results))

    def test_cross_validation_returns_metrics_for_each_model(self) -> None:
        report = self.analyst.cross_validate_baselines(folds=3)

        self.assertEqual(
            set(report),
            {
                "dummy_regressor",
                "linear_regression",
                "ridge_regression",
                "random_forest_regressor",
                "gradient_boosting_regressor",
            },
        )
        for metrics in report.values():
            self.assertEqual(set(metrics), {
                "mae_mean", "mae_std", "rmse_mean", "rmse_std",
                "mse_mean", "mse_std", "r2_mean", "r2_std", "mape_mean", "mape_std"
            })

    def test_train_validation_test_split_prevents_data_leakage(self) -> None:
        splits = self.analyst.prepare_data_splits(test_size=0.2, validation_size=0.2)

        x_train, x_val, x_test, y_train, y_val, y_test = splits
        self.assertEqual(len(set(x_train.index) & set(x_val.index)), 0)
        self.assertEqual(len(set(x_train.index) & set(x_test.index)), 0)
        self.assertEqual(len(set(x_val.index) & set(x_test.index)), 0)
        self.assertGreater(len(x_train), len(x_val))
        self.assertGreater(len(x_val), 0)
        self.assertEqual(len(y_train), len(x_train))

    def test_feature_engineering_adds_reproducible_columns(self) -> None:
        engineered = self.analyst.engineer_features(build_demo_dataset())

        self.assertEqual(engineered.columns.tolist(), build_demo_dataset().columns.tolist())

    def test_experiment_report_includes_metrics_and_metadata(self) -> None:
        report = self.analyst.run_experiment()

        self.assertRegex(report["experiment_id"], r"^exp_\d{8}_\d{6}$")
        self.assertIn("timestamp", report)
        self.assertIn("split", report)
        self.assertEqual(report["split"]["test_rows"], 2)
        self.assertEqual(set(report["candidate_models"]), {
            "dummy_regressor", "linear_regression", "ridge_regression", "random_forest_regressor",
            "gradient_boosting_regressor"
        })
        self.assertIn("final_test_results", report)
        self.assertIn("mse", report["final_test_results"])
        self.assertIn("interpretation", report)
        self.assertTrue(all(math.isfinite(value) for value in report["final_test_results"].values()))
        for metrics in report["cross_validation"]["models"].values():
            self.assertTrue(math.isfinite(metrics["mean_cv_mae"]))
            self.assertTrue(math.isfinite(metrics["std_cv_mae"]))
            self.assertTrue(math.isfinite(metrics["mean_cv_rmse"]))
            self.assertTrue(math.isfinite(metrics["std_cv_rmse"]))

    def test_selection_uses_lowest_cv_mae(self) -> None:
        report = self.analyst.run_experiment(folds=3)
        cv_models = report["cross_validation"]["models"]
        expected = min(cv_models, key=lambda name: cv_models[name]["mean_cv_mae"])

        self.assertEqual(report["model_selection"]["selected_model"], expected)
        self.assertEqual(report["model_selection"]["selection_metric"], "cv_mae")

    def test_experiment_report_exposes_limitations_for_cli(self) -> None:
        report = self.analyst.experiment_report()

        self.assertIn("limitations", report)
        self.assertEqual(report["limitations"], report["experiment"]["limitations"])
        self.assertIsInstance(report["limitations"], list)

    def test_train_test_sets_do_not_overlap(self) -> None:
        x_train, x_val, x_test, *_ = self.analyst.prepare_data_splits()

        self.assertEqual(set(x_train.index) & set(x_test.index), set())
        self.assertEqual(set(x_val.index) & set(x_test.index), set())
        self.assertEqual(len(x_train) + len(x_val) + len(x_test), len(self.analyst.data))

    def test_mape_is_finite_with_zero_actual_values(self) -> None:
        metrics = self.analyst._score_predictions(pd.Series([0.0, 10.0]), [3.0, 12.0])

        self.assertTrue(math.isfinite(metrics["mape"]))
        self.assertEqual(metrics["mape"], 0.2)

    def test_best_model_predicts_new_rows(self) -> None:
        selected_model = self.analyst.fit_best_model()
        new_rows = build_demo_dataset().drop(columns=["sale_price"]).iloc[:2]

        predictions = self.analyst.predict(new_rows)

        self.assertIn(selected_model, {
            "dummy_regressor", "linear_regression", "ridge_regression",
            "random_forest_regressor", "gradient_boosting_regressor"
        })
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions.name, "sale_price")

    def test_prediction_rejects_missing_features(self) -> None:
        self.analyst.fit_best_model()

        with self.assertRaises(ValueError):
            self.analyst.predict(pd.DataFrame({"area_sqft": [1500]}))

    def test_predict_csv_returns_predictions_and_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "new_rows.csv"
            output_path = Path(directory) / "predictions.csv"
            build_demo_dataset().drop(columns=["sale_price"]).iloc[:2].to_csv(
                input_path, index=False
            )

            result = self.analyst.predict_csv(str(input_path), str(output_path))

            self.assertIn("sale_price", result.columns)
            self.assertTrue(output_path.exists())
            self.assertEqual(len(pd.read_csv(output_path)), 2)

    def test_missing_target_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MLExperimentAnalyst(build_demo_dataset(), target="missing_column").profile()

    def test_csv_loader_records_source_and_rejects_invalid_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "houses.csv"
            build_demo_dataset().to_csv(csv_path, index=False)

            analyst = MLExperimentAnalyst.from_csv(str(csv_path), target="sale_price")
            self.assertEqual(analyst.profile()["filename"], "houses.csv")
            self.assertEqual(analyst.profile()["rows"], 10)

            invalid_path = Path(directory) / "invalid.csv"
            build_demo_dataset().assign(sale_price="unknown").to_csv(invalid_path, index=False)
            with self.assertRaises(ValueError):
                MLExperimentAnalyst.from_csv(str(invalid_path), target="sale_price")

    def test_csv_loader_rejects_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            MLExperimentAnalyst.from_csv("does-not-exist.csv", target="sale_price")

    def test_report_can_be_saved_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            self.analyst.save_report(str(report_path), folds=3)

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertIn("experiment_id", report)
        self.assertIn("model_selection", report)
        self.assertIn("final_test_results", report)
        self.assertIn("dataset", report["experiment"])
        self.assertIn("dataset", report)
        self.assertIn("feature_columns", report["experiment"]["dataset"])
        self.assertIn("duplicate_rows", report["experiment"]["dataset"])
        self.assertIn("split", report["experiment"])
        self.assertEqual(len(report["baselines"]), 5)

    def test_numerical_only_dataset_runs(self) -> None:
        data = pd.DataFrame({
            "feature_a": range(20),
            "feature_b": [value * 2 for value in range(20)],
            "target": [value * 3 + 1 for value in range(20)],
        })
        analyst = MLExperimentAnalyst(data, target="target", config=ExperimentConfig(cv_folds=3))

        self.assertEqual(analyst.profile()["categorical_columns"], [])
        self.assertEqual(set(analyst.run_experiment()["candidate_models"]), set(ExperimentConfig().models))

    def test_mixed_dataset_uses_categorical_preprocessing(self) -> None:
        data = pd.DataFrame({
            "numeric_feature": range(20),
            "category": ["a", "b"] * 10,
            "target": [value * 2 + (5 if value % 2 else 0) for value in range(20)],
        })
        analyst = MLExperimentAnalyst(data, target="target", config=ExperimentConfig(cv_folds=3))
        analyst.run_experiment()

        preprocessor = analyst._model_pipeline.named_steps["preprocessor"]
        self.assertIn("category", preprocessor.transformers_[1][2])
        self.assertEqual(
            analyst.profile()["categorical_columns"], ["category"]
        )

    def test_missing_numeric_and_categorical_values_are_imputed(self) -> None:
        data = pd.DataFrame({
            "numeric_feature": [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
            "category": ["a", None, "a", "b", "b", "a", "b", "a", "b", "a"],
            "target": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        })
        analyst = MLExperimentAnalyst(data, target="target", config=ExperimentConfig(cv_folds=2))
        report = analyst.run_experiment()

        self.assertEqual(report["dataset"]["missing_values"]["numeric_feature"], 1)
        self.assertEqual(report["dataset"]["missing_values"]["category"], 1)
        self.assertTrue(all(math.isfinite(value) for value in report["final_test_results"].values()))

    def test_duplicate_rows_are_reported(self) -> None:
        data = pd.concat([build_demo_dataset(), build_demo_dataset().iloc[[0]]], ignore_index=True)
        profile = MLExperimentAnalyst(data, target="sale_price").profile()

        self.assertEqual(profile["duplicate_rows"], 1)

    def test_non_finite_feature_values_are_rejected(self) -> None:
        data = pd.DataFrame({"feature": [1.0, float("inf"), 3.0], "target": [1.0, 2.0, 3.0]})

        with self.assertRaises(ValueError):
            MLExperimentAnalyst(data, target="target")

    def test_small_dataset_warning_is_reported(self) -> None:
        warnings = self.analyst.profile()["warnings"]

        self.assertTrue(any("very small" in warning for warning in warnings))
        self.assertTrue(any("high variance" in warning for warning in warnings))

    def test_larger_dataset_does_not_get_small_warning(self) -> None:
        data = pd.DataFrame({
            "feature_a": range(60),
            "feature_b": [value % 7 for value in range(60)],
            "target": [value * 1.5 + 10 for value in range(60)],
        })
        warnings = MLExperimentAnalyst(data, target="target").profile()["warnings"]

        self.assertFalse(any("very small" in warning for warning in warnings))

    def test_dataset_profiler_detects_feature_types_and_warnings(self) -> None:
        data = pd.DataFrame({
            "numeric_col": [1.0, 2.0, 3.0, 4.0],
            "category_col": ["a", "b", "a", "b"],
            "boolean_col": [True, False, True, False],
            "datetime_col": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "text_col": ["hello", "world", "hello", "again"],
            "id_col": [101, 102, 103, 104],
            "const_col": [1, 1, 1, 1],
            "target": [10.0, 20.0, 30.0, 40.0],
        })

        profile = DatasetProfiler(data, "target").profile()

        self.assertEqual(profile["rows"], 4)
        self.assertEqual(profile["columns"], 8)
        self.assertIn("numeric_col", profile["numeric_columns"])
        self.assertIn("category_col", profile["categorical_columns"])
        self.assertIn("boolean_col", profile["boolean_columns"])
        self.assertIn("datetime_col", profile["datetime_columns"])
        self.assertIn("text_col", profile["text_columns"])
        self.assertIn("id_col", profile["id_like_columns"])
        self.assertIn("const_col", profile["constant_columns"])
        self.assertIn("target", profile["target"]["column"])
        self.assertIsInstance(profile["warnings"], list)

    def test_task_detector_detects_regression_and_classification(self) -> None:
        regression_data = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [10.0, 11.0, 12.0, 13.0]})
        binary_data = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 1, 0]})
        multiclass_data = pd.DataFrame({"feature": [1, 2, 3, 4], "target": ["a", "b", "a", "c"]})

        self.assertEqual(TaskDetector.detect(regression_data, "target")["task"], "regression")
        self.assertEqual(TaskDetector.detect(binary_data, "target")["task"], "binary_classification")
        self.assertEqual(TaskDetector.detect(multiclass_data, "target")["task"], "multiclass_classification")

    def test_leakage_auditor_flags_target_in_features(self) -> None:
        data = pd.DataFrame({
            "feature": [1, 2, 3, 4],
            "target": [10, 20, 30, 40],
            "target_copy": [10, 20, 30, 40],
        })

        findings = LeakageAuditor(data, "target").audit()

        self.assertTrue(any(item["column"] == "target_copy" for item in findings))
        self.assertTrue(any(item["risk"] == "target_leakage" for item in findings))

    def test_validation_strategy_selector_reports_strategy_details(self) -> None:
        classification_data = pd.DataFrame({
            "feature_1": range(20),
            "feature_2": range(20),
            "target": [0, 1] * 10,
        })

        strategy = ValidationStrategySelector(classification_data, "target").select(cv_folds=5)

        self.assertEqual(strategy["strategy"], "stratified_kfold")
        self.assertEqual(strategy["n_splits"], 5)
        self.assertTrue(strategy["shuffle"])

    def test_custom_experiment_configuration_is_recorded(self) -> None:
        config = ExperimentConfig(
            test_size=0.25,
            validation_size=0.1,
            cv_folds=3,
            random_state=7,
            models=("dummy_regressor", "ridge_regression"),
        )
        report = MLExperimentAnalyst(
            build_demo_dataset(), target="sale_price", config=config
        ).run_experiment()

        self.assertEqual(report["split"]["test_size"], 0.25)
        self.assertEqual(report["configuration"]["cv_folds"], 3)
        self.assertEqual(report["configuration"]["random_state"], 7)
        self.assertEqual(report["candidate_models"], ["dummy_regressor", "ridge_regression"])

    def test_salary_csv_has_expected_real_data_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "salary_dataset.csv"
            pd.DataFrame({
                "Age": [25, 30, 45, 55],
                "Education": ["Bachelors", "Masters", "Doctorate", "Diploma"],
                "Salary": [43.9, 53.5, 77.5, 20.0],
            }).to_csv(csv_path, index=False)
            analyst = MLExperimentAnalyst.from_csv(str(csv_path), target="Salary")
            profile = analyst.profile()

        self.assertEqual(profile["rows"], 4)
        self.assertEqual(profile["feature_columns"], ["Age", "Education"])
        self.assertEqual(profile["categorical_columns"], ["Education"])
        self.assertEqual(profile["numerical_columns"], ["Age"])
        self.assertTrue(profile["target_suitability"]["suitable_for_regression"])

    def test_saved_model_artifact_loads_and_predicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path, metadata_path = self.analyst.save_model_artifact(directory)
            loaded_model = joblib.load(model_path)
            new_rows = build_demo_dataset().drop(columns=["sale_price"]).iloc[:2]
            predictions = loaded_model.predict(self.analyst.engineer_features(new_rows))

            self.assertTrue(model_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(len(predictions), 2)
            self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8"))["experiment_id"], self.analyst.experiment_id)


if __name__ == "__main__":
    unittest.main()