import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aitraining import (
    ExperimentComparator,
    ExperimentConfig,
    ExperimentLineage,
    ExperimentTracker,
    MLExperimentAnalyst,
    ResearchReportGenerator,
    RobustnessAnalyzer,
    StatisticalAnalyzer,
    build_demo_dataset,
)


class Phase4Tests(unittest.TestCase):
    def test_experiment_id_generation_and_metadata(self):
        analyst = MLExperimentAnalyst(build_demo_dataset(), "sale_price")
        self.assertRegex(analyst.experiment_id, r"^exp_\d{8}_\d{6}$")
        metadata = ExperimentTracker().build_metadata(analyst)
        self.assertEqual(metadata["experiment_id"], analyst.experiment_id)
        self.assertEqual(metadata["target_column"], "sale_price")
        self.assertIn("timestamp", metadata)

    def test_experiment_tracking_persists_json_artifacts(self):
        analyst = MLExperimentAnalyst(build_demo_dataset(), "sale_price")
        with tempfile.TemporaryDirectory() as directory:
            tracker = ExperimentTracker(artifact_dir=directory)
            saved = tracker.save(analyst, analyst.run_experiment())
            self.assertTrue(Path(saved["experiment_path"]).exists())
            self.assertTrue(Path(saved["report_path"]).exists())
            self.assertTrue(Path(saved["metadata_path"]).exists())
            loaded = tracker.load(saved["directory"])
            self.assertEqual(loaded["experiment_id"], analyst.experiment_id)

    def test_experiment_reload(self):
        analyst = MLExperimentAnalyst(build_demo_dataset(), "sale_price")
        with tempfile.TemporaryDirectory() as directory:
            experiment_entry = analyst.run_experiment()
            tracker = ExperimentTracker(artifact_dir=directory)
            tracker.save(analyst, experiment_entry)
            reloaded = MLExperimentAnalyst.load_experiment(Path(directory) / analyst.experiment_id)
            self.assertEqual(reloaded["experiment_id"], analyst.experiment_id)

    def test_experiment_comparison_regression(self):
        first = {
            "experiment_id": "exp_a",
            "final_test_results": {"rmse": 2.0, "mae": 1.5, "r2": 0.80},
            "task": {"task": "regression"},
        }
        second = {
            "experiment_id": "exp_b",
            "final_test_results": {"rmse": 1.0, "mae": 1.2, "r2": 0.90},
            "task": {"task": "regression"},
        }
        comparison = ExperimentComparator().compare_experiments(first, second)
        self.assertIn("metric_directions", comparison)
        self.assertIn("rmse", comparison["comparisons"])
        self.assertEqual(comparison["comparisons"]["rmse"]["better_metric"], "lower")

    def test_experiment_comparison_classification(self):
        first = {
            "experiment_id": "exp_a",
            "final_test_results": {"accuracy": 0.72, "f1": 0.70, "roc_auc": 0.74},
            "task": {"task": "binary_classification"},
        }
        second = {
            "experiment_id": "exp_b",
            "final_test_results": {"accuracy": 0.81, "f1": 0.77, "roc_auc": 0.85},
            "task": {"task": "binary_classification"},
        }
        comparison = ExperimentComparator().compare_experiments(first, second)
        self.assertEqual(comparison["comparisons"]["accuracy"]["better_metric"], "higher")
        self.assertGreater(comparison["comparisons"]["accuracy"]["delta"], 0)

    def test_metric_direction_handles_regression_and_classification(self):
        self.assertEqual(ExperimentComparator.metric_direction("rmse"), "lower")
        self.assertEqual(ExperimentComparator.metric_direction("r2"), "higher")
        self.assertEqual(ExperimentComparator.metric_direction("f1"), "higher")
        self.assertEqual(ExperimentComparator.metric_direction("mae"), "lower")

    def test_experiment_lineage_capture(self):
        analyst = MLExperimentAnalyst(build_demo_dataset(), "sale_price")
        report = analyst.run_experiment()
        lineage = ExperimentLineage(analyst).build(report)
        self.assertIn("dataset", lineage)
        self.assertIn("selected_model", lineage)
        self.assertIn("transformations", lineage)

    def test_robustness_analysis_with_multiple_seeds(self):
        seed_results = {
            42: {"rmse": 2.0, "mae": 1.0},
            123: {"rmse": 2.2, "mae": 1.2},
            456: {"rmse": 1.8, "mae": 1.1},
        }
        robustness = RobustnessAnalyzer().analyze(seed_results)
        self.assertEqual(robustness["n_seeds"], 3)
        self.assertIn("summary", robustness)
        self.assertIn("rmse", robustness["summary"])
        self.assertIn("status", robustness)

    def test_statistical_summary_uses_descriptive_statistics(self):
        analyzer = StatisticalAnalyzer()
        result = analyzer.summarize([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["median"], 2.5)
        self.assertAlmostEqual(result["range"], 3.0)

    def test_small_dataset_safety_warning(self):
        small = pd.DataFrame({"x": range(10), "y": range(10)})
        analyst = MLExperimentAnalyst(small, "y")
        report = analyst.run_experiment()
        self.assertTrue(any("small" in warning.lower() for warning in report["warnings"]))

    def test_reproducibility_metadata_is_recorded(self):
        analyst = MLExperimentAnalyst(build_demo_dataset(), "sale_price")
        report = analyst.run_experiment()
        self.assertIn("reproducibility", report)
        self.assertIn("python_version", report["reproducibility"])
        self.assertIn("random_seed", report["reproducibility"])

    def test_research_report_generation(self):
        analyst = MLExperimentAnalyst(build_demo_dataset(), "sale_price")
        report = analyst.run_experiment()
        research = ResearchReportGenerator(analyst).build(report)
        self.assertIn("executive_summary", research)
        self.assertIn("evidence_based_conclusion", research)
        self.assertIn("limitations", research)

    def test_config_validation_for_phase4_fields(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(robustness_enabled=True, robustness_seeds=[])
        with self.assertRaises(ValueError):
            ExperimentConfig(robustness_seeds=[1, 1])

    def test_backward_compatibility_for_existing_cli_and_api(self):
        analyst = MLExperimentAnalyst(build_demo_dataset(), "sale_price")
        self.assertIn("summary", analyst.profile())
        self.assertEqual(analyst.task, "regression")
        self.assertEqual(analyst.predict(pd.DataFrame({"area_sqft": [1000], "bedrooms": [2], "neighborhood": ["North"]})).iloc[0] >= 0, True)

    def test_cli_compatibility(self):
        import sys
        from unittest.mock import patch
        from aitraining import parse_args

        with patch.object(sys, 'argv', ['aitraining.py']):
            args = parse_args()
            self.assertIsNotNone(args)

    def test_json_serialization_of_report(self):
        report = MLExperimentAnalyst(build_demo_dataset(), "sale_price").run_experiment()
        json.dumps(report)

    def test_cli_phase4_flags(self):
        import sys
        from unittest.mock import patch
        from aitraining import parse_args
        test_args = ['aitraining.py', '--experiment-name', 'test_exp', '--no-track-experiment', '--robustness', '--robustness-seeds', '1,2,3', '--no-statistical-analysis', '--no-research-report']
        with patch.object(sys, 'argv', test_args):
            args = parse_args()
            self.assertEqual(args.experiment_name, 'test_exp')
            self.assertFalse(args.experiment_tracking)
            self.assertTrue(args.robustness_enabled)
            self.assertEqual(args.robustness_seeds, '1,2,3')
            self.assertFalse(args.statistical_analysis)
            self.assertFalse(args.research_report)

    def test_compare_directories(self):
        import tempfile
        from pathlib import Path
        from aitraining import MLExperimentAnalyst, build_demo_dataset, ExperimentConfig, ExperimentComparator
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dir1 = temp_path / "exp1"
            dir2 = temp_path / "exp2"
            
            # Create two experiments and save them
            ds = build_demo_dataset()
            analyst1 = MLExperimentAnalyst(ds, "sale_price", config=ExperimentConfig(experiment_name="Exp1", experiment_tracking=True, models=("linear_regression",)))
            analyst1.run_experiment()
            analyst1.save_model_artifact(dir1)
            actual_dir1 = dir1 / analyst1.experiment_id
            
            analyst2 = MLExperimentAnalyst(ds, "sale_price", config=ExperimentConfig(experiment_name="Exp2", experiment_tracking=True, models=("random_forest_regressor",)))
            analyst2.run_experiment()
            analyst2.save_model_artifact(dir2)
            actual_dir2 = dir2 / analyst2.experiment_id
            
            # Compare directories
            comparison = ExperimentComparator.compare_directories(actual_dir1, actual_dir2)
            self.assertIn("experiments", comparison)
            self.assertIn("comparisons", comparison)
            self.assertIn("tradeoffs_preserved", comparison)

if __name__ == "__main__":
    unittest.main()
