"""ML Experiment Analyst Agent: a first reproducible portfolio prototype."""

from dataclasses import dataclass, field, replace
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
	GradientBoostingClassifier,
	GradientBoostingRegressor,
	RandomForestClassifier,
	RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	confusion_matrix,
	f1_score,
	make_scorer,
	mean_absolute_error,
	mean_squared_error,
	precision_score,
	recall_score,
	roc_auc_score,
	r2_score,
)
from sklearn.model_selection import (
	KFold,
	RandomizedSearchCV,
	StratifiedKFold,
	cross_validate,
	train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _generate_experiment_id() -> str:
	"""Create a human-readable experiment identifier."""
	return datetime.now().strftime("exp_%Y%m%d_%H%M%S")


def _safe_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
	if len(actual) < 2 or np.allclose(np.var(actual), 0):
		return 0.0
	return float(r2_score(actual, predicted))


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
	mask = np.asarray(actual) != 0
	if not np.any(mask):
		return 0.0
	return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


DEFAULT_MODEL_NAMES = (
	"dummy_regressor",
	"linear_regression",
	"ridge_regression",
	"random_forest_regressor",
	"gradient_boosting_regressor",
)
DEFAULT_CLASSIFICATION_MODEL_NAMES = (
	"dummy_classifier",
	"logistic_regression",
	"random_forest_classifier",
	"gradient_boosting_classifier",
)
ALL_MODEL_NAMES = set(DEFAULT_MODEL_NAMES) | set(DEFAULT_CLASSIFICATION_MODEL_NAMES)
VALID_TASKS = {"auto", "regression", "classification"}
VALID_CV_STRATEGIES = {"auto", "kfold", "stratified", "timeseries", "group"}


class ColumnNameList(list):
	"""List-like column container that remains backward compatible with count checks."""

	def __eq__(self, other):
		if isinstance(other, int):
			return len(self) == other
		return super().__eq__(other)

	def __ne__(self, other):
		if isinstance(other, int):
			return len(self) != other
		return super().__ne__(other)


@dataclass(frozen=True)
class ExperimentConfig:
	test_size: float = 0.2
	validation_size: float = 0.2
	cv_folds: int = 5
	random_state: int = 42
	models: tuple[str, ...] = field(default_factory=lambda: DEFAULT_MODEL_NAMES)
	task: str = "auto"
	cv_strategy: str = "auto"
	enable_feature_engineering: bool = False
	feature_engineering_mode: str = "conservative"
	max_interaction_features: int = 3
	max_categorical_cardinality: int = 50
	tune: bool = False
	n_iter: int = 10
	tuning_cv: int | None = None
	tuning_scoring: str | None = None
	tuning_random_seed: int | None = None
	error_analysis: bool = True
	feature_importance: bool = True
	max_tuning_candidates: int = 100
	experiment_tracking: bool = True
	experiment_name: str | None = None
	robustness_enabled: bool = False
	robustness_seeds: tuple[int, ...] | None = None
	statistical_analysis: bool = True
	experiment_comparison: bool = True
	research_report: bool = True
	dataset_fingerprint: bool = True

	def __post_init__(self) -> None:
		if not 0 < self.test_size < 1:
			raise ValueError("test_size must be between 0 and 1")
		if not 0 < self.validation_size < 1:
			raise ValueError("validation_size must be between 0 and 1")
		if self.test_size + self.validation_size >= 1:
			raise ValueError("test_size + validation_size must be less than 1")
		if self.cv_folds < 2:
			raise ValueError("cv_folds must be at least 2")
		if self.task not in VALID_TASKS:
			raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
		if self.cv_strategy not in VALID_CV_STRATEGIES:
			raise ValueError(f"cv_strategy must be one of {sorted(VALID_CV_STRATEGIES)}")
		if self.feature_engineering_mode not in {"conservative", "none"}:
			raise ValueError("feature_engineering_mode must be 'conservative' or 'none'")
		if self.max_interaction_features < 0:
			raise ValueError("max_interaction_features must be non-negative")
		if self.max_categorical_cardinality < 2:
			raise ValueError("max_categorical_cardinality must be at least 2")
		if self.n_iter < 1:
			raise ValueError("n_iter must be at least 1")
		if self.tuning_cv is not None and self.tuning_cv < 2:
			raise ValueError("tuning_cv must be at least 2")
		if self.max_tuning_candidates < 1:
			raise ValueError("max_tuning_candidates must be at least 1")
		if self.robustness_seeds is not None:
			if not self.robustness_seeds:
				raise ValueError("robustness_seeds must contain at least one seed")
			if any(not isinstance(seed, int) for seed in self.robustness_seeds):
				raise ValueError("robustness_seeds must contain integers")
			if len(set(self.robustness_seeds)) != len(self.robustness_seeds):
				raise ValueError("robustness_seeds must not contain duplicates")
		unknown_models = set(self.models) - ALL_MODEL_NAMES
		if unknown_models:
			raise ValueError(f"Unknown model names: {sorted(unknown_models)}")
		if not self.models:
			raise ValueError("At least one model must be configured")
		if self.task == "classification" and self.cv_strategy == "timeseries":
			raise ValueError("timeseries validation is incompatible with classification without an explicit time-aware classifier workflow")


class FeatureTransformationRegistry:
	"""Stable names and descriptions for transformations emitted by the transformer."""

	TRANSFORMATIONS = {
		"log1p": {"group": "numeric", "description": "safe log1p for non-negative skewed features"},
		"datetime_components": {"group": "datetime", "description": "year, month, day, weekday, and quarter"},
		"limited_interactions": {"group": "numeric", "description": "bounded product and ratio interactions"},
		"one_hot": {"group": "categorical", "description": "one-hot encoding with unknown-category handling"},
	}

	@classmethod
	def describe(cls, names: list[str]) -> list[dict[str, str]]:
		return [{"name": name, **cls.TRANSFORMATIONS.get(name, {"group": "unknown", "description": ""})} for name in names]


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
	"""Apply deterministic, training-fold-fitted feature transformations."""

	def __init__(self, enabled: bool = False, mode: str = "conservative", max_interaction_features: int = 3) -> None:
		self.enabled = enabled
		self.mode = mode
		self.max_interaction_features = max_interaction_features

	def fit(self, X: pd.DataFrame, y=None):
		if not isinstance(X, pd.DataFrame):
			raise TypeError("Feature engineering requires a pandas DataFrame")
		self.input_columns_ = X.columns.tolist()
		self.datetime_columns_ = [
			column for column in X.columns
			if pd.api.types.is_datetime64_any_dtype(X[column]) or DatasetProfiler._is_datetime_like(X[column])
		]
		self.numeric_columns_ = X.select_dtypes(include="number").columns.tolist()
		self.log_columns_ = []
		if self.enabled and self.mode != "none":
			for column in self.numeric_columns_:
				values = X[column].dropna()
				if not values.empty and (values >= 0).all() and values.nunique() > 2 and abs(values.skew()) > 1:
					self.log_columns_.append(column)
			self.interaction_columns_ = self.numeric_columns_[: self.max_interaction_features]
		else:
			self.interaction_columns_ = []
		self.feature_plan_ = {
			"enabled": bool(self.enabled and self.mode != "none"),
			"mode": self.mode,
			"log_columns": list(self.log_columns_),
			"datetime_columns": list(self.datetime_columns_) if self.enabled else [],
			"interaction_columns": list(self.interaction_columns_),
			"limits": {"max_interaction_features": self.max_interaction_features},
			"requested_transformations": ["log1p", "datetime_components", "limited_interactions"],
			"skipped_transformations": [],
		}
		return self

	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		frame = X.copy()
		if not self.enabled or self.mode == "none":
			return frame
		for column in self.log_columns_:
			values = pd.to_numeric(frame[column], errors="coerce")
			frame[f"{column}__log1p"] = np.log1p(values.clip(lower=0))
		for column in self.datetime_columns_:
			parsed = pd.to_datetime(frame[column], errors="coerce")
			frame[f"{column}__year"] = parsed.dt.year
			frame[f"{column}__month"] = parsed.dt.month
			frame[f"{column}__day"] = parsed.dt.day
			frame[f"{column}__day_of_week"] = parsed.dt.dayofweek
			frame[f"{column}__quarter"] = parsed.dt.quarter
			frame = frame.drop(columns=[column])
		if len(self.interaction_columns_) >= 2:
			first, second = self.interaction_columns_[:2]
			frame[f"{first}__x__{second}"] = frame[first] * frame[second]
			denominator = frame[second].replace(0, np.nan)
			frame[f"{first}__div__{second}"] = frame[first] / denominator
		return frame

	def get_feature_names_out(self, input_features=None):
		features = list(self.input_columns_ if input_features is None else input_features)
		if not self.enabled or self.mode == "none":
			return np.asarray(features, dtype=object)
		for column in self.datetime_columns_:
			if column in features:
				features.remove(column)
			features.extend(
				f"{column}__{suffix}"
				for suffix in ("year", "month", "day", "day_of_week", "quarter")
			)
		for column in self.log_columns_:
			features.append(f"{column}__log1p")
		if len(self.interaction_columns_) >= 2:
			first, second = self.interaction_columns_[:2]
			features.extend([f"{first}__x__{second}", f"{first}__div__{second}"])
		return np.asarray(features, dtype=object)


class ModelRegistry:
	"""Centralized task-aware baseline model factory."""

	def __init__(self, task: str, random_state: int = 42):
		if task not in {"regression", "classification"}:
			raise ValueError("ModelRegistry task must be regression or classification")
		self.task = task
		self.random_state = random_state

	def names(self) -> tuple[str, ...]:
		return DEFAULT_MODEL_NAMES if self.task == "regression" else DEFAULT_CLASSIFICATION_MODEL_NAMES

	def factories(self) -> dict[str, object]:
		if self.task == "regression":
			return {
				"dummy_regressor": lambda: DummyRegressor(strategy="mean"),
				"linear_regression": LinearRegression,
				"ridge_regression": lambda: Ridge(alpha=1.0),
				"random_forest_regressor": lambda: RandomForestRegressor(n_estimators=100, random_state=self.random_state),
				"gradient_boosting_regressor": lambda: GradientBoostingRegressor(random_state=self.random_state),
			}
		return {
			"dummy_classifier": lambda: DummyClassifier(strategy="prior"),
			"logistic_regression": lambda: LogisticRegression(max_iter=500, random_state=self.random_state),
			"random_forest_classifier": lambda: RandomForestClassifier(n_estimators=100, random_state=self.random_state),
			"gradient_boosting_classifier": lambda: GradientBoostingClassifier(random_state=self.random_state),
		}

	def models(self, configured_names: tuple[str, ...] | None = None) -> dict[str, object]:
		names = self.names() if configured_names is None else configured_names
		factories = self.factories()
		return {name: factories[name]() for name in names if name in factories}

	def specifications(self) -> list[dict]:
		return [
			{
				"name": name,
				"task": self.task,
				"estimator": type(self.factories()[name]()).__name__,
				"default_parameters": self.factories()[name]().get_params(),
				"supported_hyperparameters": sorted(self.factories()[name]().get_params()),
			}
			for name in self.names()
		]

	def search_spaces(self) -> dict[str, dict[str, list]]:
		"""Return bounded, model-specific search spaces for Phase 3 tuning."""
		if self.task == "regression":
			return {
				"ridge_regression": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
				"random_forest_regressor": {
					"model__n_estimators": [50, 100, 200],
					"model__max_depth": [None, 3, 5, 10],
					"model__min_samples_leaf": [1, 2, 4],
				},
				"gradient_boosting_regressor": {
					"model__n_estimators": [50, 100, 200],
					"model__learning_rate": [0.03, 0.05, 0.1, 0.2],
					"model__max_depth": [2, 3, 5],
				},
			}
		return {
			"logistic_regression": {
				"model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
				"model__solver": ["lbfgs"],
			},
			"random_forest_classifier": {
				"model__n_estimators": [50, 100, 200],
				"model__max_depth": [None, 3, 5, 10],
				"model__min_samples_leaf": [1, 2, 4],
			},
			"gradient_boosting_classifier": {
				"model__n_estimators": [50, 100, 200],
				"model__learning_rate": [0.03, 0.05, 0.1, 0.2],
				"model__max_depth": [2, 3, 5],
			},
		}


class HyperparameterOptimizer:
	"""Run bounded, reproducible, pipeline-safe randomized searches."""

	def __init__(self, analyst: "MLExperimentAnalyst") -> None:
		self.analyst = analyst

	def run(
		self,
		x_train: pd.DataFrame,
		y_train: pd.Series,
		model_names: tuple[str, ...] | None = None,
	) -> dict[str, dict]:
		config = self.analyst.config
		seed = config.tuning_random_seed if config.tuning_random_seed is not None else self.analyst.random_state
		folds = config.tuning_cv or config.cv_folds
		if folds > len(x_train):
			folds = len(x_train)
		if folds < 2:
			raise ValueError("Tuning requires at least two training rows/folds")
		if self.analyst.task == "classification":
			class_count = int(y_train.value_counts().min())
			folds = min(folds, class_count)
			if folds < 2:
				raise ValueError("Classification tuning requires at least two samples per class")
			cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
			scoring = config.tuning_scoring or "f1_weighted"
		else:
			cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
			scoring = config.tuning_scoring or "neg_mean_absolute_error"
		spaces = self.analyst.model_registry.search_spaces()
		names = model_names or tuple(self.analyst._baseline_models())
		results: dict[str, dict] = {}
		for name in names:
			space = spaces.get(name)
			if not space:
				results[name] = {"status": "skipped", "reason": "No Phase 3 search space is defined for this model."}
				continue
			started = time.perf_counter()
			try:
				search = RandomizedSearchCV(
					estimator=self.analyst._pipeline(x_train, self.analyst._baseline_models()[name]),
					param_distributions=space,
					n_iter=min(config.n_iter, config.max_tuning_candidates),
					scoring=scoring,
					cv=cv,
					random_state=seed,
					n_jobs=1,
					return_train_score=False,
					error_score="raise",
				)
				search.fit(x_train, y_train)
				results[name] = {
					"status": "completed",
					"scoring": scoring,
					"cv_folds": folds,
					"n_iter": min(config.n_iter, config.max_tuning_candidates),
					"best_score": float(search.best_score_),
					"best_params": search.best_params_,
					"cv_results": {
						"mean_test_score": [float(value) for value in search.cv_results_["mean_test_score"]],
						"std_test_score": [float(value) for value in search.cv_results_["std_test_score"]],
					},
					"runtime_seconds": time.perf_counter() - started,
					"estimator": search.best_estimator_,
				}
			except Exception as error:
				results[name] = {
					"status": "failed",
					"error": str(error),
					"runtime_seconds": time.perf_counter() - started,
				}
		return results


class DatasetProfiler:
	"""Profile a dataset without transforming or discarding columns."""

	def __init__(self, data: pd.DataFrame, target: str) -> None:
		if not isinstance(data, pd.DataFrame):
			raise TypeError("data must be a pandas DataFrame")
		if data.empty:
			raise ValueError("Dataset is empty")
		if not target or target not in data.columns:
			raise ValueError(f"Target column '{target}' was not found. Available columns: {list(data.columns)}")
		self.data = data.copy()
		self.target = target

	@staticmethod
	def _column_missing_summary(column: pd.Series) -> dict[str, float | int]:
		missing_count = int(column.isna().sum())
		return {
			"missing_count": missing_count,
			"missing_percentage": float((missing_count / len(column)) * 100.0) if len(column) else 0.0,
		}

	@staticmethod
	def _is_boolean_like(series: pd.Series) -> bool:
		if pd.api.types.is_bool_dtype(series):
			return True
		non_null = series.dropna()
		if non_null.empty:
			return False
		values = non_null.astype(str).str.strip().str.lower()
		return set(values.unique()) <= {"true", "false", "1", "0", "yes", "no"}

	@staticmethod
	def _is_datetime_like(series: pd.Series) -> bool:
		if pd.api.types.is_datetime64_any_dtype(series):
			return True
		if pd.api.types.is_numeric_dtype(series):
			return False
		non_null = series.dropna()
		if non_null.empty:
			return False
		parsed = pd.to_datetime(non_null, errors="coerce")
		return float(parsed.notna().mean()) >= 0.9

	@staticmethod
	def _is_text_like(series: pd.Series) -> bool:
		if pd.api.types.is_numeric_dtype(series):
			return False
		non_null = series.dropna()
		if non_null.empty:
			return False
		string_like = non_null.map(lambda value: isinstance(value, str))
		if not string_like.all():
			return False
		return True

	@staticmethod
	def _is_id_like(series: pd.Series) -> bool:
		if series.nunique(dropna=True) <= 1:
			return False
		unique_ratio = series.nunique(dropna=True) / len(series.dropna()) if len(series.dropna()) else 0.0
		name = str(series.name).lower()
		return unique_ratio >= 0.95 or "id" in name or "uuid" in name

	def profile(self) -> dict:
		features = self.data.drop(columns=[self.target])
		target_values = self.data[self.target]
		feature_unique_counts = features.nunique(dropna=False)
		missing_values = {
			column: int(self.data[column].isna().sum()) for column in self.data.columns
		}
		missing_percentages = {
			column: float((self.data[column].isna().mean()) * 100.0) for column in self.data.columns
		}
		unique_counts = {
			column: int(self.data[column].nunique(dropna=False)) for column in self.data.columns
		}
		constant_columns = [
			column for column in features.columns if feature_unique_counts[column] <= 1
		]
		near_constant_columns = [
			column for column in features.columns
			if feature_unique_counts[column] <= 1
			or features[column].value_counts(dropna=False, normalize=True).iloc[0] >= 0.99
		]
		numeric_columns = features.select_dtypes(include="number").columns.tolist()
		categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
		boolean_columns = [
			column for column in features.columns if self._is_boolean_like(features[column])
		]
		datetime_columns = [
			column for column in features.columns if self._is_datetime_like(features[column])
		]
		text_columns = [
			column for column in features.columns if self._is_text_like(features[column])
		]
		id_like_columns = [
			column for column in features.columns if self._is_id_like(features[column])
		]
		warnings: list[str] = []
		if len(self.data) < 50:
			warnings.append("Dataset is very small; generalization estimates may be unstable.")
		if len(self.data) < max(50, 10 * len(features.columns)):
			warnings.append("Observation count is low relative to the number of features.")
		if len(self.data) / 5 < 10:
			warnings.append("Cross-validation estimates may have high variance with this dataset size.")
		if constant_columns:
			warnings.append(f"Constant feature columns detected: {constant_columns}.")
		if any(self.data[column].duplicated().any() for column in self.data.columns):
			warnings.append("Duplicate values were detected in one or more columns.")
		target_series = self.data[self.target]
		quartiles = target_series.quantile([0.25, 0.75]) if pd.api.types.is_numeric_dtype(target_series) else None
		lower_bound = None
		upper_bound = None
		if quartiles is not None and not quartiles.isna().any():
			interquartile_range = quartiles.iloc[1] - quartiles.iloc[0]
			if interquartile_range != 0:
				lower_bound = quartiles.iloc[0] - 1.5 * interquartile_range
				upper_bound = quartiles.iloc[1] + 1.5 * interquartile_range
				outlier_count = int(((target_series < lower_bound) | (target_series > upper_bound)).sum())
				if outlier_count:
					warnings.append(f"Target contains {outlier_count} possible outliers based on the IQR rule.")
		if pd.api.types.is_numeric_dtype(target_values):
			target_distribution = {
				"min": float(target_values.min()) if target_values.notna().any() else None,
				"max": float(target_values.max()) if target_values.notna().any() else None,
				"mean": float(target_values.mean()) if target_values.notna().any() else None,
				"median": float(target_values.median()) if target_values.notna().any() else None,
				"std": float(target_values.std(ddof=0)) if target_values.notna().any() else None,
			}
		else:
			target_distribution = {
				"min": None,
				"max": None,
				"mean": None,
				"median": None,
				"std": None,
				"class_counts": {str(key): int(value) for key, value in target_values.value_counts().items()},
			}
		return {
			"rows": len(self.data),
			"columns": ColumnNameList(self.data.columns),
			"column_count": len(self.data.columns),
			"feature_count": len(features.columns),
			"feature_columns": list(features.columns),
			"numerical_columns": numeric_columns,
			"numeric_columns": numeric_columns,
			"categorical_columns": categorical_columns,
			"boolean_columns": boolean_columns,
			"datetime_columns": datetime_columns,
			"text_columns": text_columns,
			"id_like_columns": id_like_columns,
			"constant_columns": constant_columns,
			"near_constant_columns": near_constant_columns,
			"missing_values": missing_values,
			"missing": missing_values,
			"missing_percentages": missing_percentages,
			"unique_counts": unique_counts,
			"cardinality": unique_counts,
			"duplicate_rows": int(self.data.duplicated().sum()),
			"duplicates": {
				"row_duplicates": int(self.data.duplicated().sum()),
				"column_duplicates": {
					column: int(self.data[column].duplicated().sum())
					for column in self.data.columns
				},
			},
			"infinite_values": {
				column: int(np.isinf(pd.to_numeric(self.data[column], errors="coerce").to_numpy(dtype=float)).sum())
				for column in self.data.columns
				if pd.api.types.is_numeric_dtype(self.data[column])
			},
			"target": {
				"column": self.target,
				"dtype": str(target_values.dtype),
				"n_unique": int(target_values.nunique(dropna=True)),
				"missing_count": int(target_values.isna().sum()),
				"missing_percentage": float((target_values.isna().mean()) * 100.0),
				"distribution": target_distribution,
			},
			"target_distribution": target_distribution,
			"suspicious_target_values": {
				"negative_count": int((target_values < 0).sum()) if pd.api.types.is_numeric_dtype(target_values) else 0,
				"zero_count": int((target_values == 0).sum()) if pd.api.types.is_numeric_dtype(target_values) else 0,
				"iqr_outlier_count": int(
					((target_values < lower_bound) | (target_values > upper_bound)).sum()
				) if quartiles is not None and not quartiles.isna().any() and interquartile_range != 0 else 0,
			},
			"invalid_numeric_values": {
				column: int(pd.to_numeric(self.data[column], errors="coerce").isna().sum())
				for column in self.data.select_dtypes(include="number").columns
			},
			"non_positive_numeric_values": {
				column: int((self.data[column] <= 0).sum())
				for column in self.data.select_dtypes(include="number").columns
			},
			"target_suitability": {
				"is_numeric": bool(pd.api.types.is_numeric_dtype(target_values)),
				"has_missing_values": bool(target_values.isna().any()),
				"has_finite_values": bool(np.isfinite(target_values.to_numpy(dtype=float)).all()) if pd.api.types.is_numeric_dtype(target_values) else True,
				"suitable_for_regression": True,
			},
			"target_type": str(self.data[self.target].dtype),
			"warnings": warnings,
		}


class TaskDetector:
	"""Detect the likely machine-learning task for a tabular dataset."""

	@staticmethod
	def _normalized_target_info(data: pd.DataFrame, target: str) -> tuple[pd.Series, bool, int, bool]:
		target_values = data[target]
		numerical = pd.api.types.is_numeric_dtype(target_values)
		unique_count = int(target_values.nunique(dropna=True))
		integer_like = False
		if numerical:
			values = pd.to_numeric(target_values, errors="coerce")
			integer_like = bool(np.isclose(values.to_numpy(dtype=float), np.round(values.to_numpy(dtype=float))).all())
		return target_values, numerical, unique_count, integer_like

	@classmethod
	def detect(cls, data: pd.DataFrame, target: str, task: str = "auto") -> dict:
		if task not in VALID_TASKS:
			raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
		target_values, is_numeric, n_unique, integer_like = cls._normalized_target_info(data, target)
		if task == "regression":
			return {
				"task": "regression",
				"target_type": "numeric" if is_numeric else "non_numeric",
				"n_unique": n_unique,
				"integer_like": integer_like,
				"reason": "Explicit regression override applied.",
			}
		if task == "classification":
			if n_unique <= 2:
				return {
					"task": "binary_classification",
					"target_type": "numeric" if is_numeric else "categorical",
					"n_unique": n_unique,
					"integer_like": integer_like,
					"reason": "Explicit classification override and a binary target were detected.",
				}
			return {
				"task": "multiclass_classification",
				"target_type": "categorical" if not is_numeric else "numeric",
				"n_unique": n_unique,
				"integer_like": integer_like,
				"reason": "Explicit classification override applied.",
			}
		if is_numeric and integer_like and n_unique == 2:
			return {
				"task": "binary_classification",
				"target_type": "numeric",
				"n_unique": n_unique,
				"integer_like": integer_like,
				"reason": "Numeric target with two discrete classes was interpreted as binary classification.",
			}
		if is_numeric and integer_like and 2 < n_unique <= min(20, max(2, int(len(target_values) * 0.2))):
			return {
				"task": "multiclass_classification",
				"target_type": "numeric",
				"n_unique": n_unique,
				"integer_like": integer_like,
				"reason": "Low-cardinality integer target was interpreted as discrete multiclass classification.",
			}
		if is_numeric and n_unique > 2:
			return {
				"task": "regression",
				"target_type": "numeric",
				"n_unique": n_unique,
				"integer_like": integer_like,
				"reason": "Numeric targets with more than two distinct values are treated as regressions by default.",
			}
		if n_unique == 2:
			return {
				"task": "binary_classification",
				"target_type": "categorical" if not is_numeric else "numeric",
				"n_unique": n_unique,
				"integer_like": integer_like,
				"reason": "Target with exactly two meaningful classes was interpreted as binary classification.",
			}
		if not is_numeric:
			return {
				"task": "multiclass_classification",
				"target_type": "categorical",
				"n_unique": n_unique,
				"integer_like": integer_like,
				"reason": "Non-numeric target was interpreted as a classification problem.",
			}
		return {
			"task": "regression",
			"target_type": "numeric",
			"n_unique": n_unique,
			"integer_like": integer_like,
			"reason": "Continuous numeric target was classified as regression.",
		}


class LeakageAuditor:
	"""Perform an explicit leakage audit for suspicious target-derived features."""

	def __init__(self, data: pd.DataFrame, target: str) -> None:
		self.data = data.copy()
		self.target = target

	@staticmethod
	def _normalized_name(value: str) -> str:
		return value.lower().replace("-", "_").replace(" ", "_")

	def audit(self) -> list[dict]:
		findings: list[dict] = []
		target_series = self.data[self.target]
		for column in self.data.columns:
			if column == self.target:
				continue
			feature = self.data[column]
			name_matches = self._normalized_name(column) == self._normalized_name(self.target) or (
				self._normalized_name(self.target) in self._normalized_name(column)
			)
			if name_matches:
				findings.append({
					"column": column,
					"risk": "target_leakage",
					"severity": "confirmed" if feature.equals(target_series) else "warning",
					"category": "target_derived_column",
					"reason": "Column name strongly resembles the target and may be target-derived.",
					"evidence": {"target": self.target, "column": column},
					"recommended_action": "Remove the column from training features and verify the source data definition.",
				})
			if feature.equals(target_series):
				findings.append({
					"column": column,
					"risk": "target_leakage",
					"severity": "confirmed",
					"category": "duplicate_target_information",
					"reason": "Column values are identical to the target column.",
					"evidence": {"correlation": 1.0, "column": column},
					"recommended_action": "Exclude the column from the feature set before training.",
				})
			if pd.api.types.is_numeric_dtype(feature) and pd.api.types.is_numeric_dtype(target_series):
				correlation = float(feature.astype(float).corr(target_series.astype(float)))
				if pd.notna(correlation) and abs(correlation) >= 0.999:
					findings.append({
						"column": column,
						"risk": "target_leakage",
						"severity": "warning",
						"category": "near_perfect_predictor",
						"reason": "Feature is nearly a perfect duplicate of the target signal.",
						"evidence": {"correlation": correlation},
						"recommended_action": "Inspect whether this column is a target-derived leakage source.",
					})
			if DatasetProfiler._is_id_like(feature):
				findings.append({
					"column": column,
					"risk": "review_required",
					"severity": "informational",
					"category": "id_like_column",
					"reason": "High-cardinality identifier-like feature may hurt generalization or encode row identity.",
					"evidence": {"unique_count": int(feature.nunique(dropna=True))},
					"recommended_action": "Review whether the identifier is available and meaningful at inference time.",
				})
		return findings

	def audit_report(self, feature_engineering_inside_pipeline: bool = True) -> dict:
		findings = self.audit()
		return {
			"findings": findings,
			"errors": [item for item in findings if item.get("severity") == "confirmed"],
			"warnings": [item for item in findings if item.get("severity") == "warning"],
			"informational": [item for item in findings if item.get("severity") == "informational"],
			"preprocessing_outside_cv": not feature_engineering_inside_pipeline,
			"target_encoding_outside_cv": False,
		}


class ValidationStrategySelector:
	"""Choose a validation strategy that is safe for the detected task and data shape."""

	def __init__(self, data: pd.DataFrame, target: str, task: str = "auto") -> None:
		self.data = data.copy()
		self.target = target
		self.task = task if task in {"binary_classification", "multiclass_classification"} else TaskDetector.detect(data, target, task=task)["task"]

	def select(self, cv_folds: int = 5, strategy: str = "auto", random_state: int = 42) -> dict:
		fallback_reason = None
		if strategy == "auto":
			if self.task in {"binary_classification", "multiclass_classification"}:
				selected_strategy = "stratified"
			elif any(pd.api.types.is_datetime64_any_dtype(self.data[column]) for column in self.data.columns):
				selected_strategy = "timeseries"
			else:
				selected_strategy = "kfold"
		else:
			selected_strategy = strategy
		if selected_strategy in {"stratified", "stratified_kfold"}:
			class_count = int(self.data[self.target].value_counts().min())
			n_splits = max(2, min(cv_folds, len(self.data), class_count))
			if n_splits != cv_folds:
				fallback_reason = f"Requested {cv_folds} folds; reduced to {n_splits} based on the smallest class count."
			return {
				"strategy": "stratified_kfold",
				"n_splits": n_splits,
				"shuffle": True,
				"random_state": random_state,
				"rationale": "Classification targets benefit from stratified splitting to preserve class proportions.",
				"fallback_reason": fallback_reason,
			}
		if selected_strategy == "timeseries":
			return {
				"strategy": "timeseries",
				"n_splits": max(2, cv_folds),
				"shuffle": False,
				"random_state": random_state,
				"rationale": "Time-aware validation prevents future observations from leaking into earlier folds.",
				"fallback_reason": fallback_reason,
			}
		if selected_strategy == "group":
			return {
				"strategy": "group_kfold",
				"n_splits": max(2, cv_folds),
				"shuffle": False,
				"random_state": random_state,
				"rationale": "Group-aware validation should be used when a grouping column is supplied to prevent leakage across groups.",
				"fallback_reason": fallback_reason,
			}
		return {
			"strategy": "kfold",
			"n_splits": max(2, cv_folds),
			"shuffle": True,
			"random_state": random_state,
			"rationale": "General-purpose K-fold validation was selected for the current dataset/task combination.",
			"fallback_reason": fallback_reason,
		}


@dataclass
class ExperimentResult:
	model_name: str
	mean_absolute_error: float
	mean_squared_error: float
	r2_score: float
	rmse: float
	mape: float
	task: str = "regression"
	metrics: dict[str, float] = field(default_factory=dict)


class StatisticalAnalyzer:
	"""Produce descriptive summaries without making significance claims."""

	def summarize(self, values: list[float] | np.ndarray, confidence: float = 0.95) -> dict:
		array = np.asarray(values, dtype=float)
		array = array[np.isfinite(array)]
		if array.size == 0:
			return {"count": 0, "warning": "No finite observations were available."}
		mean = float(np.mean(array))
		std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
		result = {
			"count": int(array.size),
			"mean": mean,
			"median": float(np.median(array)),
			"std": std,
			"variance": float(std ** 2),
			"min": float(np.min(array)),
			"max": float(np.max(array)),
			"range": float(np.max(array) - np.min(array)),
			"coefficient_of_variation": float(abs(std / mean)) if mean else None,
		}
		if array.size >= 2:
			margin = 1.96 * std / np.sqrt(array.size)
			result["confidence_interval"] = {"level": confidence, "lower": mean - margin, "upper": mean + margin}
		else:
			result["confidence_interval"] = None
			result["warning"] = "Only one observation is available; uncertainty cannot be estimated."
		return result

	def summarize_metrics(self, records: dict[int | str, dict[str, float]]) -> dict[str, dict]:
		metric_names = sorted({name for record in records.values() for name in record})
		return {name: self.summarize([record[name] for record in records.values() if name in record]) for name in metric_names}


class RobustnessAnalyzer:
	"""Summarize controlled repeated-seed evaluations using cautious language."""

	def __init__(self, statistics: StatisticalAnalyzer | None = None) -> None:
		self.statistics = statistics or StatisticalAnalyzer()

	def analyze(self, seed_results: dict[int | str, dict[str, float]]) -> dict:
		summary = self.statistics.summarize_metrics(seed_results)
		variable = any(item.get("coefficient_of_variation") is not None and item["coefficient_of_variation"] > 0.1 for item in summary.values())
		status = "variable across tested seeds" if variable else "stable across tested seeds"
		if len(seed_results) < 3:
			status = "insufficient evidence of stability"
		return {"enabled": True, "n_seeds": len(seed_results), "seeds": [int(seed) for seed in seed_results], "results": seed_results, "summary": summary, "status": status}


class ExperimentLineage:
	"""Build the auditable data-to-report chain from existing analyst outputs."""

	def __init__(self, analyst: "MLExperimentAnalyst") -> None:
		self.analyst = analyst

	def build(self, experiment: dict) -> dict:
		return {
			"dataset": {"source": self.analyst.source, "target": self.analyst.target, "fingerprint": _dataset_fingerprint(self.analyst.data)},
			"stages": ["dataset", "profiling", "task_detection", "leakage_audit", "validation_strategy", "feature_engineering", "baseline_models", "tuning", "selected_model", "error_analysis", "feature_importance", "final_report"],
			"transformations": experiment.get("feature_engineering_plan", {}).get("transformations_applied", []),
			"generated_features": experiment.get("feature_engineering_plan", {}).get("feature_name_lineage", {}),
			"selected_model": experiment.get("model_selection", {}).get("selected_model"),
			"model_parameters": experiment.get("parameters", {}).get("models", {}).get(experiment.get("model_selection", {}).get("selected_model"), {}),
			"random_seed": self.analyst.random_state,
			"validation": experiment.get("validation_strategy", {}),
		}


class ExperimentComparator:
	"""Compare experiments while preserving metric direction and trade-offs."""

	LOWER_IS_BETTER = {"mae", "mse", "rmse", "mape"}

	@classmethod
	def metric_direction(cls, metric: str) -> str:
		return "lower" if metric.lower() in cls.LOWER_IS_BETTER else "higher"

	def compare_experiments(self, first: dict, second: dict) -> dict:
		first_metrics = first.get("final_test_results", {})
		second_metrics = second.get("final_test_results", {})
		comparisons = {}
		for metric in sorted(set(first_metrics) & set(second_metrics)):
			left, right = float(first_metrics[metric]), float(second_metrics[metric])
			comparisons[metric] = {
				"first": left, "second": right, "delta": right - left,
				"direction": self.metric_direction(metric),
				"better_metric": self.metric_direction(metric),
				"better_experiment": (first.get("experiment_id") if left < right else second.get("experiment_id")) if self.metric_direction(metric) == "lower" else (first.get("experiment_id") if left > right else second.get("experiment_id")),
			}
		return {"experiments": [first.get("experiment_id"), second.get("experiment_id")], "task": first.get("task"), "metric_directions": {metric: self.metric_direction(metric) for metric in comparisons}, "comparisons": comparisons, "tradeoffs_preserved": True}

	@classmethod
	def compare_directories(cls, first_dir: str | Path, second_dir: str | Path) -> dict:
		"""Load and compare two saved experiments from their artifact directories."""
		first_exp = ExperimentTracker().load(first_dir)
		second_exp = ExperimentTracker().load(second_dir)
		return cls().compare_experiments(first_exp, second_exp)


class ResearchReportGenerator:
	"""Assemble cautious, evidence-based research sections from an experiment."""

	def __init__(self, analyst: "MLExperimentAnalyst") -> None:
		self.analyst = analyst

	def build(self, experiment: dict) -> dict:
		selected = experiment.get("model_selection", {}).get("selected_model")
		metrics = experiment.get("final_test_results", {})
		stability = experiment.get("robustness", {}).get("status", "single-run result")
		return {
			"executive_summary": f"Tested {len(experiment.get('candidate_models', []))} candidate models for {experiment.get('task', {}).get('task', self.analyst.task)}; selected {selected}.",
			"research_question": "Which configured candidate provides the strongest validation evidence for this dataset and task?",
			"dataset_overview": experiment.get("dataset", {}),
			"task_detection": experiment.get("task", {}),
			"data_quality": experiment.get("profiler", {}),
			"leakage_audit": experiment.get("leakage_analysis", {}),
			"validation_strategy": experiment.get("validation_strategy", {}),
			"feature_engineering": experiment.get("feature_engineering_plan", {}),
			"baseline_models": experiment.get("cross_validation", {}).get("models", {}),
			"hyperparameter_optimization": experiment.get("hyperparameter_optimization", {}),
			"model_comparison": experiment.get("model_comparison", {}),
			"final_model": selected,
			"final_test_results": metrics,
			"error_analysis": experiment.get("error_analysis", {}),
			"feature_importance": experiment.get("feature_importance", {}),
			"robustness_analysis": experiment.get("robustness", {"status": "single-run result"}),
			"statistical_summary": experiment.get("statistics", {}),
			"experiment_lineage": experiment.get("lineage", {}),
			"limitations": experiment.get("limitations", []),
			"reproducibility": experiment.get("reproducibility", {}),
			"evidence_based_conclusion": f"The selected model was supported by the reported validation and holdout metrics ({metrics}). Results are {stability}; the listed dataset and validation limitations prevent broader generalization claims.",
		}


def _dataset_fingerprint(data: pd.DataFrame) -> str:
	return hashlib.sha256(pd.util.hash_pandas_object(data, index=True).values.tobytes()).hexdigest()


class ExperimentTracker:
	"""Persist local JSON experiment records without external services."""

	def __init__(self, artifact_dir: str = "artifacts") -> None:
		self.artifact_dir = Path(artifact_dir)

	def build_metadata(self, analyst: "MLExperimentAnalyst", experiment: dict | None = None) -> dict:
		return {
			"experiment_id": analyst.experiment_id,
			"experiment_name": analyst.config.experiment_name,
			"timestamp": (experiment or {}).get("timestamp", datetime.now(timezone.utc).isoformat()),
			"task": analyst.task,
			"dataset": analyst.source,
			"dataset_fingerprint": _dataset_fingerprint(analyst.data) if analyst.config.dataset_fingerprint else None,
			"target_column": analyst.target,
			"row_count": len(analyst.data),
			"feature_count": len(analyst.data.columns) - 1,
			"configuration": {key: value for key, value in analyst.config.__dict__.items()},
			"random_seed": analyst.random_state,
		}

	def save(self, analyst: "MLExperimentAnalyst", experiment: dict) -> dict:
		directory = self.artifact_dir / analyst.experiment_id
		directory.mkdir(parents=True, exist_ok=True)
		metadata = self.build_metadata(analyst, experiment)
		paths = {"directory": str(directory), "experiment_path": str(directory / "experiment.json"), "report_path": str(directory / "report.json"), "metadata_path": str(directory / "metadata.json")}
		Path(paths["experiment_path"]).write_text(json.dumps(experiment, indent=2), encoding="utf-8")
		Path(paths["report_path"]).write_text(json.dumps(ResearchReportGenerator(analyst).build(experiment), indent=2), encoding="utf-8")
		Path(paths["metadata_path"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
		return {**paths, "experiment_id": analyst.experiment_id}

	def load(self, experiment_directory: str | Path) -> dict:
		return json.loads((Path(experiment_directory) / "experiment.json").read_text(encoding="utf-8"))


class MLExperimentAnalyst:
	"""Profile tabular data, run baselines, and recommend the next experiment."""

	def __init__(
		self,
		data: pd.DataFrame,
		target: str,
		random_state: int = 42,
		source: str = "in_memory",
		config: ExperimentConfig | None = None,
	) -> None:
		self.config = config or ExperimentConfig(random_state=random_state)
		self._validate_input_data(data, target, self.config.task)
		self.data = data.copy()
		self.target = target
		self.random_state = self.config.random_state
		self.source = source
		self.results: list[ExperimentResult] = []
		self._model_pipeline = None
		self._feature_columns: list[str] = []
		self._split: dict[str, int] | None = None
		self._last_experiment: dict | None = None
		self.experiment_id = _generate_experiment_id()
		self._selected_model_name: str | None = None
		self._classification_cv_warning: str | None = None
		self._tuning_results: dict[str, dict] = {}
		self._tuned_estimators: dict[str, Pipeline] = {}
		self.tracker = ExperimentTracker()
		self.statistical_analyzer = StatisticalAnalyzer()
		self.profiler = DatasetProfiler(self.data, self.target)
		self.task_detector = TaskDetector()
		self.task_info = TaskDetector.detect(self.data, self.target, task=self.config.task)
		self.task = "classification" if self.task_info["task"].endswith("classification") else "regression"
		if self.task == "classification" and self.data[target].nunique(dropna=True) < 2:
			raise ValueError("Classification requires at least two target classes")
		self.model_registry = ModelRegistry(self.task, random_state=self.random_state)
		if self.config.models != DEFAULT_MODEL_NAMES and not set(self.config.models).intersection(self.model_registry.names()):
			raise ValueError(f"Configured models are incompatible with task '{self.task}': {self.config.models}")
		self.leakage_auditor = LeakageAuditor(self.data, self.target)
		self.validation_strategy_selector = ValidationStrategySelector(self.data, self.target, task=self.task_info["task"])

	@staticmethod
	def _validate_input_data(data: pd.DataFrame, target: str, task: str = "auto") -> None:
		if not isinstance(data, pd.DataFrame):
			raise TypeError("data must be a pandas DataFrame")
		if data.empty:
			raise ValueError("Dataset is empty")
		if not target or target not in data.columns:
			raise ValueError(
				f"Target column '{target}' was not found. Available columns: {list(data.columns)}"
			)
		if len(data.columns) < 2:
			raise ValueError("Dataset must contain the target and at least one feature column")
		if data[target].isna().any():
			raise ValueError(f"Target column '{target}' contains missing values")
		if task != "classification" and not pd.api.types.is_numeric_dtype(data[target]):
			raise ValueError(f"Target column '{target}' must be numeric")
		if pd.api.types.is_numeric_dtype(data[target]) and not np.isfinite(data[target].to_numpy(dtype=float)).all():
			raise ValueError(f"Target column '{target}' contains non-finite values")
		all_null_features = [
			column for column in data.columns
			if column != target and data[column].isna().all()
		]
		if all_null_features:
			raise ValueError(f"Feature columns contain only missing values: {all_null_features}")
		numeric_features = data.drop(columns=[target]).select_dtypes(include="number")
		non_finite_features = [
			column for column in numeric_features.columns
			if np.isinf(numeric_features[column].dropna().to_numpy(dtype=float)).any()
		]
		if non_finite_features:
			raise ValueError(f"Numeric feature columns contain non-finite values: {non_finite_features}")

	def _model_factories(self) -> dict[str, object]:
		return self.model_registry.factories()

	def _baseline_models(self) -> dict[str, object]:
		configured_names = self.config.models
		if configured_names == DEFAULT_MODEL_NAMES and self.task == "classification":
			configured_names = DEFAULT_CLASSIFICATION_MODEL_NAMES
		return self.model_registry.models(configured_names)

	def _main_model(self) -> object:
		return self._baseline_models()[self.model_registry.names()[-1]]

	def _score_predictions(self, actual: pd.Series, predicted: np.ndarray, scores: np.ndarray | None = None) -> dict[str, float]:
		if self.task == "classification":
			metrics = {
				"accuracy": float(accuracy_score(actual, predicted)),
				"balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
				"precision": float(precision_score(actual, predicted, average="weighted", zero_division=0)),
				"recall": float(recall_score(actual, predicted, average="weighted", zero_division=0)),
				"f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
			}
			if scores is not None:
				try:
					if len(np.unique(actual)) == 2:
						metrics["roc_auc"] = float(roc_auc_score(actual, scores[:, 1] if scores.ndim == 2 else scores))
					elif scores.ndim == 2:
						metrics["roc_auc"] = float(roc_auc_score(actual, scores, multi_class="ovr"))
				except ValueError:
					pass
			return metrics
		actual_array = np.asarray(actual, dtype=float)
		predicted_array = np.asarray(predicted, dtype=float)
		mae = mean_absolute_error(actual_array, predicted_array)
		rmse = np.sqrt(mean_squared_error(actual_array, predicted_array))
		r2 = _safe_r2(actual_array, predicted_array)
		mask = actual_array != 0
		mape = float(
			np.mean(np.abs((actual_array[mask] - predicted_array[mask]) / actual_array[mask]))
		) if np.any(mask) else 0.0
		return {
			"mae": float(mae),
			"mse": float(mean_squared_error(actual_array, predicted_array)),
			"rmse": float(rmse),
			"r2": float(r2),
			"mape": float(mape),
		}

	def tune_models(
		self,
		model_names: tuple[str, ...] | None = None,
	) -> dict[str, dict]:
		"""Tune candidates on the training portion and retain fitted search results."""
		x_train, _, _, y_train, _, _ = self.prepare_data_splits()
		results = HyperparameterOptimizer(self).run(x_train, y_train, model_names)
		self._tuning_results = results
		self._tuned_estimators = {
			name: result["estimator"]
			for name, result in results.items()
			if result.get("status") == "completed" and "estimator" in result
		}
		return {
			name: {key: value for key, value in result.items() if key != "estimator"}
			for name, result in results.items()
		}

	def _error_analysis(
		self,
		actual: pd.Series,
		predicted: np.ndarray,
	) -> dict:
		"""Create row-level and aggregate error diagnostics for the final holdout."""
		actual_values = np.asarray(actual)
		predicted_values = np.asarray(predicted)
		rows = []
		if self.task == "classification":
			labels = sorted({str(value) for value in actual_values} | {str(value) for value in predicted_values})
			matrix = confusion_matrix(
				actual_values.astype(str), predicted_values.astype(str), labels=labels,
			).tolist()
			for index, (observed, estimate) in enumerate(zip(actual_values, predicted_values)):
				rows.append({
					"index": int(actual.index[index]) if hasattr(actual.index[index], "__int__") else str(actual.index[index]),
					"actual": str(observed),
					"predicted": str(estimate),
					"correct": bool(observed == estimate),
				})
			return {
				"task": "classification",
				"rows": rows,
				"error_count": sum(not row["correct"] for row in rows),
				"confusion_matrix": matrix,
				"labels": labels,
				"per_class_metrics": {
					label: {
						"precision": float(value),
						"recall": float(recall),
						"f1": float(f1),
					}
					for label, value, recall, f1 in zip(
						labels,
						precision_score(actual_values.astype(str), predicted_values.astype(str), labels=labels, average=None, zero_division=0),
						recall_score(actual_values.astype(str), predicted_values.astype(str), labels=labels, average=None, zero_division=0),
						f1_score(actual_values.astype(str), predicted_values.astype(str), labels=labels, average=None, zero_division=0),
					)
				},
			}
		residuals = actual_values.astype(float) - predicted_values.astype(float)
		for index, observed, estimate, residual in zip(actual.index, actual_values, predicted_values, residuals):
			rows.append({
				"index": int(index) if hasattr(index, "__int__") else str(index),
				"actual": float(observed),
				"predicted": float(estimate),
				"residual": float(residual),
				"absolute_error": float(abs(residual)),
				"squared_error": float(residual ** 2),
			})
		return {
			"task": "regression",
			"rows": rows,
			"summary": {
				"mean_residual": float(np.mean(residuals)),
				"mean_absolute_error": float(np.mean(np.abs(residuals))),
				"mean_squared_error": float(np.mean(residuals ** 2)),
				"max_absolute_error": float(np.max(np.abs(residuals))),
			},
		}

	def analyze_errors(self, actual: pd.Series, predicted: np.ndarray) -> dict:
		"""Public wrapper for structured regression/classification error analysis."""
		return self._error_analysis(actual, predicted)

	def _feature_importance(self, pipeline: Pipeline) -> dict:
		"""Return model-native importance values mapped to transformed feature names."""
		model = pipeline.named_steps["model"]
		preprocessor = pipeline.named_steps["preprocessor"]
		if hasattr(model, "feature_importances_"):
			values = np.asarray(model.feature_importances_, dtype=float)
			importance_type = "feature_importances_"
		elif hasattr(model, "coef_"):
			coefficients = np.asarray(model.coef_, dtype=float)
			values = np.abs(coefficients if coefficients.ndim == 1 else coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
			importance_type = "absolute_coefficients"
		else:
			return {"available": False, "type": "unavailable", "features": []}
		feature_names = preprocessor.get_feature_names_out().tolist()
		lineage = self._feature_engineering_report(pipeline).get("feature_name_lineage", {})
		order = np.argsort(values)[::-1]
		return {
			"available": True,
			"type": importance_type,
			"features": [
				{
					"feature": feature_names[index],
					"source_feature": lineage.get(feature_names[index], feature_names[index]),
					"importance": float(values[index]),
				}
				for index in order
				if index < len(feature_names)
			],
		}

	def analyze_feature_importance(self, pipeline: Pipeline | None = None) -> dict:
		"""Return importance for the fitted model or a supplied fitted pipeline."""
		active_pipeline = pipeline or self._model_pipeline
		if active_pipeline is None:
			self.fit_best_model()
			active_pipeline = self._model_pipeline
		return self._feature_importance(active_pipeline)

	def engineer_features(self, frame: pd.DataFrame) -> pd.DataFrame:
		"""Return source features unchanged; preprocessing is dataset-agnostic."""
		return frame.copy()

	def prepare_data_splits(
		self,
		test_size: float | None = None,
		validation_size: float | None = None,
		) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
		"""Create deterministic train/validation/test splits with the test set isolated."""
		test_size = self.config.test_size if test_size is None else test_size
		validation_size = self.config.validation_size if validation_size is None else validation_size
		if not 0 < test_size < 1 or not 0 < validation_size < 1:
			raise ValueError("split sizes must be between 0 and 1")
		if test_size + validation_size >= 1:
			raise ValueError("test_size + validation_size must be less than 1")

		prepared = self.engineer_features(self.data)
		x = prepared.drop(columns=[self.target])
		y = prepared[self.target]
		x_dev, x_test, y_dev, y_test = train_test_split(
			x, y,
			test_size=test_size,
			random_state=self.random_state,
			stratify=y if self.task == "classification" else None,
		)
		remaining_fraction = 1.0 - test_size
		val_fraction = validation_size / remaining_fraction
		x_train, x_val, y_train, y_val = train_test_split(
			x_dev,
			y_dev,
			test_size=val_fraction,
			random_state=self.random_state,
			stratify=y_dev if self.task == "classification" and y_dev.value_counts().min() >= 2 else None,
		)
		self._split = {
			"full_rows": len(self.data),
			"development_rows": len(x_dev),
			"train_rows": len(x_train),
			"validation_rows": len(x_val),
			"test_rows": len(x_test),
			"test_size": test_size,
			"validation_size": validation_size,
			"random_state": self.random_state,
			"development_fraction": 1.0 - test_size,
		}
		return x_train, x_val, x_test, y_train, y_val, y_test

	def _preprocessor(self, features: pd.DataFrame) -> ColumnTransformer:
		if self.config.enable_feature_engineering:
			numeric_features = make_column_selector(dtype_include=np.number)
			categorical_features = make_column_selector(dtype_exclude=np.number)
		else:
			numeric_features = features.select_dtypes(include="number").columns.tolist()
			categorical_features = [
				column for column in features.columns if column not in numeric_features and column != self.target
			]

		numeric_pipeline = Pipeline([
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		])
		categorical_pipeline = Pipeline([
			("imputer", SimpleImputer(strategy="most_frequent")),
			("one_hot", OneHotEncoder(
				handle_unknown="ignore",
				drop="first",
				max_categories=self.config.max_categorical_cardinality,
			)),
		])

		return ColumnTransformer([
			("numeric", numeric_pipeline, numeric_features),
			("categorical", categorical_pipeline, categorical_features),
		])

	def _pipeline(self, features: pd.DataFrame, model: object) -> Pipeline:
		steps = []
		if self.config.enable_feature_engineering:
			steps.append(("feature_engineering", FeatureEngineeringTransformer(
				enabled=True,
				mode=self.config.feature_engineering_mode,
				max_interaction_features=self.config.max_interaction_features,
			)))
		steps.extend([
			("preprocessor", self._preprocessor(features)),
			("model", model),
		])
		return Pipeline(steps)

	@classmethod
	def from_csv(
		cls,
		csv_path: str,
		target: str,
		config: ExperimentConfig | None = None,
	) -> "MLExperimentAnalyst":
		"""Load a local CSV and create an analyst with source provenance."""
		path = Path(csv_path).expanduser()
		if not path.is_file():
			raise FileNotFoundError(f"Dataset file was not found: {path}")
		try:
			data = pd.read_csv(path)
		except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
			raise ValueError(f"Could not read CSV dataset '{path}': {error}") from error
		return cls(data, target, config=config, source=str(path.resolve()))

	def profile(self) -> dict:
		profile = self.profiler.profile()
		task_info = TaskDetector.detect(self.data, self.target, task=self.config.task)
		profile["task"] = task_info
		profile["task_detection"] = task_info
		profile["source"] = self.source
		profile["filename"] = Path(self.source).name if self.source != "in_memory" else None
		profile["validation_strategy"] = ValidationStrategySelector(
			self.data,
			self.target,
			task=self.config.task,
		).select(cv_folds=self.config.cv_folds, strategy=self.config.cv_strategy, random_state=self.random_state)
		profile["leakage_audit"] = self.leakage_auditor.audit()
		profile["leakage_analysis"] = self.leakage_auditor.audit_report(
			feature_engineering_inside_pipeline=True,
		)
		profile["summary"] = {"rows": len(self.data), "columns": len(self.data.columns)}
		return profile

	def run_baselines(self) -> list[ExperimentResult]:
		x_train, x_val, _, y_train, y_val, _ = self.prepare_data_splits()
		self.results = []
		for name, model in self._baseline_models().items():
			pipeline = self._pipeline(x_train, model)
			pipeline.fit(x_train, y_train)
			predictions = pipeline.predict(x_val)
			metrics = self._score_predictions(y_val, predictions)
			if self.task == "classification":
				self.results.append(ExperimentResult(
					model_name=name,
					mean_absolute_error=1.0 - metrics["f1"],
					mean_squared_error=0.0,
					r2_score=0.0,
					rmse=0.0,
					mape=0.0,
					task="classification",
					metrics=metrics,
				))
			else:
				self.results.append(ExperimentResult(
					model_name=name,
					mean_absolute_error=metrics["mae"],
					mean_squared_error=metrics["mse"],
					r2_score=metrics["r2"],
					rmse=metrics["rmse"],
					mape=metrics["mape"],
				))
		return self.results

	def evaluate_candidate_models_cv(
		self,
		x_train: pd.DataFrame,
		y_train: pd.Series,
		folds: int | None = None,
		parallel: bool = False,
	) -> dict[str, dict[str, float]]:
		"""Use cross-validation on the training portion only to select the best candidate."""
		folds = self.config.cv_folds if folds is None else folds
		if folds > len(x_train):
			folds = len(x_train)
		if folds < 2:
			raise ValueError("folds must be between 2 and the number of training rows")

		if self.task == "classification":
			class_counts = y_train.value_counts()
			available_folds = int(class_counts.min())
			fold_count = min(folds, available_folds if available_folds >= 2 else int(self.data[self.target].value_counts().min()))
			if fold_count < 2:
				raise ValueError("Classification cross-validation requires at least two total folds")
			if fold_count != folds:
				self._classification_cv_warning = f"Requested {folds} folds; reduced to {fold_count} because of class counts."
			cv = (
				StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=self.random_state)
				if available_folds >= 2
				else KFold(n_splits=fold_count, shuffle=True, random_state=self.random_state)
			)
			self._effective_cv_folds = fold_count
			return self._evaluate_classification_cv(x_train, y_train, cv, parallel)
		self._effective_cv_folds = folds
		report: dict[str, dict[str, float]] = {}
		for name, model in self._baseline_models().items():
			try:
				pipeline = self._pipeline(x_train, model)
				scores = cross_validate(
					pipeline,
					x_train,
					y_train,
					cv=folds,
					scoring={
						"mae": "neg_mean_absolute_error",
						"mse": "neg_mean_squared_error",
						"rmse": "neg_root_mean_squared_error",
						"r2": make_scorer(_safe_r2),
						"mape": make_scorer(_safe_mape, greater_is_better=False),
					},
					n_jobs=-1 if parallel else 1,
				)
				report[name] = {
					"mean_cv_mae": float(-scores["test_mae"].mean()),
					"std_cv_mae": float(scores["test_mae"].std()),
					"mean_cv_rmse": float(-scores["test_rmse"].mean()),
					"std_cv_rmse": float(scores["test_rmse"].std()),
					"mean_cv_mse": float(-scores["test_mse"].mean()),
					"std_cv_mse": float(scores["test_mse"].std()),
					"mean_cv_r2": float(scores["test_r2"].mean()),
					"std_cv_r2": float(scores["test_r2"].std()),
					"mean_cv_mape": float(-scores["test_mape"].mean()),
					"std_cv_mape": float(scores["test_mape"].std()),
				}
			except Exception as error:
				report[name] = {
					"task": "regression",
					"training_status": "failed",
					"error": str(error),
					"warnings": ["Model failed during cross-validation."],
				}
		return report

	def _evaluate_classification_cv(
		self,
		x_train: pd.DataFrame,
		y_train: pd.Series,
		cv: StratifiedKFold,
		parallel: bool = False,
	) -> dict[str, dict[str, float]]:
		report: dict[str, dict[str, float]] = {}
		scoring = {
			"accuracy": "accuracy",
			"balanced_accuracy": "balanced_accuracy",
			"precision": make_scorer(precision_score, average="weighted", zero_division=0),
			"recall": make_scorer(recall_score, average="weighted", zero_division=0),
			"f1": make_scorer(f1_score, average="weighted", zero_division=0),
		}
		for name, model in self._baseline_models().items():
			try:
				pipeline = self._pipeline(x_train, model)
				scores = cross_validate(
					pipeline,
					x_train,
					y_train,
					cv=cv,
					scoring=scoring,
					n_jobs=-1 if parallel else 1,
					error_score=np.nan,
				)
				record = {
					"task": "classification",
					"training_status": "completed",
					"mean_cv_accuracy": float(np.nanmean(scores["test_accuracy"])),
					"std_cv_accuracy": float(np.nanstd(scores["test_accuracy"])),
					"mean_cv_balanced_accuracy": float(np.nanmean(scores["test_balanced_accuracy"])),
					"std_cv_balanced_accuracy": float(np.nanstd(scores["test_balanced_accuracy"])),
					"mean_cv_precision": float(np.nanmean(scores["test_precision"])),
					"std_cv_precision": float(np.nanstd(scores["test_precision"])),
					"mean_cv_recall": float(np.nanmean(scores["test_recall"])),
					"std_cv_recall": float(np.nanstd(scores["test_recall"])),
					"mean_cv_f1": float(np.nanmean(scores["test_f1"])),
					"std_cv_f1": float(np.nanstd(scores["test_f1"])),
					"mean_cv_score": float(np.nanmean(scores["test_f1"])),
					"cv_std": float(np.nanstd(scores["test_f1"])),
				}
			except Exception as error:
				record = {
					"task": "classification",
					"training_status": "failed",
					"error": str(error),
					"warnings": ["Model failed during cross-validation."],
				}
			report[name] = record
		return report

	def cross_validate_baselines(
		self, folds: int | None = None, parallel: bool = False
	) -> dict[str, dict[str, float]]:
		"""Evaluate candidates on the training portion of development data only."""
		x_train, _, _, y_train, _, _ = self.prepare_data_splits()
		cv_report = self.evaluate_candidate_models_cv(
			x_train, y_train, folds=folds, parallel=parallel
		)
		if self.task == "classification":
			return cv_report
		return {
			name: {
				"mae_mean": metrics["mean_cv_mae"],
				"mae_std": metrics["std_cv_mae"],
				"rmse_mean": metrics["mean_cv_rmse"],
				"rmse_std": metrics["std_cv_rmse"],
				"mse_mean": metrics["mean_cv_mse"],
				"mse_std": metrics["std_cv_mse"],
				"r2_mean": metrics["mean_cv_r2"],
				"r2_std": metrics["std_cv_r2"],
				"mape_mean": metrics["mean_cv_mape"],
				"mape_std": metrics["std_cv_mape"],
			}
			for name, metrics in cv_report.items()
		}

	def _feature_engineering_report(self, pipeline: Pipeline | None = None) -> dict:
		before = len(self.data.columns) - 1
		if not self.config.enable_feature_engineering:
			return {
				"enabled": False,
				"mode": self.config.feature_engineering_mode,
				"transformations_applied": [],
				"feature_groups": {"numeric": [], "categorical": [], "datetime": [], "interactions": []},
				"feature_count_before_transformation": before,
				"generated_feature_count": 0,
				"feature_count_after_transformation": before,
				"final_transformed_feature_count": before,
				"feature_name_lineage": {column: column for column in self.data.columns if column != self.target},
				"dropped_features": [],
				"skipped_transformations": [],
				"safeguards": {"max_interaction_features": self.config.max_interaction_features, "max_categorical_cardinality": self.config.max_categorical_cardinality},
			}
		transformer = pipeline.named_steps["feature_engineering"] if pipeline and "feature_engineering" in pipeline.named_steps else None
		transformed = transformer.transform(self.data.drop(columns=[self.target])) if transformer else self.data.drop(columns=[self.target])
		plan = transformer.feature_plan_ if transformer else {}
		try:
			final_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
		except (AttributeError, KeyError):
			final_feature_names = transformed.columns.tolist()
		lineage = {}
		for feature_name in final_feature_names:
			clean_name = feature_name.split("__", 1)[-1]
			source = next((column for column in self.data.columns if column != self.target and clean_name.startswith(column)), clean_name)
			lineage[feature_name] = source
		return {
			"enabled": True,
			"mode": self.config.feature_engineering_mode,
			"transformations_applied": [
				*(["log1p"] if plan.get("log_columns") else []),
				*(["datetime_components"] if plan.get("datetime_columns") else []),
				*(["limited_interactions"] if plan.get("interaction_columns") else []),
			],
			"feature_groups": {
				"numeric": plan.get("log_columns", []),
				"categorical": self.data.drop(columns=[self.target]).select_dtypes(exclude="number").columns.tolist(),
				"datetime": plan.get("datetime_columns", []),
				"interactions": plan.get("interaction_columns", []),
			},
			"feature_count_before_transformation": before,
			"generated_feature_count": max(0, len(transformed.columns) - before),
			"feature_count_after_transformation": len(transformed.columns),
			"final_transformed_feature_count": len(final_feature_names),
			"feature_name_lineage": lineage,
			"dropped_features": [],
			"skipped_transformations": plan.get("skipped_transformations", []),
			"safeguards": {
				"max_interaction_features": self.config.max_interaction_features,
				"max_categorical_cardinality": self.config.max_categorical_cardinality,
				"fit_inside_pipeline": True,
			},
		}

	def _classification_report(self) -> dict:
		if self.task != "classification":
			return {}
		counts = self.data[self.target].value_counts()
		imbalance = bool(len(counts) > 1 and counts.max() / counts.min() >= 2)
		return {
			"task_type": self.task_info["task"],
			"number_of_classes": int(counts.size),
			"class_distribution": {str(key): int(value) for key, value in counts.items()},
			"class_labels": [str(value) for value in counts.index.tolist()],
			"imbalance_warnings": ["Class distribution is imbalanced; weighted metrics are reported."] if imbalance else [],
		}

	def fit_best_model(self) -> str:
		"""Fit the CV-selected model on all available rows for inference."""
		if self._selected_model_name is None:
			self.run_experiment()

		best_model_name = self._selected_model_name
		prepared = self.engineer_features(self.data)
		features = prepared.drop(columns=[self.target])
		self._feature_columns = features.columns.tolist()
		self._model_pipeline = self._pipeline(
			features, self._baseline_models()[best_model_name]
		)
		self._model_pipeline.fit(features, prepared[self.target])
		return best_model_name

	def predict(self, new_data: pd.DataFrame) -> pd.Series:
		"""Predict target values for new rows using the fitted model."""
		if self._model_pipeline is None:
			self.fit_best_model()

		input_data = self.engineer_features(new_data.copy())
		missing_columns = [column for column in self._feature_columns if column not in input_data.columns]
		if missing_columns:
			raise ValueError(f"Missing prediction columns: {sorted(missing_columns)}")

		return pd.Series(
			self._model_pipeline.predict(input_data[self._feature_columns]),
			index=input_data.index,
			name=self.target,
		)

	def predict_csv(self, csv_path: str, output_path: str | None = None) -> pd.DataFrame:
		"""Predict a CSV batch and optionally save it with a prediction column."""
		new_data = pd.read_csv(csv_path)
		predictions = self.predict(new_data)
		result = new_data.copy()
		result[self.target] = predictions
		if output_path:
			result.to_csv(output_path, index=False)
		return result

	def run_experiment(self, folds: int | None = None) -> dict:
		"""Run a strict experiment with test isolation and CV-driven model selection."""
		experiment_started = time.perf_counter()
		x_train, x_val, x_test, y_train, y_val, y_test = self.prepare_data_splits()
		x_development = pd.concat([x_train, x_val])
		y_development = pd.concat([y_train, y_val])
		folds = self.config.cv_folds if folds is None else folds
		cv_report = self.evaluate_candidate_models_cv(x_train, y_train, folds=folds)
		if self.config.tune:
			self.tune_models()
		completed_models = {
			name: metrics for name, metrics in cv_report.items()
			if metrics.get("training_status", "completed") == "completed"
		}
		if not completed_models:
			raise RuntimeError("All candidate models failed during cross-validation")
		if self.config.tune and self._tuned_estimators:
			tuned_completed = {
				name: result for name, result in self._tuning_results.items()
				if result.get("status") == "completed"
			}
			if self.task == "classification":
				selected_model_name = max(tuned_completed, key=lambda name: tuned_completed[name]["best_score"])
				selection_metric = "tuned_cv_f1"
			else:
				selected_model_name = max(tuned_completed, key=lambda name: tuned_completed[name]["best_score"])
				selection_metric = "tuned_cv_neg_mean_absolute_error"
		else:
			if self.task == "classification":
				selection_metric = "mean_cv_f1"
				selected_model_name = max(completed_models.items(), key=lambda item: item[1][selection_metric])[0]
			else:
				selection_metric = "mean_cv_mae"
				selected_model_name = min(completed_models.items(), key=lambda item: item[1][selection_metric])[0]
		self._selected_model_name = selected_model_name
		selected_pipeline = self._tuned_estimators.get(selected_model_name)
		if selected_pipeline is None:
			selected_pipeline = self._pipeline(x_train, self._baseline_models()[selected_model_name])
			selected_pipeline.fit(x_train, y_train)
		validation_predictions = selected_pipeline.predict(x_val)
		validation_scores = selected_pipeline.predict_proba(x_val) if self.task == "classification" and hasattr(selected_pipeline, "predict_proba") else None
		validation_metrics = self._score_predictions(y_val, validation_predictions, validation_scores)
		final_pipeline = clone(selected_pipeline)
		final_pipeline.fit(x_development, y_development)
		final_predictions = final_pipeline.predict(x_test)
		final_scores = final_pipeline.predict_proba(x_test) if self.task == "classification" and hasattr(final_pipeline, "predict_proba") else None
		final_metrics = self._score_predictions(y_test, final_predictions, final_scores)
		self._model_pipeline = final_pipeline
		self._feature_columns = x_development.columns.tolist()
		profile = self.profile()
		feature_engineering_plan = self._feature_engineering_report(final_pipeline)
		error_analysis = self._error_analysis(y_test, final_predictions) if self.config.error_analysis else {}
		feature_importance = self._feature_importance(final_pipeline) if self.config.feature_importance else {}
		validation_strategy = profile["validation_strategy"]
		warnings = list(profile["warnings"])
		if self._classification_cv_warning:
			warnings.append(self._classification_cv_warning)
		engineered_columns = [
			column for column in self.engineer_features(self.data).columns
			if column not in self.data.columns
		]
		self._last_experiment = {
			"experiment_id": self.experiment_id,
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"dataset": {
				**profile,
				"target": self.target,
			},
			"split": self._split,
			"cross_validation": {
				"folds": getattr(self, "_effective_cv_folds", folds),
				"primary_metric": "f1" if self.task == "classification" else "mae",
				"models": cv_report,
			},
			"candidate_models": list(self._baseline_models()),
			"model_selection": {
				"selected_model": selected_model_name,
				"selection_metric": selection_metric.removeprefix("mean_"),
				"selection_value": (
					self._tuning_results[selected_model_name]["best_score"]
					if self.config.tune and selected_model_name in self._tuning_results
					else cv_report[selected_model_name][selection_metric]
				),
				"tuned": bool(self.config.tune and selected_model_name in self._tuned_estimators),
			},
			"validation_results": validation_metrics,
			"final_test_results": final_metrics,
			"interpretation": self._interpret_model(final_pipeline),
			"configuration": {
				"random_state": self.random_state,
				"test_size": self.config.test_size,
				"validation_size": self.config.validation_size,
				"cv_folds": folds,
				"tune": self.config.tune,
				"n_iter": self.config.n_iter,
				"tuning_cv": self.config.tuning_cv,
				"tuning_scoring": self.config.tuning_scoring,
				"tuning_random_seed": self.config.tuning_random_seed,
				"error_analysis": self.config.error_analysis,
				"feature_importance": self.config.feature_importance,
				"candidate_models": list(self._baseline_models()),
				"feature_engineering": feature_engineering_plan,
				"preprocessing": {
					"numeric": ["median_imputer", "standard_scaler"],
					"categorical": ["most_frequent_imputer", "one_hot_encoder_drop_first"],
				},
			},
			"parameters": {
				"models": {
					name: model.get_params() if hasattr(model, "get_params") else {}
					for name, model in self._baseline_models().items()
				},
				"registry": self.model_registry.specifications(),
			},
			"limitations": profile["warnings"],
			"task": self.task_info,
			"profiler": profile,
			"validation_strategy": validation_strategy,
			"leakage_analysis": profile["leakage_analysis"],
			"feature_engineering_plan": feature_engineering_plan,
			"model_comparison": cv_report,
			"hyperparameter_optimization": {
				"enabled": self.config.tune,
				"results": {
					name: {key: value for key, value in result.items() if key != "estimator"}
					for name, result in self._tuning_results.items()
				},
			},
			"baseline_vs_tuned": {
				"baseline_model": selected_model_name,
				"baseline_cv": cv_report.get(selected_model_name, {}),
				"tuned": (
					{key: value for key, value in self._tuning_results[selected_model_name].items() if key != "estimator"}
					if self.config.tune and selected_model_name in self._tuning_results
					else None
				),
			},
			"error_analysis": error_analysis,
			"feature_importance": feature_importance,
			"computational_summary": {
				"runtime_seconds": time.perf_counter() - experiment_started,
				"tuning_enabled": self.config.tune,
				"tuning_n_iter": self.config.n_iter if self.config.tune else 0,
			},
			"warnings": warnings,
			"classification": self._classification_report(),
		}
		self._last_experiment["research_summary"] = {
			"dataset": self.source,
			"task": self.task_info["task"],
			"validation_strategy": validation_strategy["strategy"],
			"validation_rationale": validation_strategy["rationale"],
			"leakage_concerns_detected": bool(profile["leakage_audit"]),
			"feature_engineering_applied": feature_engineering_plan["transformations_applied"],
			"models_compared": list(cv_report),
			"selected_model": selected_model_name,
			"cv_metrics": cv_report[selected_model_name],
			"test_metrics": final_metrics,
			"limitations": warnings,
		}
		self._last_experiment["reproducibility"] = {
			"python_version": platform.python_version(),
			"numpy_version": np.__version__,
			"pandas_version": pd.__version__,
			"random_seed": self.random_state,
			"configuration": self._last_experiment["configuration"],
			"dataset_fingerprint": _dataset_fingerprint(self.data) if self.config.dataset_fingerprint else None,
		}
		self._last_experiment["statistics"] = self.statistical_analyzer.summarize_metrics({
			selected_model_name: final_metrics,
		}) if self.config.statistical_analysis else {}
		self._last_experiment["lineage"] = ExperimentLineage(self).build(self._last_experiment)
		self._last_experiment["robustness"] = self._run_robustness(selected_model_name, folds) if self.config.robustness_enabled else {
			"enabled": False,
			"status": "single-run result",
		}
		return self._last_experiment

	def _run_robustness(self, selected_model_name: str, folds: int) -> dict:
		seeds = self.config.robustness_seeds or (self.random_state,)
		seed_results: dict[int, dict] = {}
		for seed in seeds:
			try:
				config = replace(
					self.config,
					random_state=seed,
					models=(selected_model_name,),
					tune=False,
					robustness_enabled=False,
					experiment_tracking=False,
					statistical_analysis=False,
					research_report=False,
				)
				result = MLExperimentAnalyst(self.data, self.target, source=self.source, config=config).run_experiment(folds=folds)
				seed_results[seed] = result["final_test_results"]
			except (ValueError, RuntimeError) as error:
				seed_results[seed] = {"error": str(error)}
		valid_results = {seed: metrics for seed, metrics in seed_results.items() if "error" not in metrics}
		if not valid_results:
			return {"enabled": True, "seeds": list(seed_results), "results": seed_results, "status": "insufficient evidence of stability"}
		return RobustnessAnalyzer(self.statistical_analyzer).analyze(valid_results)

	def _interpret_model(self, pipeline: Pipeline) -> dict:
		"""Return a compact, model-appropriate interpretation summary."""
		preprocessor = pipeline.named_steps["preprocessor"]
		model = pipeline.named_steps["model"]
		feature_names = preprocessor.get_feature_names_out().tolist()
		if hasattr(model, "coef_"):
			values = np.asarray(model.coef_).ravel()
			order = np.argsort(np.abs(values))[::-1][:10]
			return {
				"type": "linear_coefficients",
				"features": [
					{"feature": feature_names[index], "coefficient": float(values[index])}
					for index in order
				],
			}
		if hasattr(model, "feature_importances_"):
			values = np.asarray(model.feature_importances_)
			order = np.argsort(values)[::-1][:10]
			return {
				"type": "feature_importances",
				"features": [
					{"feature": feature_names[index], "importance": float(values[index])}
					for index in order
				],
			}
		return {"type": "unavailable", "features": []}

	def experiment_report(self, folds: int | None = None) -> dict:
		"""Run the complete analysis and return a reproducible experiment report."""
		baseline_results = self.run_baselines()
		experiment = self.run_experiment(folds=folds)
		return {
			"experiment_id": self.experiment_id,
			"timestamp": experiment["timestamp"],
			"dataset": experiment["dataset"],
			"profile": self.profile(),
			"baselines": [result.__dict__ for result in baseline_results],
			"cross_validation": experiment["cross_validation"],
			"candidate_models": experiment["candidate_models"],
			"model_selection": experiment["model_selection"],
			"validation_results": experiment["validation_results"],
			"final_test_results": experiment["final_test_results"],
			"interpretation": experiment["interpretation"],
			"split": experiment["split"],
			"configuration": experiment["configuration"],
			"parameters": experiment["parameters"],
			"limitations": experiment["limitations"],
			"task": experiment["task"],
			"profiler": experiment["profiler"],
			"validation_strategy": experiment["validation_strategy"],
			"leakage_analysis": experiment["leakage_analysis"],
			"feature_engineering_plan": experiment["feature_engineering_plan"],
			"model_comparison": experiment["model_comparison"],
			"hyperparameter_optimization": experiment["hyperparameter_optimization"],
			"baseline_vs_tuned": experiment["baseline_vs_tuned"],
			"error_analysis": experiment["error_analysis"],
			"feature_importance": experiment["feature_importance"],
			"computational_summary": experiment["computational_summary"],
			"warnings": experiment["warnings"],
			"classification": experiment["classification"],
			"research_summary": experiment["research_summary"],
			"robustness": experiment.get("robustness", {}),
			"statistics": experiment.get("statistics", {}),
			"lineage": experiment.get("lineage", {}),
			"reproducibility": experiment.get("reproducibility", {}),
			"research_report": ResearchReportGenerator(self).build(experiment) if self.config.research_report else {},
			"experiment": experiment,
		}

	@staticmethod
	def load_experiment(experiment_directory: str | Path) -> dict:
		"""Reload a JSON experiment without fitting a model or contacting a service."""
		return ExperimentTracker().load(experiment_directory)

	def save_report(self, output_path: str, folds: int | None = None) -> None:
		"""Save the complete experiment report as readable JSON."""
		with open(output_path, "w", encoding="utf-8") as report_file:
			json.dump(self.experiment_report(folds=folds), report_file, indent=2)

	def save_model_artifact(self, output_directory: str) -> tuple[Path, Path]:
		"""Save the selected preprocessing pipeline and its experiment metadata."""
		if self._last_experiment is None or self._model_pipeline is None:
			self.run_experiment()
		artifact_directory = Path(output_directory) / self.experiment_id
		artifact_directory.mkdir(parents=True, exist_ok=True)
		model_path = artifact_directory / "model.joblib"
		metadata_path = artifact_directory / "metadata.json"
		joblib.dump(self._model_pipeline, model_path)
		metadata_path.write_text(
			json.dumps(self._last_experiment, indent=2),
			encoding="utf-8",
		)
		if self.config.tune:
			search_results_path = artifact_directory / "search_results.json"
			search_results_path.write_text(
				json.dumps(self._last_experiment.get("hyperparameter_optimization", {}), indent=2),
				encoding="utf-8",
			)
		if self.config.experiment_tracking:
			ExperimentTracker(output_directory).save(self, self._last_experiment)
		return model_path, metadata_path

	def train_production_model(self) -> tuple[str, Pipeline]:
		"""Train the CV-selected model on all available data for later inference."""
		if self._selected_model_name is None:
			self.run_experiment()
		prepared = self.engineer_features(self.data)
		features = prepared.drop(columns=[self.target])
		selected_model_name = self._selected_model_name
		production_model = self._baseline_models()[selected_model_name]
		production_pipeline = self._pipeline(features, production_model)
		production_pipeline.fit(features, prepared[self.target])
		return selected_model_name, production_pipeline

	def recommend_next_experiment(self) -> str:
		if not self.results:
			raise RuntimeError("Run run_baselines() before requesting a recommendation")

		best = min(self.results, key=lambda result: result.mean_absolute_error)
		missing_columns = self.data.columns[self.data.isna().any()].tolist()
		if missing_columns:
			return (
				f"The best baseline is {best.model_name}. Investigate missing values "
				f"in {missing_columns}, then compare the result with cross-validation."
			)
		return (
			f"The best baseline is {best.model_name}. Tune the main model and "
			"validate the result with a held-out test set."
		)


def build_demo_dataset() -> pd.DataFrame:
	"""Create a deterministic dataset so the prototype runs without external files."""
	return pd.DataFrame({
		"area_sqft": [900, 1100, 1250, 1500, 1800, 2100, 2400, 2700, 3000, 3300],
		"bedrooms": [2, 2, 3, 3, 4, 4, 4, 5, 5, 6],
		"neighborhood": [
			"North", "North", "East", "East", "South",
			"South", "West", "West", "West", "North",
		],
		"sale_price": [180, 220, 260, 310, 390, 440, 510, 570, 640, 700],
	})


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Profile a dataset, run ML baselines, and recommend an experiment."
	)
	parser.add_argument(
		"--data",
		dest="data_path",
		help="Path to a local CSV dataset. If omitted, the built-in demo dataset is used.",
	)
	parser.add_argument(
		"--csv",
		dest="legacy_csv_path",
		help="Backward-compatible alias for --data.",
	)
	parser.add_argument(
		"--target",
		default="sale_price",
		help="Name of the numeric target column (default: sale_price).",
	)
	parser.add_argument(
		"--report",
		help="Optional path where the experiment report JSON will be saved.",
	)
	parser.add_argument(
		"--artifact-dir",
		default="artifacts",
		help="Directory where the selected model artifact is saved.",
	)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--validation-size", type=float, default=0.2)
	parser.add_argument("--cv-folds", type=int, default=5)
	parser.add_argument("--random-seed", type=int, default=42)
	parser.add_argument(
		"--models",
		default=",".join(DEFAULT_MODEL_NAMES),
		help="Comma-separated model names to evaluate.",
	)
	parser.add_argument(
		"--task",
		choices=["auto", "regression", "classification"],
		default="auto",
		help="Task override: auto, regression, or classification.",
	)
	parser.add_argument(
		"--cv-strategy",
		choices=["auto", "kfold", "stratified", "timeseries", "group"],
		default="auto",
		help="Validation strategy for model selection.",
	)
	parser.add_argument("--feature-engineering", dest="feature_engineering", action="store_true")
	parser.add_argument("--no-feature-engineering", dest="feature_engineering", action="store_false")
	parser.add_argument("--tune", action="store_true", help="Enable bounded randomized hyperparameter tuning.")
	parser.add_argument("--n-iter", type=int, default=10, help="Number of tuning candidates per model.")
	parser.add_argument("--tuning-cv", type=int, help="Cross-validation folds for tuning.")
	parser.add_argument("--tuning-scoring", help="Optional scikit-learn scoring name for tuning.")
	parser.add_argument("--tuning-random-seed", type=int, help="Random seed for tuning.")
	parser.add_argument("--no-error-analysis", dest="error_analysis", action="store_false")
	parser.add_argument("--no-feature-importance", dest="feature_importance", action="store_false")
	
	parser.add_argument("--experiment-name", help="Name for the experiment.")
	parser.add_argument("--track-experiment", dest="experiment_tracking", action="store_true", help="Enable tracking.")
	parser.add_argument("--no-track-experiment", dest="experiment_tracking", action="store_false")
	parser.add_argument("--robustness", dest="robustness_enabled", action="store_true", help="Enable robustness.")
	parser.add_argument("--robustness-seeds", help="Comma-separated list of integer seeds for robustness.")
	parser.add_argument("--statistical-analysis", dest="statistical_analysis", action="store_true")
	parser.add_argument("--no-statistical-analysis", dest="statistical_analysis", action="store_false")
	parser.add_argument("--research-report", dest="research_report", action="store_true")
	parser.add_argument("--no-research-report", dest="research_report", action="store_false")

	parser.set_defaults(feature_engineering=False)
	parser.set_defaults(error_analysis=True, feature_importance=True)
	parser.set_defaults(experiment_tracking=True, robustness_enabled=False, statistical_analysis=True, research_report=True)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	data_path = args.data_path or args.legacy_csv_path
	config = ExperimentConfig(
		test_size=args.test_size,
		validation_size=args.validation_size,
		cv_folds=args.cv_folds,
		random_state=args.random_seed,
		models=tuple(name.strip() for name in args.models.split(",") if name.strip()),
		task=args.task,
		cv_strategy=args.cv_strategy,
		enable_feature_engineering=args.feature_engineering,
		tune=args.tune,
		n_iter=args.n_iter,
		tuning_cv=args.tuning_cv,
		tuning_scoring=args.tuning_scoring,
		tuning_random_seed=args.tuning_random_seed,
		error_analysis=args.error_analysis,
		feature_importance=args.feature_importance,
		experiment_name=args.experiment_name,
		experiment_tracking=args.experiment_tracking,
		robustness_enabled=args.robustness_enabled,
		robustness_seeds=tuple(int(s.strip()) for s in args.robustness_seeds.split(",") if s.strip()) if args.robustness_seeds else None,
		statistical_analysis=args.statistical_analysis,
		research_report=args.research_report,
	)
	analyst = (
		MLExperimentAnalyst.from_csv(data_path, target=args.target, config=config)
		if data_path
		else MLExperimentAnalyst(build_demo_dataset(), target=args.target, config=config)
	)
	print("DATA PROFILE")
	print(analyst.profile())
	print("\nBASELINE RESULTS")
	for result in analyst.run_baselines():
		print(
			f"{result.model_name}: "
			f"MAE={result.mean_absolute_error:.2f}, "
			f"R2={result.r2_score:.2f}"
		)
	report = analyst.experiment_report()
	print("\nCROSS-VALIDATION")
	for model_name, metrics in report["cross_validation"]["models"].items():
		print(
			f"{model_name}: "
			f"MAE={metrics['mean_cv_mae']:.2f} +/- {metrics['std_cv_mae']:.2f}, "
			f"RMSE={metrics['mean_cv_rmse']:.2f} +/- {metrics['std_cv_rmse']:.2f}"
		)
	print(f"\nSELECTED MODEL: {report['model_selection']['selected_model']}")
	print("WARNINGS")
	for warning in report["limitations"]:
		print(f"- {warning}")
	print("FINAL TEST METRICS")
	for metric, value in report["final_test_results"].items():
		print(f"{metric.upper()}={value:.4f}")
	print("\nNEXT EXPERIMENT")
	print(analyst.recommend_next_experiment())
	if args.report:
		Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
		print(f"\nREPORT SAVED: {args.report}")
	model_path, metadata_path = analyst.save_model_artifact(args.artifact_dir)
	print(f"MODEL ARTIFACT: {model_path}")
	print(f"MODEL METADATA: {metadata_path}")

