# ML Research Agent

## 1. Project title
ML Research Agent

## 2. Project overview
This project is an automated, dataset-agnostic machine learning experiment and research platform. It allows users to quickly profile datasets, run validation strategies, compare baseline models, and generate comprehensive research reports with statistical and robustness guarantees. It features a robust Python CLI and a lightweight FastAPI serving layer.

## 3. Problem statement
Machine learning practitioners often spend significant time writing boilerplate code for data profiling, preprocessing, cross-validation, and model comparison. The ML Research Agent eliminates this overhead by providing a unified pipeline that automatically handles these tasks safely (preventing data leakage), generates insightful diagnostic reports, and exposes the winning models via a standard API, accelerating the transition from raw data to a deployable model.

## 4. Key capabilities
- Automated dataset profiling and target variable inference
- Automated preprocessing (imputation, scaling, one-hot encoding)
- Cross-validation with strict data leakage prevention
- Support for multiple ML models natively 
- Automated hyperparameter tuning and model selection
- Feature engineering and importance scoring
- Experiment tracking, lineage, and metadata capturing
- Statistical analysis of validation scores
- Robustness/stability analysis across different random seeds
- Automated research report generation
- FastAPI integration for easy model deployment

## 5. Architecture overview
The project is built around two primary components:
1. `aitraining.py`: The core ML orchestrator (`MLExperimentAnalyst`) that manages data ingestion, profiling, training, validation, experiment tracking, and artifact generation.
2. `api.py`: A FastAPI application that dynamically loads the best saved model artifact and exposes `/profile`, `/predict`, and `/predict-batch` endpoints.

## 6. ML workflow
1. **Ingestion & Profiling**: Load a CSV, infer schema, check for missing/invalid values, and warn on small datasets.
2. **Task Detection**: Automatically infer if the problem is regression, binary classification, or multiclass classification.
3. **Validation Strategy**: Select K-Fold, Stratified K-Fold, or Time-Series splitting based on the data.
4. **Training**: Fit preprocessing pipelines and baseline models (e.g., Linear/Logistic Regression, Random Forest, Gradient Boosting).
5. **Selection & Tuning**: Select the best model based on validation metrics, and optionally run randomized hyperparameter tuning.
6. **Analysis & Tracking**: Perform error analysis, calculate feature importance, test for robustness across random seeds, and compute statistical intervals.
7. **Reporting**: Generate a final research report, save model artifacts, and persist all experiment lineage.

## 7. Supported tasks
- **Regression**: Supported for continuous numerical targets (e.g., `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`).
- **Binary classification**: Supported for targets with exactly two classes (e.g., `LogisticRegression`, `RandomForestClassifier`).
- **Multiclass classification**: Supported for targets with more than two discrete classes.

## 8. Feature engineering
An optional feature engineering mode generates sensible interaction terms and polynomial features for numerical columns while controlling for high-cardinality categorical features. This is executed inside the pipeline to prevent data leakage.

## 9. Validation and leakage protection
Strict data leakage prevention is enforced by nesting all preprocessing steps (imputation, scaling, encoding) inside scikit-learn `Pipeline` objects. These pipelines are fitted exclusively on the training folds during cross-validation, ensuring the validation and test sets remain completely unseen.

## 10. Hyperparameter optimization
The system supports bounded, randomized hyperparameter tuning (`--tune`) using `RandomizedSearchCV`. It automatically defines sensible search spaces for supported models to find optimal configurations within a computational budget (`--n-iter`).

## 11. Error analysis
Generates detailed error analysis comparing training vs. validation metrics to diagnose overfitting or underfitting. It flags high-variance estimators and provides actionable recommendations.

## 12. Feature importance
For models that support it (e.g., Random Forests, Gradient Boosting), the pipeline extracts and normalizes feature importance scores, allowing researchers to understand which variables drive the model's predictions.

## 13. Experiment tracking
The `ExperimentTracker` module automatically records experiment metadata, dataset fingerprints, hyperparameter configurations, and metrics, serializing them into JSON artifacts alongside the saved `.joblib` model.

## 14. Experiment comparison
The `ExperimentComparator` module enables programmatic comparison between two saved experiment directories, automatically identifying the better-performing experiment across all tracked metrics while preserving metric directions (e.g., lower MAE is better, higher R2 is better).

## 15. Robustness analysis
The `RobustnessAnalyzer` tests the stability of the selected model by evaluating it across multiple random seeds (accessible via `--robustness` and `--robustness-seeds`). It reports the variance in performance to ensure the model isn't overly sensitive to the data split.

## 16. Statistical analysis
The `StatisticalAnalyzer` computes descriptive statistics (mean, standard deviation) and 95% confidence intervals for cross-validation metrics, providing statistical guarantees on the expected performance.

## 17. Research report generation
The `ResearchReportGenerator` synthesizes the entire experiment—from data profiling to final test metrics—into a human-readable, evidence-based JSON report. It clearly states limitations, highlights warnings, and justifies model selection.

## 18. Reproducibility
Every experiment records a reproducibility payload containing the Python version, NumPy version, a dataset fingerprint, the exact feature set, and the random seeds used, ensuring experiments can be perfectly replicated.

## 19. Project structure
- `aitraining.py`: Core ML pipeline, CLI entrypoint, and experiment classes.
- `api.py`: FastAPI server for querying the trained model.
- `test_*.py`: Comprehensive test suite covering all phases of development.
- `requirements.txt`: Python dependencies.
- `artifacts/`: Directory where models and experiment reports are saved.

## 20. Installation
Ensure you have Python 3.13+ installed.
```bash
pip install -r requirements.txt
```

## 21. Basic usage
To run a basic experiment using the built-in demo dataset:
```bash
python aitraining.py --experiment-name "DemoRun"
```

## 22. CLI examples
Run an experiment with a custom dataset, enable hyperparameter tuning, and save the report:
```bash
python aitraining.py --data my_data.csv --target my_target --tune --n-iter 20 --report custom_report.json
```

Run an experiment with tracking, statistical analysis, and robustness testing enabled:
```bash
python aitraining.py --data my_data.csv --target my_target --track-experiment --statistical-analysis --robustness --robustness-seeds "42,100,999"
```

## 23. API usage
Start the FastAPI server:
```bash
uvicorn api:app --reload
```
Check the API health:
```bash
curl http://127.0.0.1:8000/health
```
Make a prediction:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"features\": {\"area_sqft\": 2100, \"bedrooms\": 4, \"neighborhood\": \"North\"}}"
```

## 24. Testing
The project uses `pytest` for all unit and integration testing.
To run the full test suite (77 tests):
```bash
pytest
```

## 25. Example workflow
1. **Dataset**: Provide `income.csv` containing features (Age, Education, etc.) and target (`Salary`).
2. **Profiling**: The agent detects missing values and data types.
3. **Task Detection**: Identifies the target is continuous -> Regression.
4. **Validation**: Chooses 5-Fold Cross-Validation.
5. **Feature Engineering**: `--feature-engineering` flag generates interaction terms.
6. **Baseline Models**: Evaluates Linear Regression, Ridge, Random Forest, etc.
7. **Hyperparameter Optimization**: `--tune` optimizes the best baseline model.
8. **Error Analysis**: Compares train/val R2 to detect overfitting.
9. **Feature Importance**: Identifies `Education` as the most predictive feature.
10. **Experiment Tracking**: Saves `model.joblib` and `metadata.json` to `/artifacts`.
11. **Robustness/Statistics**: `--robustness` runs the model across 3 seeds; stats confirm a tight confidence interval.
12. **Research Report**: Final `report.json` is generated with evidence-based conclusions.

## 26. Limitations
- Small datasets (e.g., < 50 rows) will trigger variance warnings, and complex models may fail or overfit.
- Feature engineering is currently limited to polynomial and interaction terms for numeric features.
- Extremely high-cardinality categorical variables are warned against but not automatically aggressively pruned.

## 27. Future work
- Support for distributed training on massive datasets.
- Integration with external tracking tools (e.g., MLflow, Weights & Biases).
- Advanced NLP and image feature extraction native capabilities.
