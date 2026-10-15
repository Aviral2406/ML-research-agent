"""FastAPI serving layer for the ML Experiment Analyst."""

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aitraining import MLExperimentAnalyst, build_demo_dataset

DEFAULT_TARGET = "sale_price"


class PredictionRequest(BaseModel):
    features: dict[str, Any]


class BatchPredictionRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1)


def load_default_analyst() -> MLExperimentAnalyst:
    """Load a custom CSV when configured, otherwise fall back to the demo dataset."""
    dataset_path = os.getenv("MODEL_DATASET_PATH", "").strip()
    target = os.getenv("MODEL_TARGET", DEFAULT_TARGET).strip() or DEFAULT_TARGET

    if dataset_path:
        path = Path(dataset_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"MODEL_DATASET_PATH points to a missing file: {path}"
            )
        return MLExperimentAnalyst.from_csv(str(path), target=target)

    return MLExperimentAnalyst(build_demo_dataset(), target=target)


def create_app(analyst: MLExperimentAnalyst | None = None) -> FastAPI:
    """Create the API and fit the default model once for fast requests."""
    active_analyst = analyst or load_default_analyst()
    active_analyst.fit_best_model()

    def json_value(value: Any) -> Any:
        return value.item() if hasattr(value, "item") else value
    app = FastAPI(
        title="ML Experiment Analyst API",
        version="1.0.0",
        description="Profile datasets and generate ML predictions.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/profile")
    def profile() -> dict:
        return active_analyst.profile()

    @app.post("/predict")
    def predict(request: PredictionRequest) -> dict[str, Any]:
        try:
            values = active_analyst.predict(pd.DataFrame([request.features]))
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "target": active_analyst.target,
            "prediction": json_value(values.iloc[0]),
        }

    @app.post("/predict-batch")
    def predict_batch(request: BatchPredictionRequest) -> dict[str, Any]:
        try:
            rows = pd.DataFrame(request.rows)
            values = active_analyst.predict(rows)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "target": active_analyst.target,
            "predictions": [json_value(value) for value in values],
        }

    return app


app = create_app()