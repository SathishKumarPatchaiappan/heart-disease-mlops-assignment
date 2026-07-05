"""FastAPI application serving the packaged heart-disease classifier."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from threading import Lock

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.schemas import PatientFeatures, PredictionResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("heart-disease-api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/heart_disease_pipeline.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
MODEL = None
MODEL_LOCK = Lock()

REQUEST_COUNT = Counter(
    "heart_api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "heart_api_request_duration_seconds", "API request latency", ["endpoint"]
)
PREDICTION_COUNT = Counter(
    "heart_api_predictions_total", "Predictions grouped by class", ["prediction"]
)


def load_model(path: Path | None = None):
    global MODEL
    resolved_path = path or MODEL_PATH
    with MODEL_LOCK:
        if MODEL is None:
            if not resolved_path.exists():
                raise FileNotFoundError(f"Model not found at {resolved_path}")
            MODEL = joblib.load(resolved_path)
            LOGGER.info("Loaded model from %s", resolved_path)
    return MODEL


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    version=MODEL_VERSION,
    description="Cloud-ready inference API for the UCI Heart Disease classifier.",
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.perf_counter() - start)
        REQUEST_COUNT.labels(
            method=request.method, endpoint=request.url.path, status=status
        ).inc()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "heart-disease-prediction-api",
        "version": MODEL_VERSION,
        "documentation": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    try:
        load_model()
        return {"status": "healthy", "model": "loaded", "version": MODEL_VERSION}
    except FileNotFoundError as exc:
        LOGGER.error("Health check failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    frame = pd.DataFrame([features.model_dump()])
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    confidence = probability if prediction == 1 else 1.0 - probability
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    LOGGER.info(
        "prediction=%s disease_probability=%.4f confidence=%.4f",
        prediction,
        probability,
        confidence,
    )
    return PredictionResponse(
        prediction=prediction,
        risk="high" if prediction == 1 else "low",
        confidence=round(confidence, 4),
        model_version=MODEL_VERSION,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
