"""Production-style FastAPI inference service with Prometheus monitoring."""
import logging
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from .schema import FEATURES

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "model.pkl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-disease-api")

REQUESTS = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)
LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["endpoint"],
)
PREDICTIONS = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["prediction"],
)
MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the inference model is loaded successfully (1=yes, 0=no)",
)

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Expose Prometheus metrics without FastAPI slash redirection."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: float = Field(..., ge=50, le=300)
    chol: float = Field(..., ge=50, le=800)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float = Field(..., ge=50, le=300)
    exang: Literal[0, 1]
    oldpeak: float = Field(..., ge=0, le=10)
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[3, 6, 7]


@lru_cache(maxsize=1)
def get_model():
    """Load the model once per API process."""
    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")

    try:
        with MODEL_PATH.open("rb") as file:
            model = pickle.load(file)
        MODEL_LOADED.set(1)
        logger.info("model_loaded path=%s", MODEL_PATH)
        return model
    except Exception:
        MODEL_LOADED.set(0)
        logger.exception("model_load_failed path=%s", MODEL_PATH)
        raise


@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        endpoint = request.url.path
        REQUESTS.labels(
            endpoint=endpoint,
            method=request.method,
            status=str(status),
        ).inc()
        LATENCY.labels(endpoint=endpoint).observe(elapsed)
        logger.info(
            "request method=%s path=%s status=%s latency_ms=%.2f",
            request.method,
            endpoint,
            status,
            elapsed * 1000,
        )


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok", "model_loaded": True}
    except Exception as exc:
        logger.error("health_check_failed error=%s", exc)
        raise HTTPException(status_code=503, detail="model_not_loaded") from exc


@app.get("/")
def root():
    return {
        "service": "heart-disease-api",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.post("/predict")
def predict(patient: Patient):
    try:
        model = get_model()
        row = pd.DataFrame([patient.model_dump()], columns=FEATURES)
        probability = float(model.predict_proba(row)[0, 1])
        prediction = int(probability >= 0.5)
        PREDICTIONS.labels(prediction=str(prediction)).inc()
        return {
            "prediction": prediction,
            "label": "heart_disease_present" if prediction else "heart_disease_absent",
            "confidence": round(max(probability, 1 - probability), 6),
            "probability_positive": round(probability, 6),
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="model_not_loaded") from exc
