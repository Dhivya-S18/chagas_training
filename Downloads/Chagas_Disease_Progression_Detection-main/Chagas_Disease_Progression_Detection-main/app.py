from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import inference


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts_26"

MODEL_PATH = ARTIFACT_DIR / "model.keras"
IMPUTER_PATH = ARTIFACT_DIR / "imputer.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"


app = FastAPI(
    title="Chagas ML Inference API",
)


class PredictRequest(BaseModel):
    sequence: Any


@app.get("/health")
def health():
    model_exists = MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0
    imputer_exists = IMPUTER_PATH.exists() and IMPUTER_PATH.stat().st_size > 0
    scaler_exists = SCALER_PATH.exists() and SCALER_PATH.stat().st_size > 0
    schema_exists = SCHEMA_PATH.exists() and SCHEMA_PATH.stat().st_size > 0

    preprocessing_available = (
        imputer_exists
        and scaler_exists
        and schema_exists
    )

    model_loaded = model_exists

    ready = model_loaded and preprocessing_available

    return {
        "status": "ok" if ready else "degraded",
        "api_running": True,
        "model_loaded": model_loaded,
        "preprocessing_artifacts_available": preprocessing_available,
        "feature_count": 26,
        "sequence_length": 10,
        "artifact_directory": str(ARTIFACT_DIR),
        "ready": ready,
    }


@app.post("/predict")
def predict_endpoint(body: PredictRequest):
    try:
        return inference.predict(body.sequence)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc