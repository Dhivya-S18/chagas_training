import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "artifacts_26"

MODEL_PATH = ARTIFACT_DIR / "model.keras"
IMPUTER_PATH = ARTIFACT_DIR / "imputer.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"


# ============================================================
# PREDICTION CONFIGURATION
# ============================================================

# Binary classification threshold.
#
# The model output is treated as the score for the positive
# class: chagas = 1.
#
# score >= 0.5 -> class 1 -> Chagas
# score <  0.5 -> class 0 -> No Chagas
#
# This is a model decision threshold, NOT a clinical probability.
CLASSIFICATION_THRESHOLD = 0.5


# ============================================================
# LOAD FEATURE SCHEMA
# ============================================================

if not SCHEMA_PATH.exists():
    raise FileNotFoundError(
        f"Feature schema not found: {SCHEMA_PATH}"
    )

with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
    schema = json.load(file)


FEATURES = schema["features"]
FEATURE_COUNT = schema["feature_count"]
TIME_STEPS = schema["time_steps"]
INPUT_SHAPE = tuple(schema["input_shape"])
TARGET = schema["target"]


# ============================================================
# VALIDATE SCHEMA
# ============================================================

if FEATURE_COUNT != 26:
    raise ValueError(
        f"Expected 26 features, but schema contains "
        f"{FEATURE_COUNT}."
    )


if TIME_STEPS != 10:
    raise ValueError(
        f"Expected 10 time steps, but schema contains "
        f"{TIME_STEPS}."
    )


if INPUT_SHAPE != (10, 26):
    raise ValueError(
        f"Expected model input shape (10, 26), "
        f"but schema contains {INPUT_SHAPE}."
    )


if len(FEATURES) != 26:
    raise ValueError(
        f"Expected exactly 26 feature names, "
        f"but found {len(FEATURES)}."
    )


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# LOAD PREPROCESSING ARTIFACTS
# ============================================================

if not IMPUTER_PATH.exists():
    raise FileNotFoundError(
        f"Imputer not found: {IMPUTER_PATH}"
    )

if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )


imputer = joblib.load(
    IMPUTER_PATH
)

scaler = joblib.load(
    SCALER_PATH
)


# ============================================================
# VERIFY MODEL INPUT SHAPE
# ============================================================

model_input_shape = tuple(
    model.input_shape[1:]
)

if model_input_shape != (10, 26):
    raise ValueError(
        f"Expected model input shape (10, 26), "
        f"but loaded model expects {model_input_shape}."
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(features):
    """
    Run inference using the trained 26-feature LSTM model.

    Expected input:
        features = 10 x 26 numeric values

    Returns:
        {
            "score": float,
            "class": 0 or 1,
            "prediction": "No Chagas" or "Chagas"
        }

    Class mapping:
        0 -> No Chagas
        1 -> Chagas

    The score is the raw model output and should NOT be
    interpreted as a calibrated clinical probability.
    """

    # --------------------------------------------------------
    # Convert input to NumPy array
    # --------------------------------------------------------

    try:
        X = np.asarray(
            features,
            dtype=np.float32
        )

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Input features must contain only numeric values."
        ) from exc


    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if X.shape != (TIME_STEPS, FEATURE_COUNT):
        raise ValueError(
            f"Expected input shape "
            f"({TIME_STEPS}, {FEATURE_COUNT}), "
            f"but received {X.shape}."
        )


    # --------------------------------------------------------
    # Validate finite values
    # --------------------------------------------------------

    if not np.isfinite(X).all():
        raise ValueError(
            "Input features contain NaN or infinite values."
        )


    # --------------------------------------------------------
    # Flatten sequence for preprocessing
    #
    # Imputer and scaler were trained on individual
    # 26-feature rows.
    # --------------------------------------------------------

    X_2d = X.reshape(
        -1,
        FEATURE_COUNT
    )


    # --------------------------------------------------------
    # Apply the SAME preprocessing used during training
    # --------------------------------------------------------

    X_2d = imputer.transform(
        X_2d
    )

    X_2d = scaler.transform(
        X_2d
    )


    # --------------------------------------------------------
    # Validate preprocessing result
    # --------------------------------------------------------

    if not np.isfinite(X_2d).all():
        raise ValueError(
            "Preprocessing produced NaN or infinite values."
        )


    # --------------------------------------------------------
    # Restore LSTM sequence shape
    # --------------------------------------------------------

    X_processed = X_2d.reshape(
        1,
        TIME_STEPS,
        FEATURE_COUNT
    ).astype(np.float32)


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    raw_prediction = model.predict(
        X_processed,
        verbose=0
    )


    # --------------------------------------------------------
    # Validate model output
    # --------------------------------------------------------

    if (
        not isinstance(raw_prediction, np.ndarray)
        or raw_prediction.size == 0
    ):
        raise ValueError(
            "Model returned an empty prediction."
        )


    score = float(
        raw_prediction.reshape(-1)[0]
    )


    if not np.isfinite(score):
        raise ValueError(
            "Model returned a non-finite score."
        )


    # --------------------------------------------------------
    # Convert model score to binary class
    # --------------------------------------------------------

    predicted_class = (
        1
        if score >= CLASSIFICATION_THRESHOLD
        else 0
    )


    # --------------------------------------------------------
    # Convert class to readable prediction
    # --------------------------------------------------------

    prediction_label = (
        "Chagas"
        if predicted_class == 1
        else "No Chagas"
    )


    # --------------------------------------------------------
    # Return prediction
    # --------------------------------------------------------

    return {
        "score": score,
        "class": predicted_class,
        "prediction": prediction_label
    }