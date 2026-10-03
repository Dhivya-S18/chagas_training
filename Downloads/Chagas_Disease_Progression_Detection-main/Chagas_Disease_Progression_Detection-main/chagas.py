from pathlib import Path

import joblib

from preprocessing import load_dataset, prepare_training_data, save_feature_schema
from stacking_model import train_lstm_model

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "signals_features.csv"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.keras"
PREPROCESSING_PATH = ARTIFACT_DIR / "scaler.pkl"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"


def main():
    X, y = load_dataset(DATASET_PATH)
    splits, preprocessing_artifacts = prepare_training_data(X, y)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = train_lstm_model(splits)
    model.save(MODEL_PATH)
    joblib.dump(preprocessing_artifacts, PREPROCESSING_PATH)
    save_feature_schema(X.columns.tolist(), SCHEMA_PATH)

    print("Saved model:", MODEL_PATH)
    print("Saved preprocessing artifacts:", PREPROCESSING_PATH)
    print("Saved feature schema:", SCHEMA_PATH)

if __name__ == "__main__":
    main()