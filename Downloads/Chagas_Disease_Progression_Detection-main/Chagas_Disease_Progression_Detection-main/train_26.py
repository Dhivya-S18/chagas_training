import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "signals_features.csv"
ARTIFACT_DIR = "artifacts_26"

TIME_STEPS = 10
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# 26 FEATURES
# ============================================================

FEATURES_26 = [
    "PR_interval_mean",
    "PR_interval_std",
    "PR_interval_min",
    "PR_interval_max",

    "QRS_duration_mean",
    "QRS_duration_std",
    "QRS_duration_min",
    "QRS_duration_max",

    "QT_interval_mean",
    "QT_interval_std",
    "QT_interval_min",
    "QT_interval_max",

    "ST_segment_mean",
    "ST_segment_std",
    "ST_segment_min",
    "ST_segment_max",

    "ST_slope_mean",
    "ST_slope_std",
    "ST_slope_min",
    "ST_slope_max",

    "HRV_MeanNN",
    "HRV_SDNN",
    "HRV_RMSSD",
    "HRV_pNN50",

    "age",
    "is_male"
]

TARGET = "chagas"


# ============================================================
# CREATE 10-ROW SEQUENCES
# ============================================================

def create_sequences(X, y, time_steps=10):

    X_sequences = []
    y_sequences = []

    for i in range(len(X) - time_steps):

        X_sequences.append(
            X[i:i + time_steps]
        )

        y_sequences.append(
            y[i + time_steps]
        )

    return np.asarray(X_sequences), np.asarray(y_sequences)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

data = pd.read_csv(DATASET_PATH)

print("Original dataset shape:", data.shape)


# ============================================================
# SELECT ONLY 26 FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES_26
    if feature not in data.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )

X = data[FEATURES_26].copy()
y = data[TARGET].copy()


# ============================================================
# CONVERT DATA TYPES
# ============================================================

X["is_male"] = pd.to_numeric(
    X["is_male"],
    errors="coerce"
)

for column in FEATURES_26:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

y = pd.to_numeric(
    y,
    errors="coerce"
)


# ============================================================
# REMOVE ROWS WITH MISSING TARGET
# ============================================================

valid_target = y.notna()

X = X.loc[valid_target].reset_index(drop=True)
y = y.loc[valid_target].reset_index(drop=True)


# ============================================================
# REMOVE DUPLICATE FEATURE/TARGET ROWS
# ============================================================

combined = X.copy()
combined[TARGET] = y.values

before_duplicates = len(combined)

combined = combined.drop_duplicates().reset_index(drop=True)

after_duplicates = len(combined)

print(
    "Rows removed as duplicates:",
    before_duplicates - after_duplicates
)

X = combined[FEATURES_26].copy()
y = combined[TARGET].copy()


# ============================================================
# ORDERED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

n = len(X)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train_raw = X.iloc[:train_end].copy()
X_val_raw = X.iloc[train_end:validation_end].copy()
X_test_raw = X.iloc[validation_end:].copy()

y_train_raw = y.iloc[:train_end].copy()
y_val_raw = y.iloc[train_end:validation_end].copy()
y_test_raw = y.iloc[validation_end:].copy()

print("\nRaw split:")
print("Train:", X_train_raw.shape)
print("Validation:", X_val_raw.shape)
print("Test:", X_test_raw.shape)


# ============================================================
# IMPUTATION
# FIT ONLY ON TRAINING DATA
# ============================================================

imputer = SimpleImputer(strategy="mean")

X_train = imputer.fit_transform(X_train_raw)

X_val = imputer.transform(X_val_raw)

X_test = imputer.transform(X_test_raw)


# ============================================================
# SCALING
# FIT ONLY ON TRAINING DATA
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)


# ============================================================
# CREATE SEQUENCES
# ============================================================

X_train_seq, y_train_seq = create_sequences(
    X_train,
    y_train_raw.to_numpy(),
    TIME_STEPS
)

X_val_seq, y_val_seq = create_sequences(
    X_val,
    y_val_raw.to_numpy(),
    TIME_STEPS
)

X_test_seq, y_test_seq = create_sequences(
    X_test,
    y_test_raw.to_numpy(),
    TIME_STEPS
)


print("\nSequence shapes:")

print(
    "Training:",
    X_train_seq.shape,
    y_train_seq.shape
)

print(
    "Validation:",
    X_val_seq.shape,
    y_val_seq.shape
)

print(
    "Test:",
    X_test_seq.shape,
    y_test_seq.shape
)


# ============================================================
# BUILD NEW 26-FEATURE LSTM
# ============================================================

model = Sequential([

    Input(
        shape=(TIME_STEPS, len(FEATURES_26))
    ),

    LSTM(
        128,
        return_sequences=True
    ),

    BatchNormalization(),

    Dropout(0.2),

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

history = model.fit(
    X_train_seq,
    y_train_seq,
    validation_data=(
        X_val_seq,
        y_val_seq
    ),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\nEvaluating test set...")

test_loss, test_accuracy = model.evaluate(
    X_test_seq,
    y_test_seq,
    verbose=0
)


# ============================================================
# PREDICTIONS
# ============================================================

y_scores = model.predict(
    X_test_seq,
    verbose=0
).ravel()

# 0.5 is used only to convert the model score into
# a binary class for evaluation.
y_pred = (
    y_scores >= 0.5
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test_seq,
    y_pred
)

precision = precision_score(
    y_test_seq,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test_seq,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test_seq,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test_seq,
    y_pred
)


print("\n========================================")
print("26-FEATURE MODEL RESULTS")
print("========================================")

print(
    f"Test loss      : {test_loss:.4f}"
)

print(
    f"Test accuracy  : {accuracy:.4f}"
)

print(
    f"Precision      : {precision:.4f}"
)

print(
    f"Recall         : {recall:.4f}"
)

print(
    f"F1-score       : {f1:.4f}"
)

print("\nConfusion matrix:")
print(cm)


# ============================================================
# SAVE ARTIFACTS
# ============================================================

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)


# Model
model_path = os.path.join(
    ARTIFACT_DIR,
    "model.keras"
)

model.save(model_path)


# Imputer
imputer_path = os.path.join(
    ARTIFACT_DIR,
    "imputer.pkl"
)

joblib.dump(
    imputer,
    imputer_path
)


# Scaler
scaler_path = os.path.join(
    ARTIFACT_DIR,
    "scaler.pkl"
)

joblib.dump(
    scaler,
    scaler_path
)


# Feature schema
schema = {
    "features": FEATURES_26,
    "feature_count": len(FEATURES_26),
    "time_steps": TIME_STEPS,
    "input_shape": [
        TIME_STEPS,
        len(FEATURES_26)
    ],
    "target": TARGET
}

schema_path = os.path.join(
    ARTIFACT_DIR,
    "feature_schema.json"
)

with open(
    schema_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        schema,
        file,
        indent=4
    )


print("\n========================================")
print("ARTIFACTS SAVED")
print("========================================")

print(model_path)
print(imputer_path)
print(scaler_path)
print(schema_path)

print("\nTraining completed successfully.")