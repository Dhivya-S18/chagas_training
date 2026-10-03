import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# 26 FEATURES SELECTED FOR THE NEW MODEL
# ============================================================

SELECTED_FEATURES = [
    # ECG morphology / intervals - 20
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

    # HRV - 4
    "HRV_MeanNN",
    "HRV_SDNN",
    "HRV_RMSSD",
    "HRV_pNN50",

    # Demographic - 2
    "age",
    "is_male",
]


TARGET_COLUMN = "chagas"
ID_COLUMN = "exam_id"


# ============================================================
# LOAD AND PREPROCESS
# ============================================================

def load_and_preprocess(file_path):
    """
    Load the Chagas dataset and return only the selected
    26 input features and the target column.

    Output:
        X -> DataFrame with 26 features
        y -> target Series
    """

    data = pd.read_csv(file_path)

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = SELECTED_FEATURES + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Select only the 26 features
    # --------------------------------------------------------

    X = data[SELECTED_FEATURES].copy()

    y = data[TARGET_COLUMN].copy()

    # --------------------------------------------------------
    # Convert boolean values to integers
    # --------------------------------------------------------

    if X["is_male"].dtype == bool:
        X["is_male"] = X["is_male"].astype(int)

    # --------------------------------------------------------
    # Convert feature columns to numeric
    # --------------------------------------------------------

    for column in SELECTED_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Convert target to numeric
    # --------------------------------------------------------

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove rows where target is missing
    # --------------------------------------------------------

    valid_target = y.notna()

    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)

    # --------------------------------------------------------
    # Impute missing feature values
    # --------------------------------------------------------

    imputer = SimpleImputer(strategy="mean")

    X = imputer.fit_transform(X)

    # --------------------------------------------------------
    # Remove duplicate feature/target rows
    # --------------------------------------------------------

    processed = pd.DataFrame(
        X,
        columns=SELECTED_FEATURES
    )

    processed[TARGET_COLUMN] = y.values

    processed = processed.drop_duplicates()

    # --------------------------------------------------------
    # Separate X and y again
    # --------------------------------------------------------

    X = processed[SELECTED_FEATURES].reset_index(drop=True)

    y = processed[TARGET_COLUMN].reset_index(drop=True)

    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X = pd.DataFrame(
        X,
        columns=SELECTED_FEATURES
    )

    return X, y



# import json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler

# TARGET_COLUMN = "chagas"
# ID_COLUMN = "exam_id"
# SEQUENCE_LENGTH = 10
# EXPECTED_FEATURE_COUNT = 49
# TRAIN_FRACTION = 0.70
# VALIDATION_FRACTION = 0.15


# def load_dataset(file_path):
#     data = pd.read_csv(file_path)

#     if ID_COLUMN in data.columns:
#         data = data.drop(columns=[ID_COLUMN])
#     if TARGET_COLUMN not in data.columns:
#         raise ValueError(f"Dataset must contain the target column '{TARGET_COLUMN}'.")

#     if "is_male" in data.columns:
#         data["is_male"] = data["is_male"].astype(int)
#     data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

#     X = data.drop(columns=[TARGET_COLUMN])
#     y = data[TARGET_COLUMN]
#     if len(X.columns) != EXPECTED_FEATURE_COUNT:
#         raise ValueError(
#             f"Expected {EXPECTED_FEATURE_COUNT} features, found {len(X.columns)}."
#         )
#     return X, y


# def create_sequences(X, y, time_steps=SEQUENCE_LENGTH):
#     X = np.asarray(X, dtype=np.float32)
#     y = np.asarray(y, dtype=np.int64)
#     if len(X) <= time_steps:
#         raise ValueError(
#             f"At least {time_steps + 1} rows are required to create labeled sequences."
#         )

#     sequences = np.asarray(
#         [X[index:index + time_steps] for index in range(len(X) - time_steps)],
#         dtype=np.float32,
#     )
#     labels = y[time_steps:]
#     return sequences, labels


# def prepare_training_data(X, y, sequence_length=SEQUENCE_LENGTH):
#     if list(X.columns) == []:
#         raise ValueError("At least one feature column is required.")
#     if len(X.columns) != EXPECTED_FEATURE_COUNT:
#         raise ValueError(
#             f"Expected {EXPECTED_FEATURE_COUNT} features, found {len(X.columns)}."
#         )
#     if len(X) != len(y):
#         raise ValueError("Feature and target row counts must match.")

#     row_count = len(X)
#     train_end = int(row_count * TRAIN_FRACTION)
#     validation_end = int(row_count * (TRAIN_FRACTION + VALIDATION_FRACTION))
#     if train_end <= sequence_length or validation_end - train_end <= sequence_length:
#         raise ValueError("Dataset is too small for the configured sequence splits.")

#     imputer = SimpleImputer(strategy="mean")
#     imputer.fit(X.iloc[:train_end])
#     imputed = pd.DataFrame(
#         imputer.transform(X), columns=list(X.columns), index=X.index
#     )

#     clean = imputed.copy()
#     clean[TARGET_COLUMN] = np.asarray(y, dtype=np.int64)
#     clean["_source_position"] = np.arange(row_count)
#     clean = clean.drop_duplicates(subset=list(X.columns) + [TARGET_COLUMN])
#     positions = clean.pop("_source_position").to_numpy()
#     labels = clean.pop(TARGET_COLUMN).to_numpy(dtype=np.int64)
#     values = clean.to_numpy(dtype=np.float64)

#     train_mask = positions < train_end
#     validation_mask = (positions >= train_end) & (positions < validation_end)
#     test_mask = positions >= validation_end
#     if not train_mask.any() or not validation_mask.any() or not test_mask.any():
#         raise ValueError("Duplicate removal left an empty data split.")

#     scaler = StandardScaler()
#     scaler.fit(values[train_mask])
#     scaled = scaler.transform(values).astype(np.float32)

#     train_X, train_y = create_sequences(
#         scaled[train_mask], labels[train_mask], sequence_length
#     )
#     validation_X, validation_y = create_sequences(
#         scaled[validation_mask], labels[validation_mask], sequence_length
#     )
#     test_X, test_y = create_sequences(
#         scaled[test_mask], labels[test_mask], sequence_length
#     )

#     artifacts = {
#         "imputer": imputer,
#         "scaler": scaler,
#         "feature_names": list(X.columns),
#         "sequence_length": sequence_length,
#         "target_column": TARGET_COLUMN,
#         "split_strategy": "ordered contiguous 70%/15%/15% segments before windowing",
#     }
#     splits = {
#         "train": (train_X, train_y),
#         "validation": (validation_X, validation_y),
#         "test": (test_X, test_y),
#     }
#     return splits, artifacts


# def save_feature_schema(feature_names, schema_path):
#     if len(feature_names) != EXPECTED_FEATURE_COUNT:
#         raise ValueError(
#             f"Expected {EXPECTED_FEATURE_COUNT} features, found {len(feature_names)}."
#         )

#     schema = {
#         "feature_names": list(feature_names),
#         "feature_count": EXPECTED_FEATURE_COUNT,
#         "sequence_length": SEQUENCE_LENGTH,
#         "expected_input_shape": [SEQUENCE_LENGTH, EXPECTED_FEATURE_COUNT],
#         "target": {
#             "name": TARGET_COLUMN,
#             "description": "Binary CSV label encoded as 0/1 (False/True); sequence semantics require confirmation before real-time ECG integration.",
#         },
#         "output": {
#             "shape_per_sample": [1],
#             "activation": "sigmoid",
#             "description": "Model score for encoded target chagas=1; no class-decision threshold is configured.",
#         },
#     }
#     schema_path = Path(schema_path)
#     schema_path.parent.mkdir(parents=True, exist_ok=True)
#     schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
#     return schema