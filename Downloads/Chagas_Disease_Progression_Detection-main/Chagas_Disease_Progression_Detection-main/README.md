# Chagas Disease Progression Detection

This project trains and serves the existing single LSTM classifier using the
precomputed feature table in `signals_features.csv`. It does not accept raw ECG
samples and does not integrate with ESP32 or a MERN application.

Sequence semantics require confirmation before real-time ECG integration. The
current training code preserves the existing rule: each model input contains
10 adjacent rows in CSV order and is labeled with the `chagas` value from the
row immediately following those 10 rows. The dataset does not establish what
one row or a 10-row sequence represents.

## Install

```powershell
py -m pip install -r requirements.txt
```

Python 3.12 is the verified project runtime.

## Train

```powershell
py chagas.py
```

Training saves `artifacts/model.keras`, `artifacts/scaler.pkl` (the fitted
imputer, scaler, feature order, and preprocessing metadata), and
`artifacts/feature_schema.json`. Imputation and scaling are fitted from the
training segment only. The ordered dataset is separated into contiguous
70%/15%/15% train/validation/test segments before window creation; windowing is
performed separately in each segment to avoid overlap across splits.

## Test Inference

After training, run:

```powershell
py inference.py
py -m unittest discover -s tests -v
```

The example loads the saved artifacts without training, reads one 10-row sample
from the existing feature dataset, runs prediction, and checks invalid sequence
sizes. Inference expects exactly 10 rows and 49 numeric features per row.

## Start the ML API

```powershell
py -m uvicorn app:app --host 0.0.0.0 --port 8000
```

The model and preprocessing artifacts are loaded once when the service starts.
If loading fails or artifacts are missing, `/health` reports `ready: false` and
`/predict` returns HTTP 503.

## API

`GET /health` reports API status, artifact availability, and whether the model
loaded successfully. Before training, a response will be degraded, for example:

```json
{
	"status": "degraded",
	"api_running": true,
	"model_loaded": false,
	"preprocessing_artifacts_available": false,
	"artifacts": {
		"model_file_exists": false,
		"preprocessing_artifact_exists": false,
		"feature_schema_exists": false
	},
	"ready": false
}
```

`POST /predict` accepts a sequence of 10 ordered rows, each containing exactly
49 numeric values. The exact feature order is in
`artifacts/feature_schema.json`; list input follows that order:

```json
{
	"sequence": [
		["49 ordered numeric values"],
		["49 ordered numeric values"],
		["49 ordered numeric values"],
		["49 ordered numeric values"],
		["49 ordered numeric values"],
		["49 ordered numeric values"],
		["49 ordered numeric values"],
		["49 ordered numeric values"],
		["49 ordered numeric values"],
		["49 ordered numeric values"]
	]
}
```

Each placeholder above represents 49 actual JSON numbers; strings are rejected.
Named feature objects are also accepted only when they contain the exact schema
keys in schema order. Missing features, extra features, incorrect dimensions,
non-numeric values, and infinite values are rejected. NaN values can be imputed
using the training-fitted imputer; any non-finite values remaining after
preprocessing are rejected.

The response contains the model's sigmoid score for encoded target
`chagas=1` (CSV `True`). It is not a calibrated probability or medical
interpretation. No class-decision threshold exists in the current project, so
`prediction.class` is `null` until a threshold is justified and configured.

```json
{
	"prediction": {
		"class": null,
		"reason": "No class-decision threshold is configured."
	},
	"probability": 0.000046271466999314725,
	"target": "chagas=1 (CSV True)"
}
```

The score above is the observed output from the local inference test on the
current saved model and dataset sample; retraining or a different sequence can
produce a different score.

Raw ECG is not accepted by this API. Signal processing, feature extraction,
ESP32 integration, and MERN integration are out of scope for this project stage.