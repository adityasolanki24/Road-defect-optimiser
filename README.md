# Smart Road Maintenance Optimiser

Dashboard for council road operations teams to prioritise defects, tune ranking weights, and generate repair routes for field crews.

## Features

- Interactive Leaflet map with dark tiles and street-following route geometry (OSRM).
- Severity-coded markers: red (high), yellow (medium), green (low).
- Marker clustering when defect volume is high.
- Defect modal with severity, priority score, repair time, report time, and ML forecast.
- Route list with priority percentages, crew time, distance, and spend.
- Slider-controlled weights for severity, traffic, location, and risk.
- **ML defect classifier** — upload a road image to detect type and damage severity, then register on the map.
- **ML severity forecaster** — 7-day and 14-day deterioration predictions with trend and days-to-critical.
- **Evaluation pipeline** — confusion matrix, scatter plots, and `metrics.json` on the held-out test set.

The map starts empty. Defects are added via image upload or the API — there is no simulated seed data.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/defects` | List all defects |
| `POST` | `/defects` | Add a defect |
| `POST` | `/compute-route` | Compute route with configurable weights |
| `GET` | `/optimised-route` | Return route for current weights |
| `GET` | `/metrics` | Dashboard metrics and resource status |
| `GET` | `/weights` | Active ranking weights |
| `GET` | `/ml/status` | ML model load status |
| `POST` | `/ml/detect` | Classify a road image and optionally register a defect |
| `GET` | `/ml/forecast/{defect_id}` | Severity forecast for one defect |
| `GET` | `/ml/forecasts` | Severity forecasts for all defects |

## Folder Structure

```text
RoadDefectOptimiser/
|-- backend/
|   |-- controllers/
|   |-- models/
|   |-- routes/
|   |-- services/
|   |-- ml/
|   |   |-- artifacts/              # trained .joblib models
|   |   |-- data/
|   |   |   |-- matrices/           # NumPy feature matrices + split sample JSON
|   |   |   |-- splits/             # train / val / test indices
|   |   |   `-- rdd2022/            # downloaded RDD2022 images + XML (gitignored)
|   |   |-- datasets/rdd2022.py
|   |   |-- defect_classifier/
|   |   |-- severity_forecaster/
|   |   |-- evaluation/
|   |   |   |-- reports/            # confusion matrix, plots, metrics.json
|   |   |   |-- splits.py
|   |   |   |-- plots.py
|   |   |   `-- run.py
|   |   |-- build_matrices.py
|   |   |-- train_all.py
|   |   `-- generate_samples.py
|   |-- main.py
|   `-- requirements.txt
|-- frontend/
|   `-- src/
`-- README.md
```

## Setup

### Backend

```bash
cd RoadDefectOptimiser
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt
cd backend
python -m ml.train_all    # first time: download RDD2022, train models, generate reports
python main.py
```

Use the virtual environment — system Python will not have ML dependencies (`joblib`, `lightgbm`, etc.).

The API runs at `http://localhost:8001`.

If port 8001 is in use:

```bash
..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8002
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard runs at `http://localhost:5173`.

```bash
VITE_API_BASE_URL=http://localhost:8001 npm run dev
```

---

## ML pipeline

Models train on **[RDD2022](https://github.com/sekilab/RoadDamageDetector)** (real road damage images with Pascal VOC XML annotations). No synthetic training data.

### Commands

```bash
cd backend
python -m ml.build_matrices                  # download RDD2022 + build matrices + splits
python -m ml.build_matrices --countries Czech  # Czech only (~245 MB)
python -m ml.train_all --skip-download        # train + evaluate from existing matrices
python -m ml.train_all --rebuild-matrices     # full refresh
python -m ml.evaluation.run                   # reports only (models must exist)
python -m ml.generate_samples                 # export hold-out test crops for upload demo
```

### RDD2022 label mapping

| RDD2022 code | Defect type |
|--------------|-------------|
| D00, D10 | Longitudinal crack |
| D20 | Surface delamination |
| D40 | Pothole |

### Pipeline steps

1. **Download & parse** — country zip → images + XML bounding boxes.
2. **Crop** — each annotation becomes a padded defect patch.
3. **Feature matrices** — saved to `backend/ml/data/matrices/`:
   - `classifier_X.npy` — image features (1745 × 9226)
   - `classifier_y.npy` — defect type labels
   - `severity_y_current.npy` — ground-truth severity from bbox area
   - `severity_X.npy` — forecaster input features (1745 × 11)
   - `severity_y7.npy` / `severity_y14.npy` — forecast targets
4. **Splits** — stratified **70% train / 15% validation / 15% test** saved to `backend/ml/data/splits/`.
5. **Train** — models fit on the train split; validation split used for early feedback during training.
6. **Evaluate** — all reports computed on the **test split only** (data the models never saw during training).

### Evaluation outputs

Saved to `backend/ml/evaluation/reports/` after `python -m ml.train_all`:

| File | Description |
|------|-------------|
| `confusion_matrix_classifier.png` | Test-set confusion matrix |
| `confusion_matrix_classifier.csv` | Confusion matrix values |
| `class_distribution.png` | Sample counts per class in train / val / test |
| `severity_regressor_scatter.png` | Actual vs predicted severity (test) |
| `severity_forecast_7d_scatter.png` | 7-day forecast scatter (test) |
| `severity_forecast_14d_scatter.png` | 14-day forecast scatter (test) |
| `metrics.json` | All numeric metrics (see below) |

---

## How metrics are computed

### 1. Ground-truth severity (from RDD2022 bboxes)

Each annotated defect has a bounding box. Severity is derived from how large the damage is relative to the full image:

```text
area_ratio = bbox_area / image_area
aspect_factor = min(aspect, 1/aspect)
severity = clip(0.25 + sqrt(area_ratio) × 0.55 + aspect_factor × 0.12,  0.08, 0.98)
```

This produces a value in `[0, 1]` used as the label for the severity regressor and as `current_severity` in the forecaster feature vector.

### 2. Image features (classifier & severity regressor)

Each cropped patch is resized to 48×48 RGB and converted to a **9226-dimensional** vector:

- Flattened RGB pixels (6912 values)
- Flattened grayscale (2304 values)
- Per-channel mean and std (6 values)
- Edge density (horizontal + vertical gradients), gray std, contrast (4 values)

### 3. Forecaster features (`severity_X`, 11 columns)

| Feature | Source |
|---------|--------|
| `current_severity` | Bbox formula above |
| `traffic_density` | Country proxy from RDD2022 source country |
| `location_importance` | Country proxy |
| `risk_factor` | `0.65 × severity + 0.35 × traffic`, clipped |
| `age_days` | `(1 − severity) × 28` (proxy for time since detection) |
| `severity × traffic`, `severity × risk`, `traffic × risk` | Interaction terms |
| `type_*` (3 columns) | One-hot defect type |

### 4. Forecast targets (7-day and 14-day)

RDD2022 has no time-series labels. Targets are derived from **cross-sectional severity statistics** per defect type:

```text
growth_7d  = max(0.02, (P75_severity − P50_severity) × 0.85)
growth_14d = max(0.04, (P90_severity − P50_severity) × 0.90)

target_7d  = clip(current_severity + growth_7d[type],  0, 1)
target_14d = clip(current_severity + growth_14d[type], 0, 1)
```

The forecaster learns to map the 11 input features to these targets.

### 5. Models

| Model | Algorithm | Trained on | Predicts |
|-------|-----------|------------|----------|
| Defect classifier | HistGradientBoosting + StandardScaler | Train split | Defect type |
| Severity regressor | LightGBM | Train split | Damage severity `[0, 1]` |
| Severity forecaster (7d) | LightGBM | Train split | 7-day severity |
| Severity forecaster (14d) | LightGBM | Train split | 14-day severity |

### 6. ML evaluation metrics (`metrics.json`)

Computed on the **held-out test split** (and separately on validation for comparison).

**Classifier (classification)**

| Metric | Formula / meaning |
|--------|-------------------|
| **Accuracy** | Fraction of test crops with correct predicted type |
| **Macro F1** | Unweighted mean of per-class F1 scores |
| **Weighted F1** | Support-weighted mean of per-class F1 |
| **Precision** (per class) | `TP / (TP + FP)` — of predicted class X, how many were correct |
| **Recall** (per class) | `TP / (TP + FN)` — of actual class X, how many were found |
| **F1** (per class) | `2 × precision × recall / (precision + recall)` |
| **Support** | Number of test samples in that class |

The confusion matrix plots **actual (rows) vs predicted (columns)** counts on the test set.

**Regressors (severity regressor, 7d forecaster, 14d forecaster)**

| Metric | Formula / meaning |
|--------|-------------------|
| **MAE** | Mean absolute error — average `\|predicted − actual\|` |
| **RMSE** | Root mean squared error — penalises large errors more |
| **R²** | Coefficient of determination — 1.0 is perfect, 0.0 is baseline mean |

Scatter plots show each test sample as a point: x = ground truth, y = model prediction. The dashed diagonal is perfect prediction.

### 7. Live inference (image upload)

When you upload an image via `POST /ml/detect`:

| Output | How it is computed |
|--------|-------------------|
| **Defect type** | Classifier argmax over 3 types |
| **Confidence** | Softmax probability of the winning class (from `predict_proba`) |
| **Class probabilities** | Full probability distribution shown in the UI |
| **Damage severity** | Severity regressor prediction on the same image features, clipped to `[0, 1]` |
| **Repair estimate** | `0.25 + severity × 3.2` hours (scale from RDD2022 damage-size statistics) |

Sample images in `backend/ml/sample_images/` are exported from the **test split** so confidence values reflect real generalisation, not memorised training crops.

### 8. Severity forecast (defect modal)

For each registered defect, `GET /ml/forecast/{id}` builds an 11-feature vector from the defect's current severity, type, traffic, location, risk, and age (days since `timestamp`), then:

| Output | How it is computed |
|--------|-------------------|
| **predicted_severity_7d** | 7-day LightGBM model output, clipped to `[0, 1]` |
| **predicted_severity_14d** | 14-day LightGBM model output, clipped to `[0, 1]` |
| **risk_trend** | `rising` if 7d prediction > current + 0.04; `falling` if below −0.04; else `stable` |
| **days_until_critical** | Linear extrapolation to severity ≥ 0.72; `null` if not expected to reach critical within horizon |

### 9. Dashboard route & priority metrics

These are computed at runtime from registered defects — not from ML evaluation.

**Priority score** (per defect, weights configurable via sliders):

```text
score = w_sev × severity + w_traffic × traffic + w_location × location + w_risk × risk
```

Weights are normalised to sum to 1 before scoring.

**Route optimisation**

1. Defects ranked by priority score.
2. Top N stops selected (max 55).
3. Stop order chosen with A* distance on a k-nearest-neighbour graph, weighted by priority.
4. Road geometry fetched from OSRM for map display.

**Dashboard `/metrics`**

| Metric | Computation |
|--------|-------------|
| `total_defects` | Count of registered defects |
| `active_high_priority_defects` | Defects with severity ≥ 0.72 or priority score ≥ 0.72 |
| `total_distance` | Sum of A* segment distances for the optimised route (km) |
| `total_repair_time` | Route travel time + sum of per-defect repair hours |
| `estimated_cost` | `total_time × $185/h + total_distance × $1.35/km` |
| `resource_status` | Crew availability derived from high-priority ratio |

---

## Verification

```bash
python -m compileall backend
cd frontend
npm run build
```
