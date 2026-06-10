from __future__ import annotations

from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent
DATA_DIR = ML_ROOT / "data"
RDD2022_DIR = DATA_DIR / "rdd2022"
MATRICES_DIR = DATA_DIR / "matrices"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = ML_ROOT / "artifacts"
EVALUATION_DIR = ML_ROOT / "evaluation" / "reports"

TRAIN_INDICES_PATH = SPLITS_DIR / "train_indices.npy"
VAL_INDICES_PATH = SPLITS_DIR / "val_indices.npy"
TEST_INDICES_PATH = SPLITS_DIR / "test_indices.npy"
SPLITS_META_PATH = SPLITS_DIR / "splits_metadata.json"

DEFECT_MODEL_PATH = ARTIFACTS_DIR / "defect_classifier.joblib"
SEVERITY_MODEL_PATH = ARTIFACTS_DIR / "severity_forecaster.joblib"
SEVERITY_REGRESSOR_PATH = ARTIFACTS_DIR / "severity_regressor.joblib"
DEFECT_LABELS_PATH = ARTIFACTS_DIR / "defect_labels.json"

CLASSIFIER_X_PATH = MATRICES_DIR / "classifier_X.npy"
CLASSIFIER_Y_PATH = MATRICES_DIR / "classifier_y.npy"
SEVERITY_Y_CURRENT_PATH = MATRICES_DIR / "severity_y_current.npy"
SEVERITY_X_PATH = MATRICES_DIR / "severity_X.npy"
SEVERITY_Y7_PATH = MATRICES_DIR / "severity_y7.npy"
SEVERITY_Y14_PATH = MATRICES_DIR / "severity_y14.npy"
MATRICES_META_PATH = MATRICES_DIR / "metadata.json"

CRITICAL_SEVERITY = 0.72
FORECAST_HORIZON_DAYS = 7

RDD2022_COUNTRIES = ("Czech", "Japan", "India", "United_States", "China_MotorBike", "China_Drone")

RDD2022_DOWNLOADS = {
    "Czech": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Czech.zip",
    "Japan": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Japan.zip",
    "India": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip",
}

DEFAULT_RDD_COUNTRIES = ("Czech", "Japan")

RDD_TO_DEFECT_TYPE = {
    "D00": "Longitudinal crack",
    "D10": "Longitudinal crack",
    "D20": "Surface delamination",
    "D40": "Pothole",
}

DEFECT_TYPES = sorted(set(RDD_TO_DEFECT_TYPE.values()))

REPAIR_HOURS_PER_SEVERITY = 3.2

COUNTRY_TRAFFIC_PROXY = {
    "Czech": 0.52,
    "Japan": 0.78,
    "India": 0.71,
    "United_States": 0.62,
    "China_MotorBike": 0.66,
    "China_Drone": 0.48,
    "Unknown": 0.55,
}

COUNTRY_LOCATION_PROXY = {
    "Czech": 0.58,
    "Japan": 0.82,
    "India": 0.64,
    "United_States": 0.71,
    "China_MotorBike": 0.55,
    "China_Drone": 0.45,
    "Unknown": 0.5,
}
