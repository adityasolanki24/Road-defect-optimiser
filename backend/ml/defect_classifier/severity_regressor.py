from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error

from ml.config import REPAIR_HOURS_PER_SEVERITY, SEVERITY_REGRESSOR_PATH
from ml.datasets.rdd2022 import load_matrices, matrices_exist
from ml.evaluation.splits import load_splits


def train_severity_regressor(*, random_state: int = 42) -> lgb.LGBMRegressor:
    if not matrices_exist():
        raise FileNotFoundError("Run python -m ml.build_matrices before training")

    matrices = load_matrices()
    splits = load_splits()
    train_idx = splits["train"]
    val_idx = splits["val"]

    features = matrices["classifier_X"]
    targets = matrices["severity_y_current"]

    x_train = features[train_idx]
    y_train = targets[train_idx]
    x_val = features[val_idx]
    y_val = targets[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=48,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_val, predict_disable_shape_check=True)
    print(f"Severity regressor validation MAE: {mean_absolute_error(y_val, predictions):.4f}")

    return model


def save_severity_regressor(model: lgb.LGBMRegressor, path: Path | None = None) -> Path:
    target = path or SEVERITY_REGRESSOR_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "repair_hours_per_severity": REPAIR_HOURS_PER_SEVERITY,
        },
        target,
    )
    return target


def load_severity_regressor(path: Path | None = None) -> dict:
    target = path or SEVERITY_REGRESSOR_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"Severity regressor not found at {target}. Run: python -m ml.train_all"
        )
    return joblib.load(target)


def predict_damage_severity(features: np.ndarray, bundle: dict | None = None) -> dict[str, float]:
    artifact = bundle or load_severity_regressor()
    model: lgb.LGBMRegressor = artifact["model"]
    scale = float(artifact["repair_hours_per_severity"])

    vector = features.reshape(1, -1)
    severity = float(np.clip(model.predict(vector, predict_disable_shape_check=True)[0], 0.0, 1.0))
    repair_hours = float(np.clip(0.25 + severity * scale, 0.25, 8.0))

    return {
        "severity": round(severity, 3),
        "estimated_repair_time_hours": round(repair_hours, 1),
    }
