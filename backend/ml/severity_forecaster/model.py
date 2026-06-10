from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error

from ml.config import FORECAST_HORIZON_DAYS, SEVERITY_MODEL_PATH
from ml.datasets.rdd2022 import load_matrices, matrices_exist
from ml.evaluation.splits import load_splits
from ml.severity_forecaster.features import feature_names


def train_severity_forecaster(
    *,
    random_state: int = 42,
) -> dict[str, lgb.LGBMRegressor]:
    if not matrices_exist():
        raise FileNotFoundError("Run python -m ml.build_matrices before training")

    matrices = load_matrices()
    splits = load_splits()
    train_idx = splits["train"]
    val_idx = splits["val"]

    features = matrices["severity_X"]
    severity_7d = matrices["severity_y7"]
    severity_14d = matrices["severity_y14"]
    names = feature_names()

    x_train = features[train_idx]
    x_val = features[val_idx]
    y7_train = severity_7d[train_idx]
    y7_val = severity_7d[val_idx]
    y14_train = severity_14d[train_idx]
    y14_val = severity_14d[val_idx]

    params = dict(
        n_estimators=280,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=56,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        verbose=-1,
    )

    model_7d = lgb.LGBMRegressor(**params)
    model_14d = lgb.LGBMRegressor(**params)

    model_7d.fit(x_train, y7_train, feature_name=names)
    model_14d.fit(x_train, y14_train, feature_name=names)

    pred_7d = model_7d.predict(x_val)
    pred_14d = model_14d.predict(x_val)

    print(f"7-day validation MAE: {mean_absolute_error(y7_val, pred_7d):.4f}")
    print(f"14-day validation MAE: {mean_absolute_error(y14_val, pred_14d):.4f}")

    return {"model_7d": model_7d, "model_14d": model_14d}


def save_forecaster(models: dict[str, lgb.LGBMRegressor], path: Path | None = None) -> Path:
    target = path or SEVERITY_MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "feature_names": feature_names()}, target)
    return target


def load_forecaster(path: Path | None = None) -> dict:
    target = path or SEVERITY_MODEL_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"Severity forecaster not found at {target}. Run: python -m ml.train_all"
        )
    return joblib.load(target)


def predict_severity(features: np.ndarray, bundle: dict | None = None) -> dict[str, float]:
    artifact = bundle or load_forecaster()
    models: dict[str, lgb.LGBMRegressor] = artifact["models"]
    vector = features.reshape(1, -1)

    predicted_7d = float(
        np.clip(models["model_7d"].predict(vector, predict_disable_shape_check=True)[0], 0.0, 1.0)
    )
    predicted_14d = float(
        np.clip(models["model_14d"].predict(vector, predict_disable_shape_check=True)[0], 0.0, 1.0)
    )

    return {
        "predicted_severity_7d": round(predicted_7d, 4),
        "predicted_severity_14d": round(predicted_14d, 4),
        "horizon_days": FORECAST_HORIZON_DAYS,
    }


if __name__ == "__main__":
    trained = train_severity_forecaster()
    saved_path = save_forecaster(trained)
    print(f"Saved severity forecaster to {saved_path}")
