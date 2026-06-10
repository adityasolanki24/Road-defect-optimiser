from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder

from ml.config import DEFECT_TYPES, EVALUATION_DIR, SPLITS_META_PATH
from ml.datasets.rdd2022 import load_matrices
from ml.defect_classifier.model import load_classifier
from ml.defect_classifier.severity_regressor import load_severity_regressor
from ml.evaluation.plots import plot_class_distribution, plot_confusion_matrix, plot_regression_scatter
from ml.evaluation.splits import load_splits
from ml.severity_forecaster.model import load_forecaster


def run_evaluation(*, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or EVALUATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    matrices = load_matrices()
    splits = load_splits()
    labels = matrices["classifier_y"]

    classifier = load_classifier()
    severity_bundle = load_severity_regressor()
    forecaster_bundle = load_forecaster()

    label_encoder: LabelEncoder = classifier.label_encoder_  # type: ignore[attr-defined]
    class_names = [str(name) for name in label_encoder.classes_]

    test_idx = splits["test"]
    val_idx = splits["val"]

    x_test = matrices["classifier_X"][test_idx]
    y_test = labels[test_idx]
    y_pred_encoded = classifier.predict(x_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded.astype(int))

    plot_confusion_matrix(y_test, y_pred, class_names, output_dir)
    plot_class_distribution(labels, splits, class_names, output_dir)

    severity_model = severity_bundle["model"]
    y_severity_true = matrices["severity_y_current"][test_idx]
    y_severity_pred = severity_model.predict(x_test, predict_disable_shape_check=True)
    plot_regression_scatter(
        y_severity_true,
        y_severity_pred,
        title="Severity regressor — test set",
        filename="severity_regressor_scatter.png",
        output_dir=output_dir,
        xlabel="Actual severity (bbox)",
        ylabel="Predicted severity",
    )

    forecaster_models = forecaster_bundle["models"]
    x_forecast_test = matrices["severity_X"][test_idx]
    y7_true = matrices["severity_y7"][test_idx]
    y14_true = matrices["severity_y14"][test_idx]
    y7_pred = forecaster_models["model_7d"].predict(x_forecast_test)
    y14_pred = forecaster_models["model_14d"].predict(x_forecast_test)

    plot_regression_scatter(
        y7_true,
        y7_pred,
        title="Severity forecaster — 7-day test set",
        filename="severity_forecast_7d_scatter.png",
        output_dir=output_dir,
        xlabel="Actual 7-day severity",
        ylabel="Predicted 7-day severity",
    )
    plot_regression_scatter(
        y14_true,
        y14_pred,
        title="Severity forecaster — 14-day test set",
        filename="severity_forecast_14d_scatter.png",
        output_dir=output_dir,
        xlabel="Actual 14-day severity",
        ylabel="Predicted 14-day severity",
    )

    x_val = matrices["classifier_X"][val_idx]
    y_val = labels[val_idx]
    y_val_pred = label_encoder.inverse_transform(classifier.predict(x_val).astype(int))

    metrics = {
        "classifier": {
            "test": _classification_metrics(y_test, y_pred, class_names),
            "validation": _classification_metrics(y_val, y_val_pred, class_names),
        },
        "severity_regressor": {
            "test": _regression_metrics(y_severity_true, y_severity_pred),
            "validation": _regression_metrics(
                matrices["severity_y_current"][val_idx],
                severity_model.predict(x_val, predict_disable_shape_check=True),
            ),
        },
        "severity_forecaster_7d": {
            "test": _regression_metrics(y7_true, y7_pred),
            "validation": _regression_metrics(
                matrices["severity_y7"][val_idx],
                forecaster_models["model_7d"].predict(matrices["severity_X"][val_idx]),
            ),
        },
        "severity_forecaster_14d": {
            "test": _regression_metrics(y14_true, y14_pred),
            "validation": _regression_metrics(
                matrices["severity_y14"][val_idx],
                forecaster_models["model_14d"].predict(matrices["severity_X"][val_idx]),
            ),
        },
        "splits": json.loads(SPLITS_META_PATH.read_text(encoding="utf-8"))
        if SPLITS_META_PATH.exists()
        else {},
        "artifacts": {
            "confusion_matrix_png": str(output_dir / "confusion_matrix_classifier.png"),
            "confusion_matrix_csv": str(output_dir / "confusion_matrix_classifier.csv"),
            "class_distribution_png": str(output_dir / "class_distribution.png"),
            "severity_regressor_scatter_png": str(output_dir / "severity_regressor_scatter.png"),
            "severity_forecast_7d_scatter_png": str(output_dir / "severity_forecast_7d_scatter.png"),
            "severity_forecast_14d_scatter_png": str(output_dir / "severity_forecast_14d_scatter.png"),
        },
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Evaluation reports saved to {output_dir}")
    print(
        f"Classifier test accuracy: {metrics['classifier']['test']['accuracy']:.3f} | "
        f"macro F1: {metrics['classifier']['test']['macro_f1']:.3f}"
    )
    return metrics_path


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    report = classification_report(y_true, y_pred, labels=class_names, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class": {
            class_name: {
                "precision": round(float(report[class_name]["precision"]), 4),
                "recall": round(float(report[class_name]["recall"]), 4),
                "f1": round(float(report[class_name]["f1-score"]), 4),
                "support": int(report[class_name]["support"]),
            }
            for class_name in class_names
            if class_name in report
        },
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


if __name__ == "__main__":
    run_evaluation()
