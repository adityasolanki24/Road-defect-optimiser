from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_dir: Path,
) -> tuple[Path, Path]:
    _ensure_dir(output_dir)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    csv_path = output_dir / "confusion_matrix_classifier.csv"
    header = "," + ",".join(labels)
    rows = [header]
    for label, row in zip(labels, matrix):
        rows.append(f"{label}," + ",".join(str(value) for value in row))
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels).plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
    )
    axis.set_title("Defect classifier — test set confusion matrix")
    figure.tight_layout()
    png_path = output_dir / "confusion_matrix_classifier.png"
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    return png_path, csv_path


def plot_class_distribution(
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    class_names: list[str],
    output_dir: Path,
) -> Path:
    _ensure_dir(output_dir)
    split_names = ["train", "val", "test"]
    counts = np.array(
        [
            [(labels[splits[name]] == class_name).sum() for class_name in class_names]
            for name in split_names
        ]
    )

    x = np.arange(len(class_names))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 6))
    for index, split_name in enumerate(split_names):
        axis.bar(x + index * width, counts[index], width=width, label=split_name.capitalize())

    axis.set_xticks(x + width)
    axis.set_xticklabels(class_names, rotation=15, ha="right")
    axis.set_ylabel("Sample count")
    axis.set_title("Class distribution by split")
    axis.legend()
    figure.tight_layout()
    png_path = output_dir / "class_distribution.png"
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    return png_path


def plot_regression_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    filename: str,
    output_dir: Path,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
) -> Path:
    _ensure_dir(output_dir)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(y_true, y_pred, alpha=0.45, edgecolors="none")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    figure.tight_layout()
    png_path = output_dir / filename
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    return png_path
