from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import train_test_split

from ml.config import (
    DEFECT_TYPES,
    MATRICES_DIR,
    SPLITS_DIR,
    SPLITS_META_PATH,
    TRAIN_INDICES_PATH,
    VAL_INDICES_PATH,
    TEST_INDICES_PATH,
)


def create_splits(
    labels: np.ndarray,
    *,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    indices = np.arange(len(labels))
    train_indices, holdout_indices = train_test_split(
        indices,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=labels,
    )
    holdout_labels = labels[holdout_indices]
    relative_test_size = test_size / (val_size + test_size)
    val_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=holdout_labels,
    )

    splits = {
        "train": np.sort(train_indices),
        "val": np.sort(val_indices),
        "test": np.sort(test_indices),
    }
    save_splits(splits, labels)
    return splits


def save_splits(splits: dict[str, np.ndarray], labels: np.ndarray) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TRAIN_INDICES_PATH, splits["train"])
    np.save(VAL_INDICES_PATH, splits["val"])
    np.save(TEST_INDICES_PATH, splits["test"])

    metadata = {
        "train_samples": int(len(splits["train"])),
        "val_samples": int(len(splits["val"])),
        "test_samples": int(len(splits["test"])),
        "total_samples": int(len(labels)),
        "split_ratio": {"train": 0.7, "val": 0.15, "test": 0.15},
        "random_state": 42,
        "class_counts": {
            defect_type: {
                "train": int((labels[splits["train"]] == defect_type).sum()),
                "val": int((labels[splits["val"]] == defect_type).sum()),
                "test": int((labels[splits["test"]] == defect_type).sum()),
            }
            for defect_type in DEFECT_TYPES
        },
    }
    SPLITS_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_splits() -> dict[str, np.ndarray]:
    if not splits_exist():
        raise FileNotFoundError(
            "Dataset splits not found. Run: python -m ml.build_matrices"
        )
    return {
        "train": np.load(TRAIN_INDICES_PATH),
        "val": np.load(VAL_INDICES_PATH),
        "test": np.load(TEST_INDICES_PATH),
    }


def splits_exist() -> bool:
    return (
        TRAIN_INDICES_PATH.exists()
        and VAL_INDICES_PATH.exists()
        and TEST_INDICES_PATH.exists()
    )


def export_split_samples(annotation_records: list[dict], splits: dict[str, np.ndarray]) -> None:
    for split_name, indices in splits.items():
        samples = [annotation_records[int(index)] for index in indices]
        output_path = MATRICES_DIR / f"{split_name}_samples.json"
        output_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
