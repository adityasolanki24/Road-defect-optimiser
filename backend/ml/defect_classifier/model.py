from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml.config import DEFECT_LABELS_PATH, DEFECT_MODEL_PATH, DEFECT_TYPES
from ml.datasets.rdd2022 import load_matrices, matrices_exist
from ml.defect_classifier.features import build_feature_matrix
from ml.evaluation.splits import load_splits


def train_defect_classifier(
    *,
    random_state: int = 42,
) -> Pipeline:
    if not matrices_exist():
        raise FileNotFoundError("Run python -m ml.build_matrices before training")

    matrices = load_matrices()
    splits = load_splits()
    train_idx = splits["train"]
    val_idx = splits["val"]

    features = matrices["classifier_X"]
    labels = matrices["classifier_y"]

    x_train = features[train_idx]
    y_train = labels[train_idx]
    x_val = features[val_idx]
    y_val = labels[val_idx]

    label_encoder = LabelEncoder()
    label_encoder.fit(DEFECT_TYPES)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_depth=10,
                    learning_rate=0.06,
                    max_iter=220,
                    random_state=random_state,
                ),
            ),
        ]
    )

    y_train_encoded = label_encoder.transform(y_train)
    pipeline.fit(x_train, y_train_encoded)

    val_predictions = label_encoder.inverse_transform(
        pipeline.predict(x_val).astype(int)
    )
    print("Validation classification report:")
    print(classification_report(y_val, val_predictions))

    pipeline.label_encoder_ = label_encoder  # type: ignore[attr-defined]
    return pipeline


def save_classifier(pipeline: Pipeline, path: Path | None = None) -> Path:
    target = path or DEFECT_MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target)

    labels = list(getattr(pipeline, "label_encoder_").classes_)
    DEFECT_LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return target


def load_classifier(path: Path | None = None) -> Pipeline:
    target = path or DEFECT_MODEL_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"Defect classifier not found at {target}. Run: python -m ml.train_all"
        )
    return joblib.load(target)


def predict_defect(image: Image.Image, pipeline: Pipeline | None = None) -> dict:
    model = pipeline or load_classifier()
    label_encoder: LabelEncoder = model.label_encoder_  # type: ignore[attr-defined]

    features = build_feature_matrix([image])
    probabilities = model.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    defect_type = str(label_encoder.inverse_transform([best_index])[0])
    confidence = float(probabilities[best_index])

    return {
        "defect_type": defect_type,
        "confidence": round(confidence, 4),
        "class_probabilities": {
            str(label_encoder.inverse_transform([index])[0]): round(float(score), 4)
            for index, score in enumerate(probabilities)
        },
    }


if __name__ == "__main__":
    classifier = train_defect_classifier()
    saved_path = save_classifier(classifier)
    print(f"Saved defect classifier to {saved_path}")
