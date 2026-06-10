from __future__ import annotations

import numpy as np
from PIL import Image


def extract_image_features(image: Image.Image, size: int = 48) -> np.ndarray:
    resized = image.convert("RGB").resize((size, size))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    gray = array.mean(axis=2)

    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    channel_means = array.reshape(-1, 3).mean(axis=0)
    channel_stds = array.reshape(-1, 3).std(axis=0)

    return np.concatenate(
        [
            array.flatten(),
            gray.flatten(),
            channel_means,
            channel_stds,
            np.array([gx, gy, gray.std(), gray.max() - gray.min()], dtype=np.float32),
        ]
    )


def build_feature_matrix(images: list[Image.Image]) -> np.ndarray:
    return np.vstack([extract_image_features(image) for image in images])
