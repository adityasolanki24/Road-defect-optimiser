from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

from ml.config import (
    CLASSIFIER_X_PATH,
    CLASSIFIER_Y_PATH,
    COUNTRY_LOCATION_PROXY,
    COUNTRY_TRAFFIC_PROXY,
    DEFAULT_RDD_COUNTRIES,
    FORECAST_HORIZON_DAYS,
    MATRICES_DIR,
    MATRICES_META_PATH,
    RDD2022_DIR,
    RDD2022_DOWNLOADS,
    RDD_TO_DEFECT_TYPE,
    REPAIR_HOURS_PER_SEVERITY,
    SEVERITY_X_PATH,
    SEVERITY_Y7_PATH,
    SEVERITY_Y14_PATH,
    SEVERITY_Y_CURRENT_PATH,
    SPLITS_DIR,
)
from ml.defect_classifier.features import build_feature_matrix
from ml.evaluation.splits import create_splits, export_split_samples


@dataclass(frozen=True)
class RDDAnnotation:
    image_path: Path
    country: str
    rdd_label: str
    defect_type: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def download_country(country: str, force: bool = False) -> Path:
    if country not in RDD2022_DOWNLOADS:
        raise ValueError(f"No download URL configured for country: {country}")

    zip_path = RDD2022_DIR / f"RDD2022_{country}.zip"
    extract_dir = RDD2022_DIR / country

    if extract_dir.exists() and any(extract_dir.rglob("*.jpg")) and not force:
        return extract_dir

    RDD2022_DIR.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists() or force:
        print(f"Downloading RDD2022 {country}...")
        urlretrieve(RDD2022_DOWNLOADS[country], zip_path)
        print(f"Saved to {zip_path}")

    if not extract_dir.exists() or force:
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(RDD2022_DIR)

    return extract_dir


def _parse_xml(xml_path: Path, country: str) -> list[RDDAnnotation]:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    if not filename:
        return []

    image_dir = xml_path.parent.parent / "images"
    image_path = image_dir / filename
    if not image_path.exists():
        image_candidates = list(xml_path.parent.parent.parent.rglob(filename))
        if not image_candidates:
            return []
        image_path = image_candidates[0]
    annotations: list[RDDAnnotation] = []

    for obj in root.findall("object"):
        rdd_label = (obj.findtext("name") or "").strip().upper()
        defect_type = RDD_TO_DEFECT_TYPE.get(rdd_label)
        if not defect_type:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = int(float(bbox.findtext("xmin", "0")))
        ymin = int(float(bbox.findtext("ymin", "0")))
        xmax = int(float(bbox.findtext("xmax", "0")))
        ymax = int(float(bbox.findtext("ymax", "0")))

        if xmax <= xmin or ymax <= ymin:
            continue

        annotations.append(
            RDDAnnotation(
                image_path=image_path,
                country=country,
                rdd_label=rdd_label,
                defect_type=defect_type,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )

    return annotations


def collect_annotations(countries: tuple[str, ...] = DEFAULT_RDD_COUNTRIES) -> list[RDDAnnotation]:
    annotations: list[RDDAnnotation] = []

    for country in countries:
        country_dir = download_country(country)
        xml_files = list(country_dir.rglob("*.xml"))
        print(f"{country}: found {len(xml_files)} annotation files")

        for xml_path in xml_files:
            annotations.extend(_parse_xml(xml_path, country))

    print(f"Collected {len(annotations)} labelled damage crops from RDD2022")
    return annotations


def _crop_annotation(image: Image.Image, annotation: RDDAnnotation, padding: float = 0.12) -> Image.Image:
    width, height = image.size
    box_w = annotation.xmax - annotation.xmin
    box_h = annotation.ymax - annotation.ymin
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    left = max(0, annotation.xmin - pad_x)
    top = max(0, annotation.ymin - pad_y)
    right = min(width, annotation.xmax + pad_x)
    bottom = min(height, annotation.ymax + pad_y)

    return image.crop((left, top, right, bottom))


def _bbox_severity(annotation: RDDAnnotation, image: Image.Image) -> float:
    width, height = image.size
    area_ratio = ((annotation.xmax - annotation.xmin) * (annotation.ymax - annotation.ymin)) / max(
        width * height, 1
    )
    aspect = (annotation.xmax - annotation.xmin) / max(annotation.ymax - annotation.ymin, 1)
    aspect_factor = min(aspect, 1 / max(aspect, 1e-6))
    severity = 0.25 + np.sqrt(area_ratio) * 0.55 + aspect_factor * 0.12
    return float(np.clip(severity, 0.08, 0.98))


def _encode_defect_type(defect_type: str, defect_types: list[str]) -> list[float]:
    return [float(defect_type == name) for name in defect_types]


def _type_growth_targets(
    severities_by_type: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    growth_7d: dict[str, float] = {}
    growth_14d: dict[str, float] = {}

    for defect_type, values in severities_by_type.items():
        if len(values) < 10:
            growth_7d[defect_type] = 0.06
            growth_14d[defect_type] = 0.11
            continue

        p50 = float(np.percentile(values, 50))
        p75 = float(np.percentile(values, 75))
        p90 = float(np.percentile(values, 90))
        growth_7d[defect_type] = max(0.02, (p75 - p50) * 0.85)
        growth_14d[defect_type] = max(0.04, (p90 - p50) * 0.9)

    return growth_7d, growth_14d


def build_matrices(
    countries: tuple[str, ...] = DEFAULT_RDD_COUNTRIES,
    *,
    max_samples: int | None = None,
) -> dict[str, int]:
    annotations = collect_annotations(countries)
    if not annotations:
        raise RuntimeError("No RDD2022 annotations found. Check downloads under ml/data/rdd2022/")

    if max_samples is not None and len(annotations) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(annotations), size=max_samples, replace=False)
        annotations = [annotations[index] for index in sorted(indices)]

    defect_types = sorted(set(RDD_TO_DEFECT_TYPE.values()))
    classifier_images: list[Image.Image] = []
    classifier_labels: list[str] = []
    annotation_records: list[dict[str, str | float]] = []

    severity_rows: list[list[float]] = []
    severity_current: list[float] = []
    severity_types: list[str] = []

    image_cache: dict[Path, Image.Image] = {}

    for annotation in annotations:
        if annotation.image_path not in image_cache:
            image_cache[annotation.image_path] = Image.open(annotation.image_path).convert("RGB")
        image = image_cache[annotation.image_path]
        crop = _crop_annotation(image, annotation)

        classifier_images.append(crop)
        classifier_labels.append(annotation.defect_type)
        annotation_records.append(
            {
                "image_path": str(annotation.image_path),
                "defect_type": annotation.defect_type,
                "xmin": annotation.xmin,
                "ymin": annotation.ymin,
                "xmax": annotation.xmax,
                "ymax": annotation.ymax,
            }
        )

        current_severity = _bbox_severity(annotation, image)
        traffic = COUNTRY_TRAFFIC_PROXY.get(annotation.country, COUNTRY_TRAFFIC_PROXY["Unknown"])
        location = COUNTRY_LOCATION_PROXY.get(annotation.country, COUNTRY_LOCATION_PROXY["Unknown"])
        risk = float(np.clip(current_severity * 0.65 + traffic * 0.35, 0.15, 0.98))
        age_days = float(np.clip((1.0 - current_severity) * 28.0, 0.0, 45.0))

        row = [
            current_severity,
            traffic,
            location,
            risk,
            age_days,
            current_severity * traffic,
            current_severity * risk,
            traffic * risk,
        ]
        row.extend(_encode_defect_type(annotation.defect_type, defect_types))

        severity_rows.append(row)
        severity_current.append(current_severity)
        severity_types.append(annotation.defect_type)

    classifier_X = build_feature_matrix(classifier_images)
    classifier_y = np.array(classifier_labels)

    severities_by_type: dict[str, list[float]] = {name: [] for name in defect_types}
    for defect_type, severity in zip(severity_types, severity_current, strict=True):
        severities_by_type[defect_type].append(severity)

    growth_7d, growth_14d = _type_growth_targets(
        {key: np.array(values, dtype=np.float32) for key, values in severities_by_type.items()}
    )

    severity_y7: list[float] = []
    severity_y14: list[float] = []

    for defect_type, current in zip(severity_types, severity_current, strict=True):
        severity_y7.append(float(np.clip(current + growth_7d[defect_type], 0.0, 1.0)))
        severity_y14.append(float(np.clip(current + growth_14d[defect_type], 0.0, 1.0)))

    severity_X = np.array(severity_rows, dtype=np.float32)
    severity_y7_arr = np.array(severity_y7, dtype=np.float32)
    severity_y14_arr = np.array(severity_y14, dtype=np.float32)
    severity_y_current_arr = np.array(severity_current, dtype=np.float32)

    MATRICES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CLASSIFIER_X_PATH, classifier_X)
    np.save(CLASSIFIER_Y_PATH, classifier_y)
    np.save(SEVERITY_Y_CURRENT_PATH, severity_y_current_arr)
    np.save(SEVERITY_X_PATH, severity_X)
    np.save(SEVERITY_Y7_PATH, severity_y7_arr)
    np.save(SEVERITY_Y14_PATH, severity_y14_arr)

    splits = create_splits(classifier_y)
    export_split_samples(annotation_records, splits)

    repair_scale = float(REPAIR_HOURS_PER_SEVERITY)
    metadata = {
        "source": "RDD2022",
        "countries": list(countries),
        "samples": len(classifier_labels),
        "defect_types": defect_types,
        "rdd_mapping": RDD_TO_DEFECT_TYPE,
        "classifier_shape": list(classifier_X.shape),
        "severity_shape": list(severity_X.shape),
        "growth_7d": growth_7d,
        "growth_14d": growth_14d,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "repair_hours_per_severity": repair_scale,
        "severity_note": "Severity is derived from RDD2022 bounding-box damage area on each annotated crop.",
        "repair_note": "Repair hours scale linearly with predicted severity using RDD2022 damage-size statistics.",
        "splits_dir": str(SPLITS_DIR),
        "split_ratio": {"train": 0.7, "val": 0.15, "test": 0.15},
    }
    MATRICES_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    counts = {
        defect_type: int((classifier_y == defect_type).sum()) for defect_type in defect_types
    }
    print("Class distribution:", counts)
    print(f"Saved matrices to {MATRICES_DIR}")

    return {"samples": len(classifier_labels), **counts}


def matrices_exist() -> bool:
    required = (
        CLASSIFIER_X_PATH,
        CLASSIFIER_Y_PATH,
        SEVERITY_Y_CURRENT_PATH,
        SEVERITY_X_PATH,
        SEVERITY_Y7_PATH,
        SEVERITY_Y14_PATH,
        MATRICES_META_PATH,
    )
    return all(path.exists() for path in required)


def load_matrices() -> dict[str, np.ndarray]:
    if not matrices_exist():
        raise FileNotFoundError(
            "Training matrices not found. Run: python -m ml.build_matrices"
        )

    return {
        "classifier_X": np.load(CLASSIFIER_X_PATH),
        "classifier_y": np.load(CLASSIFIER_Y_PATH, allow_pickle=True),
        "severity_y_current": np.load(SEVERITY_Y_CURRENT_PATH),
        "severity_X": np.load(SEVERITY_X_PATH),
        "severity_y7": np.load(SEVERITY_Y7_PATH),
        "severity_y14": np.load(SEVERITY_Y14_PATH),
    }
