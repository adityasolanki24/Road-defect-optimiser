from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ml.config import CRITICAL_SEVERITY, DEFECT_TYPES, FORECAST_HORIZON_DAYS


def _encode_defect_type(defect_type: str) -> list[float]:
    return [float(defect_type == name) for name in DEFECT_TYPES]


def build_feature_vector(
    *,
    current_severity: float,
    defect_type: str,
    traffic_density: float,
    location_importance: float,
    risk_factor: float,
    age_days: float,
) -> np.ndarray:
    row = [
        current_severity,
        traffic_density,
        location_importance,
        risk_factor,
        age_days,
        current_severity * traffic_density,
        current_severity * risk_factor,
        traffic_density * risk_factor,
    ]
    row.extend(_encode_defect_type(defect_type))
    return np.array(row, dtype=np.float32)


def feature_names() -> list[str]:
    return [
        "current_severity",
        "traffic_density",
        "location_importance",
        "risk_factor",
        "age_days",
        "severity_x_traffic",
        "severity_x_risk",
        "traffic_x_risk",
        *[f"type_{name.replace(' ', '_').lower()}" for name in DEFECT_TYPES],
    ]


def age_days_from_timestamp(timestamp: datetime, reference: datetime | None = None) -> float:
    reference = reference or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    delta = reference - timestamp
    return max(0.0, delta.total_seconds() / 86400.0)


def days_until_critical(
    current_severity: float,
    predicted_severity: float,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> float | None:
    if current_severity >= CRITICAL_SEVERITY:
        return 0.0
    if predicted_severity < CRITICAL_SEVERITY:
        return None

    daily_rate = (predicted_severity - current_severity) / max(horizon_days, 1)
    if daily_rate <= 0:
        return None

    return round((CRITICAL_SEVERITY - current_severity) / daily_rate, 1)
