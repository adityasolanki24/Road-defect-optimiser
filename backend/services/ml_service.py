from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

from PIL import Image

from ml.config import CRITICAL_SEVERITY, FORECAST_HORIZON_DAYS
from ml.defect_classifier.features import build_feature_matrix
from ml.defect_classifier.model import load_classifier, predict_defect
from ml.defect_classifier.severity_regressor import load_severity_regressor, predict_damage_severity
from ml.severity_forecaster.features import (
    age_days_from_timestamp,
    build_feature_vector,
    days_until_critical,
)
from ml.severity_forecaster.model import load_forecaster, predict_severity
from models import Defect


class MLService:
    def __init__(self) -> None:
        self._classifier = None
        self._forecaster = None
        self._severity_regressor = None
        self._ready = False
        self._error: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    def load_models(self) -> None:
        try:
            self._classifier = load_classifier()
            self._forecaster = load_forecaster()
            self._severity_regressor = load_severity_regressor()
            self._ready = True
            self._error = None
        except FileNotFoundError as exc:
            self._ready = False
            self._error = str(exc)

    def status(self) -> dict:
        return {
            "ready": self._ready,
            "defect_classifier_loaded": self._classifier is not None,
            "severity_forecaster_loaded": self._forecaster is not None,
            "severity_regressor_loaded": self._severity_regressor is not None,
            "error": self._error,
        }

    def detect_from_image(
        self,
        image_bytes: bytes,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict:
        if not self._ready or self._classifier is None or self._severity_regressor is None:
            raise RuntimeError(self._error or "ML models are not loaded")

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        detection = predict_defect(image, self._classifier)
        features = build_feature_matrix([image])
        damage = predict_damage_severity(features, self._severity_regressor)

        return {
            **detection,
            "severity": damage["severity"],
            "estimated_repair_time_hours": damage["estimated_repair_time_hours"],
            "latitude": latitude,
            "longitude": longitude,
        }

    def forecast_defect(self, defect: Defect) -> dict:
        if not self._ready or self._forecaster is None:
            raise RuntimeError(self._error or "ML models are not loaded")

        age_days = age_days_from_timestamp(defect.timestamp)
        features = build_feature_vector(
            current_severity=defect.severity,
            defect_type=defect.defect_type,
            traffic_density=defect.traffic_density,
            location_importance=defect.location_importance,
            risk_factor=defect.risk_factor,
            age_days=age_days,
        )

        prediction = predict_severity(features, self._forecaster)
        current = defect.severity
        predicted_7d = prediction["predicted_severity_7d"]
        predicted_14d = prediction["predicted_severity_14d"]

        delta = predicted_7d - current
        if delta > 0.04:
            trend = "rising"
        elif delta < -0.04:
            trend = "falling"
        else:
            trend = "stable"

        days_critical = days_until_critical(current, predicted_7d, FORECAST_HORIZON_DAYS)

        return {
            "defect_id": defect.id,
            "current_severity": current,
            "predicted_severity_7d": predicted_7d,
            "predicted_severity_14d": predicted_14d,
            "days_until_critical": days_critical,
            "risk_trend": trend,
            "age_days": round(age_days, 1),
            "forecasted_at": datetime.now(timezone.utc).isoformat(),
        }

    def forecast_all(self, defects: list[Defect]) -> list[dict]:
        return [self.forecast_defect(defect) for defect in defects]


ml_service = MLService()
