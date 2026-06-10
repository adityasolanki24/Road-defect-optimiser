from __future__ import annotations

from fastapi import HTTPException, UploadFile

from models import Defect, DefectCreate, DefectDetectionResult, MLStatus, SeverityForecast
from services.defect_service import defect_service
from services.ml_service import ml_service


def get_status() -> MLStatus:
    return MLStatus(**ml_service.status())


def detect_defect(
    image: UploadFile,
    *,
    latitude: float | None,
    longitude: float | None,
    register_defect: bool,
    traffic_density: float,
    location_importance: float,
    risk_factor: float,
) -> DefectDetectionResult:
    if not ml_service.ready:
        raise HTTPException(status_code=503, detail=ml_service.error or "ML models unavailable")

    image_bytes = image.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    detection = ml_service.detect_from_image(
        image_bytes,
        latitude=latitude,
        longitude=longitude,
    )

    created_defect: Defect | None = None
    if register_defect:
        if latitude is None or longitude is None:
            raise HTTPException(
                status_code=400,
                detail="latitude and longitude are required when register_defect=true",
            )
        created_defect = defect_service.add_defect(
            DefectCreate(
                latitude=latitude,
                longitude=longitude,
                severity=detection["severity"],
                defect_type=detection["defect_type"],
                traffic_density=traffic_density,
                location_importance=location_importance,
                risk_factor=risk_factor,
                estimated_repair_time_hours=detection["estimated_repair_time_hours"],
            )
        )

    return DefectDetectionResult(
        defect_type=detection["defect_type"],
        confidence=detection["confidence"],
        severity=detection["severity"],
        estimated_repair_time_hours=detection["estimated_repair_time_hours"],
        class_probabilities=detection["class_probabilities"],
        latitude=latitude,
        longitude=longitude,
        created_defect=created_defect,
    )


def forecast_defect(defect_id: str) -> SeverityForecast:
    if not ml_service.ready:
        raise HTTPException(status_code=503, detail=ml_service.error or "ML models unavailable")

    defect = _find_defect(defect_id)
    return SeverityForecast(**ml_service.forecast_defect(defect))


def forecast_all() -> list[SeverityForecast]:
    if not ml_service.ready:
        raise HTTPException(status_code=503, detail=ml_service.error or "ML models unavailable")

    return [
        SeverityForecast(**forecast)
        for forecast in ml_service.forecast_all(defect_service.list_defects())
    ]


def _find_defect(defect_id: str):
    for defect in defect_service.list_defects():
        if defect.id == defect_id:
            return defect
    raise HTTPException(status_code=404, detail=f"Defect {defect_id} not found")
