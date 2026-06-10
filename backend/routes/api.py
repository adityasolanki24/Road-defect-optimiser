from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from controllers import defects_controller, metrics_controller, ml_controller, routes_controller
from models import (
    Defect,
    DefectCreate,
    DefectDetectionResult,
    Metrics,
    MLStatus,
    OptimisedRoute,
    SeverityForecast,
    Weights,
)


router = APIRouter()


@router.get("/defects", response_model=list[Defect])
async def get_defects() -> list[Defect]:
    return defects_controller.list_defects()


@router.post("/defects", response_model=Defect)
async def add_defect(defect: DefectCreate) -> Defect:
    return defects_controller.create_defect(defect)


@router.get("/optimised-route", response_model=OptimisedRoute)
async def get_optimised_route() -> OptimisedRoute:
    return routes_controller.get_route()


@router.post("/compute-route", response_model=OptimisedRoute)
async def compute_route(weights: Weights) -> OptimisedRoute:
    return routes_controller.compute_route(weights)


@router.get("/metrics", response_model=Metrics)
async def get_metrics() -> Metrics:
    return metrics_controller.get_metrics()


@router.get("/weights", response_model=Weights)
async def get_weights() -> Weights:
    return routes_controller.get_weights()


@router.get("/ml/status", response_model=MLStatus)
async def get_ml_status() -> MLStatus:
    return ml_controller.get_status()


@router.post("/ml/detect", response_model=DefectDetectionResult)
async def detect_defect_from_image(
    image: UploadFile = File(...),
    latitude: float | None = Form(default=None),
    longitude: float | None = Form(default=None),
    register_defect: bool = Form(default=False),
    traffic_density: float = Form(default=0.55),
    location_importance: float = Form(default=0.5),
    risk_factor: float = Form(default=0.6),
) -> DefectDetectionResult:
    return ml_controller.detect_defect(
        image,
        latitude=latitude,
        longitude=longitude,
        register_defect=register_defect,
        traffic_density=traffic_density,
        location_importance=location_importance,
        risk_factor=risk_factor,
    )


@router.get("/ml/forecast/{defect_id}", response_model=SeverityForecast)
async def get_defect_forecast(defect_id: str) -> SeverityForecast:
    return ml_controller.forecast_defect(defect_id)


@router.get("/ml/forecasts", response_model=list[SeverityForecast])
async def get_all_forecasts() -> list[SeverityForecast]:
    return ml_controller.forecast_all()
