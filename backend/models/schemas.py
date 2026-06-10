from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def make_defect_id() -> str:
    return f"defect_{uuid4().hex[:8]}"


class DefectCreate(BaseModel):
    id: str = Field(default_factory=make_defect_id)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    severity: float = Field(ge=0, le=1)
    defect_type: str = Field(min_length=2, max_length=80)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    traffic_density: float = Field(ge=0, le=1)
    location_importance: float = Field(ge=0, le=1)
    risk_factor: float = Field(ge=0, le=1)
    estimated_repair_time_hours: float = Field(default=1.5, ge=0.25, le=8)


class Defect(DefectCreate):
    pass


class Weights(BaseModel):
    severity: float = Field(default=0.4, ge=0, le=1)
    traffic_density: float = Field(default=0.3, ge=0, le=1)
    location_importance: float = Field(default=0.2, ge=0, le=1)
    risk_factor: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def require_positive_weight(self) -> "Weights":
        if (
            self.severity
            + self.traffic_density
            + self.location_importance
            + self.risk_factor
            <= 0
        ):
            raise ValueError("At least one ranking weight must be greater than 0")
        return self

    def normalized(self) -> "Weights":
        total = (
            self.severity
            + self.traffic_density
            + self.location_importance
            + self.risk_factor
        )
        return Weights(
            severity=self.severity / total,
            traffic_density=self.traffic_density / total,
            location_importance=self.location_importance / total,
            risk_factor=self.risk_factor / total,
        )


class RoutePoint(BaseModel):
    defect_id: str
    latitude: float
    longitude: float
    priority_score: float = Field(ge=0)
    severity: float = Field(ge=0, le=1)
    estimated_repair_time_hours: float = Field(ge=0.25, le=8)
    sequence: int = Field(ge=1)


class OptimisedRoute(BaseModel):
    route: List[RoutePoint]
    total_distance: float = Field(ge=0)
    total_time: float = Field(ge=0)
    estimated_cost: float = Field(ge=0)


class ResourceStatus(BaseModel):
    crew_count: int = Field(ge=0)
    available_crews: int = Field(ge=0)
    availability_percent: float = Field(ge=0, le=100)
    daily_cost: float = Field(ge=0)


class Metrics(BaseModel):
    total_defects: int = Field(ge=0)
    active_high_priority_defects: int = Field(ge=0)
    total_repair_time: float = Field(ge=0)
    total_distance: float = Field(ge=0)
    estimated_cost: float = Field(ge=0)
    cost_reduction_percent: float = Field(ge=0, le=100)
    fuel_savings_percent: float = Field(ge=0, le=100)
    resource_status: ResourceStatus


class MLStatus(BaseModel):
    ready: bool
    defect_classifier_loaded: bool
    severity_forecaster_loaded: bool
    severity_regressor_loaded: bool = False
    error: str | None = None


class DefectDetectionResult(BaseModel):
    defect_type: str
    confidence: float = Field(ge=0, le=1)
    severity: float = Field(ge=0, le=1)
    estimated_repair_time_hours: float = Field(ge=0.25, le=8)
    class_probabilities: dict[str, float]
    latitude: float | None = None
    longitude: float | None = None
    created_defect: Defect | None = None


class SeverityForecast(BaseModel):
    defect_id: str
    current_severity: float = Field(ge=0, le=1)
    predicted_severity_7d: float = Field(ge=0, le=1)
    predicted_severity_14d: float = Field(ge=0, le=1)
    days_until_critical: float | None = Field(default=None, ge=0)
    risk_trend: str
    age_days: float = Field(ge=0)
    forecasted_at: datetime
