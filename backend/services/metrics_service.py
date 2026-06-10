from __future__ import annotations

from models import Defect, Metrics, ResourceStatus, Weights
from services.ranking_service import calculate_priority_score
from services.route_service import route_service


BASE_CREW_COUNT = 6
DAILY_CREW_COST = 1650


def build_metrics(defects: list[Defect], weights: Weights, data_version: int) -> Metrics:
    route = route_service.compute_route(defects, weights, data_version)
    high_priority_count = sum(
        1
        for defect in defects
        if defect.severity >= 0.72 or calculate_priority_score(defect, weights) >= 0.72
    )

    active_ratio = min(1.0, high_priority_count / max(len(defects), 1))
    available_crews = max(0, BASE_CREW_COUNT - round(active_ratio * 2))
    availability_percent = (available_crews / BASE_CREW_COUNT * 100) if BASE_CREW_COUNT else 0

    return Metrics(
        total_defects=len(defects),
        active_high_priority_defects=high_priority_count,
        total_repair_time=route.total_time,
        total_distance=route.total_distance,
        estimated_cost=route.estimated_cost,
        cost_reduction_percent=0.0,
        fuel_savings_percent=0.0,
        resource_status=ResourceStatus(
            crew_count=BASE_CREW_COUNT,
            available_crews=available_crews,
            availability_percent=round(availability_percent),
            daily_cost=BASE_CREW_COUNT * DAILY_CREW_COST if defects else 0,
        ),
    )
