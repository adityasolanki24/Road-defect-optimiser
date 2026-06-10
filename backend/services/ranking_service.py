from __future__ import annotations

from models import Defect, Weights


def calculate_priority_score(defect: Defect, weights: Weights) -> float:
    normalized = weights.normalized()
    score = (
        normalized.severity * defect.severity
        + normalized.traffic_density * defect.traffic_density
        + normalized.location_importance * defect.location_importance
        + normalized.risk_factor * defect.risk_factor
    )
    return round(score, 4)


def prioritize_defects(defects: list[Defect], weights: Weights) -> list[Defect]:
    return sorted(
        defects,
        key=lambda defect: calculate_priority_score(defect, weights),
        reverse=True,
    )
