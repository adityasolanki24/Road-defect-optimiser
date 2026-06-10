from services.ranking_service import calculate_priority_score, prioritize_defects
from services.route_service import route_service


def optimize_route(defects, weights):
    return route_service.compute_route(defects, weights)


__all__ = ["calculate_priority_score", "prioritize_defects", "optimize_route"]
