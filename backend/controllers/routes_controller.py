from __future__ import annotations

from models import OptimisedRoute, Weights
from services.defect_service import defect_service
from services.route_service import route_service


current_weights = Weights()


def get_route() -> OptimisedRoute:
    defects = defect_service.list_defects()
    return route_service.compute_route(defects, current_weights, defect_service.version)


def compute_route(weights: Weights) -> OptimisedRoute:
    global current_weights
    current_weights = weights.normalized()
    defects = defect_service.list_defects()
    return route_service.compute_route(defects, current_weights, defect_service.version)


def get_weights() -> Weights:
    return current_weights
