from __future__ import annotations

from models import Metrics
from services.defect_service import defect_service
from services.metrics_service import build_metrics
from controllers.routes_controller import get_weights


def get_metrics() -> Metrics:
    defects = defect_service.list_defects()
    return build_metrics(defects, get_weights(), defect_service.version)
