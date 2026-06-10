from __future__ import annotations

from models import Defect, DefectCreate
from services.defect_service import defect_service


def list_defects() -> list[Defect]:
    return defect_service.list_defects()


def create_defect(defect: DefectCreate) -> Defect:
    return defect_service.add_defect(defect)
