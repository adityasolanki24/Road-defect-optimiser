from __future__ import annotations

import random
import threading
from datetime import datetime, timezone

from models import Defect, DefectCreate


class DefectService:
    def __init__(self) -> None:
        self._defects: list[Defect] = []
        self._lock = threading.RLock()
        self._version = 0

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def list_defects(self) -> list[Defect]:
        with self._lock:
            return list(self._defects)

    def add_defect(self, defect: DefectCreate) -> Defect:
        created = Defect(**defect.model_dump())
        with self._lock:
            self._defects.append(created)
            self._version += 1
        return created

    def import_rdd_defects(self, defects: list[DefectCreate]) -> int:
        created_count = 0
        with self._lock:
            for defect in defects:
                self._defects.append(Defect(**defect.model_dump()))
                created_count += 1
            if created_count:
                self._version += 1
        return created_count


defect_service = DefectService()
