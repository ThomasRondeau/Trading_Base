from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from quantlab.targets.base import TargetInterface
from quantlab.targets.target_id import TargetID

logger = logging.getLogger(__name__)


class TargetManager:
    """Registry of targets with dependency resolution and a per-update cache."""

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        self.data = data if data is not None else pd.DataFrame()
        self.data_shift = 0
        self._cache: dict[TargetID, TargetInterface] = {}
        self._targets: dict[TargetID, TargetInterface] = {}

    def register_target(self, target_id: TargetID, target: TargetInterface) -> None:
        if target_id in self._targets:
            raise ValueError(f"Target {target_id} already registered.")
        self._targets[target_id] = target

    def get_target(self, target_id: TargetID) -> TargetInterface:
        if target_id in self._cache:
            return self._cache[target_id]
        if target_id not in self._targets:
            raise ValueError(f"Target {target_id} not registered.")

        target = self._targets[target_id]
        resolved_dependencies = {dep_id: self.get_target(dep_id) for dep_id in target.dependencies}
        target.calculate(resolved_dependencies)
        self._cache[target_id] = target
        return target

    def get_targets(self, target_ids: Iterable[TargetID]) -> dict[TargetID, TargetInterface]:
        return {tid: self.get_target(tid) for tid in target_ids if tid in self._targets}

    def update(self, new_data: pd.DataFrame, training_data_shift: int = 0) -> None:
        self.data = new_data
        self.clear_cache()
        self.data_shift = training_data_shift
        logger.debug("TargetManager updated with training_data_shift=%d", training_data_shift)
        for target in self._targets.values():
            target.update(new_data, training_data_shift)

    def clear_cache(self) -> None:
        self._cache.clear()
        for target in self._targets.values():
            target.clear_cache()
