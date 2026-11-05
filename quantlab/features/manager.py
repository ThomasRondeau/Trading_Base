from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from quantlab.features.base import FeatureInterface
from quantlab.features.feature_id import FeatureID


class FeatureManager:
    """Registry of feature instances with recursive dependency resolution and a per-update cache."""

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        self.data = data if data is not None else pd.DataFrame()
        self._cache: dict[FeatureID, FeatureInterface] = {}
        self._indicators: dict[FeatureID, FeatureInterface] = {}

    def register_indicator(self, feature_id: FeatureID, indicator: FeatureInterface) -> None:
        if feature_id in self._indicators:
            raise ValueError(f"Indicator {feature_id} already registered.")
        self._indicators[feature_id] = indicator

    def get_feature(self, feature_id: FeatureID) -> FeatureInterface:
        """Return the computed feature, from the cache or by computing it (and its dependencies)."""
        if feature_id in self._cache:
            return self._cache[feature_id]
        if feature_id not in self._indicators:
            raise ValueError(f"Indicator {feature_id} not registered.")

        indicator = self._indicators[feature_id]
        resolved_dependencies = {dep_id: self.get_feature(dep_id) for dep_id in indicator.dependencies()}
        indicator.calculate(resolved_dependencies)
        self._cache[feature_id] = indicator
        return indicator

    def get_features(self, feature_ids: Iterable[FeatureID]) -> dict[FeatureID, FeatureInterface]:
        return {fid: self.get_feature(fid) for fid in feature_ids if fid in self._indicators}

    def update(self, new_data: pd.DataFrame) -> None:
        """Push new data to every registered feature and invalidate the cache."""
        self.data = new_data
        self.clear_cache()
        for indicator in self._indicators.values():
            indicator.update(new_data)

    def clear_cache(self) -> None:
        self._cache.clear()
        for indicator in self._indicators.values():
            indicator.clear_cache()
