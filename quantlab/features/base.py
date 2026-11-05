"""Contract every feature implements so `FeatureManager` can resolve and cache it."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import pandas as pd

from quantlab.features.feature_id import FeatureID

ResolvedDependencies = dict[FeatureID, "FeatureInterface"]


class FeatureInterface(ABC):
    """
    A feature declares the other features it needs (`dependencies`) without knowing the manager.
    The manager feeds data with `update`, resolves dependencies, then calls `calculate`.
    """

    data: pd.DataFrame

    def dependencies(self) -> Iterable[FeatureID]:
        """Features required to compute this one (none by default)."""
        return []

    @abstractmethod
    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        """Compute the feature values from `self.data` and the resolved dependencies."""

    @abstractmethod
    def update(self, new_data: pd.DataFrame) -> None:
        """Replace the input data."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Reset every computed value (and the input data)."""


def require_columns(data: pd.DataFrame, columns: Iterable[str], owner: str) -> None:
    missing = [c for c in columns if c not in data]
    if missing:
        raise ValueError(f"{owner}: missing columns {missing} in the data.")
