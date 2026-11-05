"""
Contract of a target (label).

Look-ahead protection: features are computed on `data[:-training_data_shift]` while targets receive
the full data and keep only the rows matching the features (`iloc[:-training_data_shift]`), so the
labels of the last feature rows can use real future prices without leaking into the features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from quantlab.targets.target_id import TargetID

ResolvedTargets = dict[TargetID, "TargetInterface"]


class TargetInterface(ABC):
    target: pd.Series | None

    @property
    def dependencies(self) -> list[TargetID]:
        """Targets required to compute this one (none by default)."""
        return []

    @abstractmethod
    def calculate(self, resolved_dependencies: ResolvedTargets | None = None) -> None:
        """Compute `self.target`."""

    @abstractmethod
    def update(self, new_data: pd.DataFrame, training_data_shift: int) -> None:
        """Replace the input data and the look-ahead shift."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Reset `self.target`, the data and the shift."""


class CloseBasedTarget(TargetInterface):
    """Shared state handling for targets computed from the close price."""

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        self.data = data if data is not None else pd.DataFrame()
        self.target = None
        self.training_data_shift = 0

    def _trim(self, full_target: pd.Series) -> pd.Series:
        """Keep only the rows aligned with the features (drop the look-ahead safety window)."""
        if self.training_data_shift > 0:
            return full_target.iloc[: -self.training_data_shift]
        return full_target

    def update(self, new_data: pd.DataFrame, training_data_shift: int) -> None:
        if "close" not in new_data:
            raise ValueError(f"{type(self).__name__}: the 'close' column is required.")
        self.data = new_data
        self.training_data_shift = training_data_shift

    def clear_cache(self) -> None:
        self.target = None
        self.data = pd.DataFrame()
        self.training_data_shift = 0
