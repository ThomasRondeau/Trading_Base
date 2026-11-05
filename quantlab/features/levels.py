"""
Support / resistance trend lines drawn between local extrema.

WARNING: NOT CAUSAL. Extrema are detected with a centered window (they use `window` future bars)
and the scan is O(n^3). Use for charting only, never as a model feature or a live trading signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.features.base import FeatureInterface, ResolvedDependencies, require_columns


def _local_extrema(values: np.ndarray, window: int, kind: str) -> list[tuple[int, float]]:
    pick = min if kind == "min" else max
    return [
        (i, values[i])
        for i in range(window, len(values) - window)
        if values[i] == pick(values[i - window : i + window + 1])
    ]


def _score_lines(extrema: list[tuple[int, float]], min_length: int) -> pd.DataFrame:
    """Every line between two extrema at least `min_length` bars apart, scored by how close inner extrema are."""
    lines: list[dict[str, Any]] = []
    for idx_start, price_start in extrema:
        for idx_end, price_end in extrema:
            if idx_start < (idx_end - min_length):
                slope = (price_end - price_start) / (idx_end - idx_start)
                intercept = price_start - slope * idx_start
                points = {i: slope * i + intercept for i in range(idx_start, idx_end + 1)}

                score = 0.0
                for idx, price in extrema:
                    if idx_start < idx < idx_end:
                        distance = abs(points[idx] - price)
                        score += 1 / (distance * distance)

                lines.append(
                    {
                        "Score": score,
                        "idxStart": idx_start,
                        "idxEnd": idx_end,
                        "PriceStart": price_start,
                        "PriceEnd": price_end,
                        "points": points,
                    }
                )
    return pd.DataFrame(lines).sort_values(by="Score", ascending=False)


class Support(FeatureInterface):
    def __init__(self, data: pd.DataFrame | None = None, window: int = 10, min_len_support: int = 50) -> None:
        self.data = data if data is not None else pd.DataFrame()
        self.window = window
        self.min_len_support = min_len_support
        self.supports: pd.DataFrame | list[Any] = []

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        require_columns(self.data, ["low", "close"], "Support")
        minima = _local_extrema(self.data["low"].values, self.window, "min")
        if minima:
            self.supports = _score_lines(minima, self.min_len_support)

    def update(self, new_data: pd.DataFrame) -> None:
        require_columns(new_data, ["low", "close"], "Support")
        self.data = new_data

    def clear_cache(self) -> None:
        self.supports = []
        self.data = pd.DataFrame()


class Resistance(FeatureInterface):
    def __init__(self, data: pd.DataFrame | None = None, window: int = 10, min_len_resistance: int = 50) -> None:
        self.data = data if data is not None else pd.DataFrame()
        self.window = window
        self.min_len_resistance = min_len_resistance
        self.resistances: pd.DataFrame | list[Any] = []

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        require_columns(self.data, ["high", "close"], "Resistance")
        maxima = _local_extrema(self.data["high"].values, self.window, "max")
        if maxima:
            self.resistances = _score_lines(maxima, self.min_len_resistance)

    def update(self, new_data: pd.DataFrame) -> None:
        require_columns(new_data, ["high", "close"], "Resistance")
        self.data = new_data

    def clear_cache(self) -> None:
        self.resistances = []
        self.data = pd.DataFrame()
