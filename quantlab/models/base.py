from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from quantlab.features.feature_id import FeatureID
from quantlab.features.manager import FeatureManager
from quantlab.targets.manager import TargetManager
from quantlab.targets.target_id import TargetID


class TradingModelInterface(ABC):
    """
    A model declares the features it needs and reads its inputs from the injected managers:
    it never receives raw data directly.
    """

    feature_manager: FeatureManager | None
    target_manager: TargetManager | None

    @abstractmethod
    def get_required_features(self) -> set[FeatureID]:
        """Features this model reads."""

    @abstractmethod
    def predict(self) -> Any:
        """Prediction on the last available row."""

    @abstractmethod
    def train(self) -> None:
        """Fit on all the data currently in the managers."""

    @abstractmethod
    def test(self) -> float:
        """Performance metric on the training data (in-sample: optimistic)."""

    def set_feature_manager(self, feature_manager: FeatureManager) -> None:
        self.feature_manager = feature_manager

    def set_target_manager(self, target_manager: TargetManager) -> None:
        self.target_manager = target_manager


class SklearnFeatureModel(TradingModelInterface):
    """
    Model built from one composite feature set (`feature_id`) and one target (`target_id`).
    Subclasses list the feature attributes (`feature_columns`) and the scoring function.
    """

    feature_id: FeatureID
    target_id: TargetID
    feature_columns: ClassVar[dict[str, str]]  # dataset column name -> attribute of the composite feature
    score: Callable[[np.ndarray, np.ndarray], float]

    def __init__(self, model: Any) -> None:
        self.feature_manager = None
        self.target_manager = None
        self.model = model

    def _managers(self) -> tuple[FeatureManager, TargetManager]:
        if self.feature_manager is None or self.target_manager is None:
            raise RuntimeError(f"{type(self).__name__}: feature and target managers must be set before use.")
        return self.feature_manager, self.target_manager

    def _build_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """X, y from the managers; rows with NaN/inf in any column are dropped."""
        feature_manager, target_manager = self._managers()
        features = feature_manager.get_feature(self.feature_id)
        target = target_manager.get_target(self.target_id)

        columns = {name: getattr(features, attr) for name, attr in self.feature_columns.items()}
        columns["target"] = target.target
        df = pd.DataFrame(columns)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        X = df.drop(columns="target").to_numpy()
        y = df["target"].to_numpy()
        return X, y

    def _last_row(self) -> np.ndarray:
        """Feature vector of the last bar, shaped (1, n_features)."""
        feature_manager, _ = self._managers()
        features = feature_manager.get_feature(self.feature_id)
        return np.array([[getattr(features, attr).iloc[-1] for attr in self.feature_columns.values()]])

    def get_required_features(self) -> set[FeatureID]:
        return {self.feature_id}

    def train(self) -> None:
        X, y = self._build_dataset()
        self.model.fit(X, y)

    def test(self) -> float:
        """In-sample score (data leakage: train data). Use `cross_validate` for an honest estimate."""
        X, y = self._build_dataset()
        return float(type(self).score(y, self.model.predict(X)))

    def cross_validate(self, n_splits: int = 5) -> tuple[float, list[float]]:
        """Time-ordered cross-validation (TimeSeriesSplit) of a clone of the configured estimator."""
        X, y = self._build_dataset()
        scores: list[float] = []
        for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
            model = clone(self.model)
            model.fit(X[train_idx], y[train_idx])
            scores.append(float(type(self).score(y[test_idx], model.predict(X[test_idx]))))
        return float(np.mean(scores)), scores
