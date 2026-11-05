from __future__ import annotations

import logging

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from quantlab.features.feature_id import FeatureID
from quantlab.models.base import TradingModelInterface

logger = logging.getLogger(__name__)


class ArimaModel(TradingModelInterface):
    """ARIMA refitted on the last `window` closes at each prediction (rolling forecast)."""

    def __init__(self, order: tuple[int, int, int] = (2, 1, 0), window: int = 50) -> None:
        self.feature_manager = None
        self.target_manager = None
        self.order = order
        self.window = window

    def train(self) -> None:
        """Nothing to fit upfront: the model is refitted on a rolling window in `predict`."""

    def test(self) -> float:
        return 0.0

    def cross_validate(self, n_splits: int = 5) -> tuple[float, list[float]]:
        return 0.0, []

    def get_required_features(self) -> set[FeatureID]:
        return set()

    def predict(self) -> float:
        """Predicted price change for the next bar (0.0 when there is not enough data or the fit fails)."""
        if self.feature_manager is None:
            raise RuntimeError("ArimaModel: the feature manager must be set before predicting.")
        series = self.feature_manager.data["close"].values.astype(float)
        if len(series) < self.window:
            return 0.0
        try:
            model_fit = ARIMA(series[-self.window :], order=self.order).fit()
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning("ARIMA%s fit failed: %s", self.order, e)
            return 0.0
        yhat = float(model_fit.forecast()[0])
        return yhat - float(series[-1])
