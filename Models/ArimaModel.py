import logging

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from Models.TradingModelInterface import TradingModelInterface
from Features.FeatureManager import FeatureManager
from Targets.TargetManager import TargetManager
from typing import Set
from Features.FeatureID import FeatureID

logger = logging.getLogger(__name__)


class ArimaModel(TradingModelInterface):
    """
    Modèle statistique ARIMA pour prédire le prochain retour de prix.
    """
    def __init__(self, order=(2, 1, 0), window: int = 50):
        self.feature_manager = None
        self.target_manager = None
        self.order = order
        self.window = window
        self.model_fit = None
        self.history = None

    def setFeatureManager(self, feature_manager: FeatureManager) -> None:
        self.feature_manager = feature_manager

    def setTargetManager(self, target_manager: TargetManager) -> None:
        self.target_manager = target_manager

    def train(self) -> None:
        """
        ARIMA s'entraîne sur les prix de clôture.
        """
        # On récupère les données via le feature manager (ou directement si besoin)
        # Ici on va utiliser les prix de clôture stockés dans le manager
        data = self.feature_manager.data['close'].values
        self.history = list(data)
        # On ne fit pas vraiment ici car ARIMA est souvent utilisé de manière glissante
        pass

    def test(self) -> float:
        return 0.0 # Score arbitraire

    def cross_validate(self, n_splits: int = 5):
        return 0.0, []

    def get_required_features(self) -> Set[FeatureID]:
        return set()

    def predict(self) -> float:
        """
        Prédit le prochain prix et retourne la direction attendue.
        """
        series = self.feature_manager.data['close'].values.astype(float)
        if len(series) < self.window:
            return 0.0
        try:
            model_fit = ARIMA(series[-self.window:], order=self.order).fit()
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning("ARIMA%s fit failed: %s", self.order, e)
            return 0.0
        yhat = float(model_fit.forecast()[0])
        return yhat - float(series[-1])  # Variation de prix prédite
