import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from Models.TradingModelInterface import TradingModelInterface
from Features.FeatureManager import FeatureManager
from Targets.TargetManager import TargetManager
from typing import Set
from Features.FeatureID import FeatureID

class ArimaModel(TradingModelInterface):
    """
    Modèle statistique ARIMA pour prédire le prochain retour de prix.
    """
    def __init__(self, order=(5,1,0)):
        self.feature_manager = None
        self.target_manager = None
        self.order = order
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
        try:
            # On prend les 50 derniers points pour la rapidité
            model = ARIMA(series[-50:], order=(2,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            current_price = series[-1]
            return float(yhat - current_price) # Retourne l'amplitude prédite
        except Exception as e:
            # print(f"ARIMA Error: {e}")
            return 0.0
