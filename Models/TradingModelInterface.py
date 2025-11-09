from abc import ABC, abstractmethod
from typing import Any, Dict, Set

from Features import FeatureManager, TargetManager
from Features.FeatureID import FeatureID


class TradingModelInterface(ABC):
    """
    Interface que tes modèles RandomForest et XGBoost peuvent implémenter.
    Cette interface permet aux modèles de déclarer leurs besoins en features
    sans être couplés au dispatcher.
    """
    
    @abstractmethod
    def get_required_features(self) -> Set[FeatureID]:
        """Retourne les features dont ce modèle a besoin."""
        pass
    
    @abstractmethod
    def predict(self, features: Dict[FeatureID, Any]) -> Any:
        """Fait une prédiction à partir des features."""
        pass

    @abstractmethod
    def train(self) -> None:
        """Entraîne le modèle avec les données d'entraînement."""
        pass

    @abstractmethod
    def test(self) -> float:
        """Teste le modèle avec les données de test et retourne une métrique de performance."""
        pass

    @abstractmethod
    def setFeatureManager(self, feature_manager : FeatureManager) -> None:
        """Optionnel : Permet de définir un gestionnaire de features si nécessaire."""
        pass
    
    @abstractmethod
    def setTargetManager(self, target_manager: TargetManager) -> None:
        """Optionnel : Permet de définir un gestionnaire de cibles si nécessaire."""
        pass