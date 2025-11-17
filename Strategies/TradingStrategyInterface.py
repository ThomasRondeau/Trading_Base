from abc import ABC, abstractmethod
from typing import Dict, Any

class TradingStrategyInterface(ABC):
    """
    Interface que les stratégies de trading doivent implémenter.
    """
    @abstractmethod
    def __init__(self) -> None:
        """
        Initialise la stratégie de trading.
        """
        pass

    @abstractmethod
    def train(self, training_data: Dict[str, Any]) -> None:
        """
        Entraîne la stratégie de trading avec les données d'entraînement.

        :param training_data: Dictionnaire contenant les données d'entraînement.
        """
        pass

    @abstractmethod
    def test(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Teste la stratégie de trading avec les données de test.

        :param test_data: Dictionnaire contenant les données de test.
        :return: Dictionnaire contenant les résultats des tests.
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """Exécute la stratégie de trading."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Arrête la stratégie de trading."""
        pass

    @abstractmethod
    def get_status(self) -> str:
        """
        Retourne l'état actuel de la stratégie de trading.

        :return: état de la stratégie.
        """
        pass

