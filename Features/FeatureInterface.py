from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from Features import *

class FeatureInterface(ABC):
    """
    Interface que les indicateurs doivent implémenter.
    Cette interface permet aux indicateurs de déclarer leurs besoins en données
    sans être couplés au dispatcher.
    """

    @abstractmethod
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), *args, **kwargs) -> None:
        """
        Initialise l'indicateur avec les données fournies.
        
        :param data: DataFrame contenant les données nécessaires pour le calcul de l'indicateur.
        """
        pass

    def dependencies(self) -> list[FeatureID]:
        """
        Liste des features nécessaires pour calculer cet indicateur.
        Par défaut vide.
        """
        return []

    @abstractmethod
    def calculate(self, resolved_dependencies: dict[FeatureID, "FeatureInterface"] = None):
        """
        Calcule la valeur de l'indicateur.
        
        :return: La valeur calculée de l'indicateur.
        """
        pass

    @abstractmethod
    def update(self, new_data: pd.DataFrame) -> None:
        """
        Met à jour l'indicateur avec de nouvelles données.
        
        :param new_data: DataFrame contenant les nouvelles données à ajouter.
        """
        pass
    
    @abstractmethod
    def clear_cache(self) -> None:
        """
        Vide le cache de l'indicateur.
        Utilisé pour réinitialiser l'état de l'indicateur.
        """
        pass