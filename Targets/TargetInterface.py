from abc import ABC, abstractmethod
import pandas as pd
from Features import *
from Targets import *

class TargetInterface:

    @abstractmethod
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), *args, **kwargs) -> None:
        """
        Initialise l'indicateur avec les données fournies.
        
        :param data: DataFrame contenant les données nécessaires pour le calcul de l'indicateur.
        """
        pass

    @property
    def dependencies(self) -> list[TargetID]:
        """
        Liste des features nécessaires pour calculer cet indicateur.
        Par défaut vide.
        """
        return []

    @abstractmethod
    def calculate(self, resolved_dependencies: dict[TargetID, TargetInterface] = None) -> None:
        """
        Calcule la valeur de l'indicateur.
        
        :return: La valeur calculée de l'indicateur.
        """
        pass

    @abstractmethod
    def update(self, new_data: pd.DataFrame, training_data_shift : int) -> None:
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