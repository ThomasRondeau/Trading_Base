import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Features.FeatureID import FeatureID
from Indicators.IndicatorInterface import IndicatorInterface

class Vwap(IndicatorInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), std_multiplier: float = 3):
        """
        Initialise l'indicateur VWAP avec les données fournies.

        :param data: DataFrame contenant les colonnes 'close', 'volume', et 'date'.
        :param std_multiplier: Multiplicateur de l'écart-type pour les bandes (ex: 1.0 = ±1σ).
        :param window: Taille de la fenêtre pour le calcul de l'écart-type.
        """
        self.data = data
        self.std_multiplier = std_multiplier
        self.point_of_control = None
        self.superior_limit = None
        self.far_superior_limit = None
        self.far_inferior_limit = None
        self.inferior_limit = None

    def calculate(self, resolved_dependencies: dict[FeatureID, IndicatorInterface] = None):
        """
        Calcule le VWAP et les bandes d'écart-type.
        """
        if 'close' not in self.data or 'low' not in self.data or 'high' not in self.data or 'volume' not in self.data:
            raise ValueError("Erreur dans VWAP : Les colonnes 'close', 'high', 'low' et 'volume' doivent être présentes dans les données.")

        # Calcul du VWAP cumulé
        typical_price = (self.data['close'] + self.data['high'] + self.data['low']) / 3
        volume = self.data['volume']

        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        self.point_of_control = vwap

        # Calcul de l'écart-type sur une fenêtre glissante autour du VWAP
        std = vwap.expanding(min_periods=2).std()

        # Bandes supérieures et inférieures
        self.far_superior_limit = vwap + (std * self.std_multiplier * 2)
        self.superior_limit = vwap + (std * self.std_multiplier)
        self.inferior_limit = vwap - (std * self.std_multiplier)
        self.far_inferior_limit = vwap - (std * self.std_multiplier * 2)

    def update(self, new_data: pd.DataFrame):
        """
        Met à jour le VWAP avec de nouvelles données.

        :param new_data: DataFrame contenant les nouvelles données à ajouter.
        """
        if not all(col in new_data for col in ['close', 'high', 'low', 'volume']):
            raise ValueError("Erreur dans VWAP : Les colonnes 'close', 'high', 'low' et 'volume' doivent être présentes dans les nouvelles données.")

        self.data = new_data

    def clear_cache(self):
        """
        Vide le cache de l'indicateur.
        Utilisé pour réinitialiser l'état de l'indicateur.
        """
        self.point_of_control = None
        self.superior_limit = None
        self.far_superior_limit = None
        self.far_inferior_limit = None
        self.inferior_limit = None
        self.data = pd.DataFrame()