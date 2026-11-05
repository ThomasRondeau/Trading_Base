import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Features import *

class Vwap(FeatureInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), std_multiplier: float = 3, session_tz: str = "America/New_York"):
        """
        Initialise l'indicateur VWAP avec les données fournies.

        :param data: DataFrame contenant les colonnes 'close', 'volume', et 'date'.
        :param std_multiplier: Multiplicateur de l'écart-type pour les bandes (ex: 1.0 = ±1σ).
        :param session_tz: Fuseau de la place de cotation, pour découper les séances.
        """
        self.data = data
        self.session_tz = session_tz
        self.std_multiplier = std_multiplier
        self.point_of_control = None
        self.superior_limit = None
        self.far_superior_limit = None
        self.far_inferior_limit = None
        self.inferior_limit = None

    def calculate(self, resolved_dependencies: dict[FeatureID, FeatureInterface] = None):
        """
        Calcule le VWAP et les bandes d'écart-type.
        """
        if 'close' not in self.data or 'low' not in self.data or 'high' not in self.data or 'volume' not in self.data:
            raise ValueError("Erreur dans VWAP : Les colonnes 'close', 'high', 'low' et 'volume' doivent être présentes dans les données.")

        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Erreur dans VWAP : un DatetimeIndex est nécessaire pour réinitialiser le VWAP à chaque séance.")

        # Le VWAP repart de zéro à chaque séance (date dans le fuseau de la place de cotation).
        index = self.data.index if self.data.index.tz is not None else self.data.index.tz_localize("UTC")
        session = pd.Series(index.tz_convert(self.session_tz).date, index=self.data.index)

        typical_price = (self.data['close'] + self.data['high'] + self.data['low']) / 3
        volume = self.data['volume']

        cum_volume = volume.groupby(session).cumsum()
        vwap = (typical_price * volume).groupby(session).cumsum() / cum_volume
        self.point_of_control = vwap

        # Écart-type du prix typique autour du VWAP, pondéré par le volume (bandes VWAP standard).
        mean_sq = (typical_price ** 2 * volume).groupby(session).cumsum() / cum_volume
        std = np.sqrt((mean_sq - vwap ** 2).clip(lower=0))

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