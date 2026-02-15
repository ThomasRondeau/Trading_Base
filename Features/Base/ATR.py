import pandas as pd
from Features.FeatureID import FeatureID
from Features.IndicatorInterface import IndicatorInterface

class ATR(IndicatorInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), window: int = 14):
        """
        Initialise le calculateur ATR avec les données fournies.

        :param data: DataFrame contenant les colonnes 'high', 'low', 'close' et 'date'.
        :param window: Période pour le calcul de l'ATR.
        """
        self.data = data
        self.window = window
        self.true_range = None
        self.atr = None

    def calculate(self, resolved_dependencies: dict[FeatureID, IndicatorInterface] = None):
        """
        Calcule l'ATR et l'ajoute à l'attribut `self.atr`.
        """
        if not all(col in self.data for col in ['high', 'low', 'close']):
            raise ValueError("Erreur dans ATR : Les colonnes 'high', 'low' et 'close' doivent être présentes dans les données.")

        high_low = self.data['high'] - self.data['low']
        high_close_prev = (self.data['high'] - self.data['close'].shift(1)).abs()
        low_close_prev = (self.data['low'] - self.data['close'].shift(1)).abs()

        self.true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        self.atr = self.true_range.rolling(window=self.window).mean()

    def update(self, new_data: pd.DataFrame):
        """
        Met à jour l'ATR avec de nouvelles données.

        :param new_data: DataFrame contenant les nouvelles données à ajouter.
        """
        if not all(col in new_data for col in ['high', 'low', 'close']):
            raise ValueError("Erreur dans ATR : Les colonnes 'high', 'low' et 'close' doivent être présentes dans les nouvelles données.")

        self.data = new_data

    def clear_cache(self):
        """
        Vide le cache de l'indicateur.
        Utilisé pour réinitialiser l'état de l'indicateur.
        """
        self.true_range = None
        self.atr = None
        self.data = pd.DataFrame()