import pandas as pd
from Features import *

class MACD(FeatureInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), short_window: int = 12, long_window: int = 26, signal_window: int = 9):
        """
        Initialise le calculateur MACD avec les données fournies.

        :param data: DataFrame contenant les colonnes 'close' et 'date'.
        :param short_window: Période pour la moyenne mobile courte.
        :param long_window: Période pour la moyenne mobile longue.
        :param signal_window: Période pour la ligne de signal.
        """
        if short_window >= long_window:
            raise ValueError(f"MACD: short_window ({short_window}) must be < long_window ({long_window}).")
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.macd = None        
        self.signal_line = None
        self.macd_histogram = None

    def calculate(self, resolved_dependencies: dict[FeatureID, FeatureInterface] = None):
        """
        Calcule le MACD et la ligne de signal, et les ajoute aux attributs `self.macd` et `self.signal_line`.
        """
        if 'close' not in self.data:
            raise ValueError("Erreur dans MACD : La colonne 'close' doit être présente dans les données.")

        short_ema = self.data['close'].ewm(span=self.short_window, adjust=False).mean()
        long_ema = self.data['close'].ewm(span=self.long_window, adjust=False).mean()
        
        self.macd = short_ema - long_ema
        self.signal_line = self.macd.ewm(span=self.signal_window, adjust=False).mean()
        self.macd_histogram = self.macd - self.signal_line
    
    def update(self, new_data: pd.DataFrame):
        """
        Met à jour le MACD avec de nouvelles données.

        :param new_data: DataFrame contenant les nouvelles données à ajouter.
        """
        if 'close' not in new_data:
            raise ValueError("Erreur dans MACD : La colonne 'close' doit être présente dans les nouvelles données.")
        
        self.data = new_data

    def clear_cache(self):
        """
        Vide le cache de l'indicateur.
        Utilisé pour réinitialiser l'état de l'indicateur.
        """
        self.macd = None
        self.signal_line = None
        self.macd_histogram = None
        self.data = pd.DataFrame()