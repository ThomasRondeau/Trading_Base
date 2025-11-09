import pandas as pd
from Features.FeatureID import FeatureID
from Indicators.IndicatorInterface import IndicatorInterface

class BollingerBands(IndicatorInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), window : int = 20, num_std_dev : int = 2):
        self.data = data
        self.window = window
        self.num_std_dev = num_std_dev
        self.sma = None
        self.std = None
        self.upper_band = None
        self.lower_band = None
        self.width = None
        self.position = None

    def calculate(self, resolved_dependencies: dict[FeatureID, IndicatorInterface] = None):
        self.sma = self.data['close'].rolling(window=self.window).mean()
        self.std = self.data['close'].rolling(window=self.window).std()
        self.upper_band = self.sma + (self.num_std_dev * self.std)
        self.lower_band = self.sma - (self.num_std_dev * self.std)
        self.width = (self.upper_band - self.lower_band)
        self.position = (self.data['close'] - self.lower_band) / (self.upper_band - self.lower_band)

    def update(self, new_data: pd.DataFrame):
        if 'close' not in new_data:
            raise ValueError("Erreur dans BollingerBands : La colonne 'close' doit être présente dans les nouvelles données.")
        
        self.data = new_data

    def clear_cache(self):
        """
        Vide le cache de l'indicateur.
        Utilisé pour réinitialiser l'état de l'indicateur.
        """
        self.sma = None
        self.std = None
        self.upper_band = None
        self.lower_band = None
        self.width = None
        self.position = None
        self.data = pd.DataFrame()