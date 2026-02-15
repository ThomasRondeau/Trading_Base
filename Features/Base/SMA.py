import pandas as pd
from Features import *

class SMA(FeatureInterface):
    """
    Simple Moving Average (SMA) Indicator
    """
    
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), window : int = 14):

        self.window = window
        self.data = data
        self.sma = None

    def calculate(self, resolved_dependencies: dict[FeatureID, FeatureInterface] = None):
        """
        Calculate the Simple Moving Average for the given data.
        
        :param data: List of prices (e.g., closing prices)
        :return: List of SMA values
        """
        if 'close' not in self.data:
            raise ValueError("Erreur dans SMA : La colonne 'close' doit être présente dans les données.")
        
        self.sma = self.data['close'].rolling(window=self.window).mean()

    def update(self, new_data: pd.DataFrame):
        """
        Update the SMA with new data.
        
        :param new_data: DataFrame containing the new data to add.
        """
        if 'close' not in new_data:
            raise ValueError("Erreur dans SMA : La colonne 'close' doit être présente dans les nouvelles données.")
        
        self.data = new_data

    def clear_cache(self):
        """
        Clear the cache of the indicator.
        Used to reset the state of the indicator.
        """
        self.sma = None
        self.data = pd.DataFrame()
