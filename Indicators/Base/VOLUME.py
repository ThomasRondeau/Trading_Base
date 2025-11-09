import pandas as pd
from Features.FeatureID import FeatureID
from Indicators.IndicatorInterface import IndicatorInterface

class VOLUME(IndicatorInterface):
    """
    Simple Moving Average (SMA) Indicator
    """
    
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), window : int = 14):

        self.window = window
        self.data = data
        self.Volume = None
        self.Volume_ma = None
        self.Volume_ratio = None

    def calculate(self, resolved_dependencies: dict[FeatureID, IndicatorInterface] = None):
        """
        Calculate the Simple Moving Average for the given data.
        
        :param data: List of prices (e.g., closing prices)
        :return: List of SMA values
        """
        if 'close' not in self.data:
            raise ValueError("Erreur dans SMA : La colonne 'close' doit être présente dans les données.")
        
        self.Volume = self.data['volume']
        self.Volume_ma =  self.data['volume'].rolling(window= self.window).mean()
        self.Volume_ratio = self.data['volume'] / self.Volume_ma

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
        self.Volume = None
        self.Volume_ma = None
        self.Volume_ratio = None
        self.data = pd.DataFrame()
