import pandas as pd
import numpy as np
from Features.FeatureInterface import FeatureInterface
from Features.FeatureID import FeatureID

class ADXFeature(FeatureInterface):
    def __init__(self, window: int = 14):
        self.window = window
        self.data = pd.DataFrame()
        self.adx = None

    def dependencies(self):
        return set()

    def calculate(self, resolved_dependencies: dict):
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        tr = np.maximum(high - low, 
                        np.maximum(abs(high - close.shift(1)), 
                                   abs(low - close.shift(1))))
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr_smooth = pd.Series(tr).rolling(window=self.window).sum()
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=self.window).sum()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=self.window).sum()
        
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        self.adx = dx.rolling(window=self.window).mean()

    def update(self, new_data: pd.DataFrame):
        self.data = new_data

    def clear_cache(self):
        self.adx = None
        self.data = pd.DataFrame()
