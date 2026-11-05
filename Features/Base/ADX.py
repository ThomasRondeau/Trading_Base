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
        
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr_smooth = tr.rolling(window=self.window).sum()
        plus_dm_smooth = plus_dm.rolling(window=self.window).sum()
        minus_dm_smooth = minus_dm.rolling(window=self.window).sum()
        
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        self.adx = dx.rolling(window=self.window).mean()

    def update(self, new_data: pd.DataFrame):
        self.data = new_data

    def clear_cache(self):
        self.adx = None
        self.data = pd.DataFrame()
