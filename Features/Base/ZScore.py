from Features.FeatureInterface import FeatureInterface
from Features.FeatureID import FeatureID
import pandas as pd
import numpy as np

class ZScoreFeature(FeatureInterface):
    def __init__(self, window: int = 20):
        self.window = window
        self.data = pd.DataFrame()
        self.zscore = None

    def dependencies(self):
        return set()

    def calculate(self, resolved_dependencies: dict):
        # On suppose que self.data contient une colonne 'ratio'
        ratio = self.data['ratio']
        mean = ratio.rolling(window=self.window).mean()
        std = ratio.rolling(window=self.window).std()
        self.zscore = (ratio - mean) / std

    def update(self, new_data: pd.DataFrame):
        self.data = new_data

    def clear_cache(self):
        self.zscore = None
        self.data = pd.DataFrame()
