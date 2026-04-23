import pandas as pd
from Strategies.TradingStrategyInterface import TradingStrategyInterface
from Features.FeatureManager import FeatureManager
from Types.Order import Order
from Features.FeatureID import FeatureID
from Features.Base.SMA import SMA
from Features.Base.ATR import ATR

class TrendFollowingStrategy(TradingStrategyInterface):
    """
    Stratégie de suivi de tendance simplifiée (SMA 5/20 Crossover).
    """
    def __init__(self) -> None:
        super().__init__()
        self.strategy_id = 4
        self.symbol = "AAPL"
        self.IsRunning = False
        self.featureManager = FeatureManager()
        self.current_data = pd.DataFrame()

    def train(self, training_data: pd.DataFrame) -> None:
        self.featureManager.register_indicator(FeatureID.SMA_5, SMA(window=5))
        self.featureManager.register_indicator(FeatureID.SMA_20, SMA(window=20))
        self.featureManager.register_indicator(FeatureID.ATR, ATR(window=14))
        self.current_data = training_data.copy()
        self.featureManager.update(self.current_data)

    def test(self) -> None:
        print("TrendFollowing: Simple SMA 5/20.")

    def on_bar(self, bar) -> list[Order]:
        if not self.IsRunning:
            return []

        self.current_data = pd.concat([self.current_data, bar.to_frame().T])
        self.featureManager.update(self.current_data)
        
        sma5 = self.featureManager.get_feature(FeatureID.SMA_5).sma
        sma20 = self.featureManager.get_feature(FeatureID.SMA_20).sma
        
        if len(sma5) < 5:
            return []
            
        # Crossover
        was_below = sma5.iloc[-2] <= sma20.iloc[-2]
        is_above = sma5.iloc[-1] > sma20.iloc[-1]
        
        was_above = sma5.iloc[-2] >= sma20.iloc[-2]
        is_below = sma5.iloc[-1] < sma20.iloc[-1]
        
        if was_below and is_above:
            return [Order(strategy_ID=self.strategy_id, symbol=self.symbol, side='BUY', quantity=10)]
        elif was_above and is_below:
            return [Order(strategy_ID=self.strategy_id, symbol=self.symbol, side='SELL', quantity=10)]
            
        return []

    def stop(self):
        self.IsRunning = False

    def get_status(self) -> bool:
        return self.IsRunning

    def update_status(self, is_running: bool) -> None:
        self.IsRunning = is_running
