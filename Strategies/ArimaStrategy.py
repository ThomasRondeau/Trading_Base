import pandas as pd
from Strategies.TradingStrategyInterface import TradingStrategyInterface
from Models.ArimaModel import ArimaModel
from Features.FeatureManager import FeatureManager
from Targets.TargetManager import TargetManager
from Types.Order import Order
from Features.FeatureID import FeatureID
from Features.Base.ATR import ATR

class ArimaStrategy(TradingStrategyInterface):
    """
    Stratégie basée sur les prédictions ARIMA.
    """
    def __init__(self, order=(5,1,1)) -> None:
        super().__init__()
        self.strategy_id = 2
        self.symbol = "AAPL"
        self.IsRunning = False
        self.featureManager = FeatureManager()
        self.targetManager = TargetManager()
        self.model = ArimaModel(order=order)
        self.current_data = pd.DataFrame()

    def train(self, training_data: pd.DataFrame) -> None:
        self.featureManager.register_indicator(FeatureID.ATR, ATR(window=14))
        self.current_data = training_data.copy()
        self.featureManager.update(self.current_data)
        self.model.setFeatureManager(self.featureManager)
        self.model.train()

    def test(self) -> None:
        print("ARIMA Model: No specific training metrics (Rolling Forecast).")

    def on_bar(self, bar) -> list[Order]:
        if not self.IsRunning:
            return []

        self.current_data = pd.concat([self.current_data, bar.to_frame().T]).iloc[-100:]
        self.featureManager.update(self.current_data)
        
        # Prédiction du prochain mouvement
        prediction = self.model.predict()
        
        if prediction == 0:
            return []

        side = 'BUY' if prediction > 0 else 'SELL'
        
        return [Order(strategy_ID=self.strategy_id, symbol=self.symbol, side=side, quantity=5)]

    def update_status(self, is_running: bool) -> None:
        self.IsRunning = is_running

    def stop(self):
        self.IsRunning = False

    def get_status(self) -> bool:
        return self.IsRunning
