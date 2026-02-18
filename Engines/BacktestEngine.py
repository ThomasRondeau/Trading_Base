import pandas as pd
from Connections.TradingApp import TradingApp
from Engines import *
from Strategies.TradingStrategyInterface import TradingStrategyInterface

class BacktestEngine:
    def __init__(self):


        self.trading_app = TradingApp.get_instance()
        self.strategies : list[TradingStrategyInterface] = []
        self.IsRunning : bool = False

        # Portfolio et book
        self.Portfolio = pd.DataFrame(columns=['symbol', 'quantity', 'entry_price', 'current_price', 'pnl'])
        self.OrderBook = pd.DataFrame(columns=['timestamp', 'symbol', 'quantity', 'price', 'side', 'status'])

        # Managers
        self.OrderManager = BacktestOrderManager()
        self.RiskManager = RiskManager()

        #Data parameters
        self.symbol : str = None
        self.days_back : int = None
        self.trainging_data : pd.DataFrame = None
        self.backtest_data : pd.DataFrame = None


    def add_strategy(self, strategy: TradingStrategyInterface) -> None:
        self.strategies.append(strategy)
        print(f"Strategy {strategy.__class__.__name__} added to engine.")

    
    def set_backtest_data(self, symbol: str, days_back: int):
        self.symbol = symbol
        self.days_back = days_back
        self.trainging_data, self.backtest_data = self.trading_app.get_historical_data(symbol, days_back)
        print(f"Backtest data set for {symbol} with {len(self.backtest_data)} bars.")

    def set_training_data(self, training_data: pd.DataFrame):
        self.trainging_data = training_data
        print(f"Training data set with {len(training_data)} bars.")



    def train_strategies(self):
        # Initialisation et training des stratégies si besoin
        for strategy in self.strategies:
            strategy.train(self.trainging_data)

        
    def run(self):
        if self.backtest_data is None or self.strategies is None or len(self.strategies) == 0:
            raise ValueError("Data and strategies must be set before running the backtest.")
        
        print(f"BacktestEngine started with {len(self.strategies)} strategies.")
        self.IsRunning = True

        for strategy in self.strategies:
            strategy.update_status()

        for timestamp, bar in enumerate(self.backtest_data):
            orders = []
            for strategy in self.strategies:
                orders.extend(strategy.on_data(bar))
                
                orders = self.OrderManager.aggregate_and_net(orders)
                self.OrderManager.process_orders(orders, self.Portfolio, self.OrderBook)


    def stop(self):
        self.IsRunning = False
        print(f"BacktestEngine stopped.")

        
    

