import pandas as pd
from Connections.TradingApp import TradingApp
from Engines.Managers import OrderManager, RiskManager
from Strategies.TradingStrategyInterface import TradingStrategyInterface

class BacktestEngine:
    def __init__(self):


        self.trading_app = TradingApp()
        self.strategies : list[TradingStrategyInterface] = []
        self.IsRunning : bool = False

        # Portfolio et book
        self.Portfolio = pd.DataFrame(columns=['symbol', 'quantity', 'entry_price', 'current_price', 'pnl'])
        self.OrderBook = pd.DataFrame(columns=['timestamp', 'symbol', 'quantity', 'price', 'side', 'status'])

        # Managers
        self.OrderManager = OrderManager()
        self.RiskManager = RiskManager()

        #Data parameters
        self.symbol : str = None
        self.days_back : int = None
        self.trainging_data : pd.DataFrame = None
        self.backtest_data : pd.DataFrame = None


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
                
                # Attribution des trades à la stratégie
                for order in orders:
                    execution = self.broker.execute(order, bar)
                    self.strategy_portfolios[strategy.name].update(execution)
                    self.global_portfolio.update(execution)


    def stop(self):
        self.IsRunning = False
        print(f"BacktestEngine stopped.")

        
    def add_strategy(self, strategy: TradingStrategyInterface) -> None:
        self.strategies.append(strategy)
        print(f"Strategy {strategy.__class__.__name__} added to engine.")

