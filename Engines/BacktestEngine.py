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

        # Data 
        self.trainging_data : pd.DataFrame = None
        self.backtest_data : pd.DataFrame = None


    def add_strategy(self, strategy: TradingStrategyInterface) -> None:
        self.strategies.append(strategy)
        print(f"Strategy {strategy.__class__.__name__} added to engine.")


    def set_backtest_data(self, backtest_data: pd.DataFrame):
        self.backtest_data = backtest_data
        print(f"Backtest data set with {len(backtest_data)} bars.")

    def set_training_data(self, training_data: pd.DataFrame):
        self.trainging_data = training_data
        print(f"Training data set with {len(training_data)} bars.")



    def train_strategies(self):
        # Initialisation et training des stratégies si besoin
        for strategy in self.strategies:
            strategy.train(self.trainging_data)

    def test_strategies(self):
        for strategy in self.strategies:
            strategy.test()

        
    def run(self):
        if self.backtest_data is None or self.strategies is None or len(self.strategies) == 0:
            raise ValueError("Data and strategies must be set before running the backtest.")
        
        print(f"BacktestEngine started with {', '.join(strategy.__class__.__name__ for strategy in self.strategies)} strategies.")
        self.IsRunning = True

        for strategy in self.strategies:
            strategy.update_status(True)

        trades = []  # List to store completed trades: {'entry_price', 'exit_price', 'side', 'pnl'}
        active_positions = []  # List of {'entry_price', 'side', 'quantity', 'bars_held'}

        for timestamp, bar in self.backtest_data.iterrows():
            # 1. Update active positions (close after 5 bars)
            remaining_positions = []
            for pos in active_positions:
                pos['bars_held'] += 1
                if pos['bars_held'] >= 5:
                    # Close position
                    symbol = pos.get('symbol', 'unknown')
                    # Use symbol specific price if available, else default to 'close'
                    price_col = f"close_{symbol}" if f"close_{symbol}" in bar else 'close'
                    exit_price = bar[price_col]
                    
                    pnl = (exit_price - pos['entry_price']) if pos['side'] == 'BUY' else (pos['entry_price'] - exit_price)
                    pnl_percent = (pnl / pos['entry_price']) * 100
                    trades.append({
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'side': pos['side'],
                        'symbol': symbol,
                        'pnl': pnl,
                        'pnl_percent': pnl_percent
                    })
                else:
                    remaining_positions.append(pos)
            active_positions = remaining_positions

            # 2. Get new orders from strategies
            orders = []
            for strategy in self.strategies:
                orders.extend(strategy.on_bar(bar))
                
            if orders:
                for order in orders:
                    # Execute order immediately at current bar close
                    price_col = f"close_{order.symbol}" if f"close_{order.symbol}" in bar else 'close'
                    active_positions.append({
                        'symbol': order.symbol,
                        'entry_price': bar[price_col],
                        'side': order.side,
                        'quantity': order.quantity,
                        'bars_held': 0
                    })

        # Close any remaining positions at the last price
        last_bar = self.backtest_data.iloc[-1]
        for pos in active_positions:
            symbol = pos.get('symbol', 'unknown')
            price_col = f"close_{symbol}" if f"close_{symbol}" in last_bar else 'close'
            last_price = last_bar[price_col]
            
            pnl = (last_price - pos['entry_price']) if pos['side'] == 'BUY' else (pos['entry_price'] - last_price)
            pnl_percent = (pnl / pos['entry_price']) * 100
            trades.append({
                'entry_price': pos['entry_price'],
                'exit_price': last_price,
                'side': pos['side'],
                'symbol': symbol,
                'pnl': pnl,
                'pnl_percent': pnl_percent
            })

        # 3. Calculate and print performance summary
        self.print_summary(trades)

    def print_summary(self, trades):
        if not trades:
            print("\n--- Backtest Summary ---")
            print("No trades executed.")
            return

        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = (len(winning_trades) / total_trades) * 100
        total_pnl_percent = sum(t['pnl_percent'] for t in trades)
        avg_pnl_percent = total_pnl_percent / total_trades
        
        print("\n" + "="*30)
        print("   BACKTEST PERFORMANCE")
        print("="*30)
        print(f"Total Trades    : {total_trades}")
        print(f"Win Rate        : {win_rate:.2f}%")
        print(f"Total PnL (%)   : {total_pnl_percent:.2f}%")
        print(f"Average PnL (%) : {avg_pnl_percent:.2f}%")
        print("="*30 + "\n")


    def stop(self):
        self.IsRunning = False
        print(f"BacktestEngine stopped.")
