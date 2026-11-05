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

        trades = []  # List to store completed trades
        active_positions = []  # List of {'entry_price', 'side', 'quantity', 'bars_held', 'symbol'}

        basis_point = 0.0001
        
        # Risk settings from RiskManager
        stop_loss_pct = self.RiskManager.stop_loss_pct
        take_profit_pct = getattr(self.RiskManager, 'take_profit_pct', 0.10) # Default 10% if not set

        total_bars = len(self.backtest_data)
        count = 0
        for timestamp, bar in self.backtest_data.iterrows():
            count += 1
            if count % 1000 == 0:
                print(f"Progression : {count}/{total_bars} barres traitées...")
            
            # 1. Update active positions
            remaining_positions = []
            for pos in active_positions:
                pos['bars_held'] += 1
                
                symbol = pos.get('symbol', 'unknown')
                price_col = f"close_{symbol}" if f"close_{symbol}" in bar else 'close'
                current_price = bar[price_col]
                
                # Check for exit conditions: Time-based (5 bars), Stop Loss, or Take Profit
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] if pos['side'] == 'BUY' else (pos['entry_price'] - current_price) / pos['entry_price']
                
                should_close = False
                exit_reason = ""
                
                if pos['bars_held'] >= 5:
                    should_close = True
                    exit_reason = "Time"
                elif pnl_pct <= -stop_loss_pct:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif pnl_pct >= take_profit_pct:
                    should_close = True
                    exit_reason = "Take Profit"
                
                if should_close:
                    exit_price = current_price
                    pnl_total = (exit_price - pos['entry_price']) * pos['quantity'] if pos['side'] == 'BUY' else (pos['entry_price'] - exit_price) * pos['quantity']
                    
                    # Transaction fees: |position| * basis-point * (Sstart + Send)
                    fees = pos['quantity'] * basis_point * (pos['entry_price'] + exit_price)
                    pnl_net = pnl_total - fees
                    
                    pnl_percent = (pnl_total / (pos['quantity'] * pos['entry_price'])) * 100
                    pnl_net_percent = (pnl_net / (pos['quantity'] * pos['entry_price'])) * 100

                    trades.append({
                        'timestamp': timestamp,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'side': pos['side'],
                        'symbol': symbol,
                        'quantity': pos['quantity'],
                        'pnl': pnl_total,
                        'pnl_net': pnl_net,
                        'pnl_percent': pnl_percent,
                        'pnl_net_percent': pnl_net_percent,
                        'fees': fees,
                        'reason': exit_reason
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
            pnl_total = pnl * pos['quantity']
            fees = pos['quantity'] * basis_point * (pos['entry_price'] + last_price)
            pnl_net = pnl_total - fees
            
            pnl_percent = (pnl / pos['entry_price']) * 100
            pnl_net_percent = (pnl_net / (pos['quantity'] * pos['entry_price'])) * 100

            trades.append({
                'timestamp': self.backtest_data.index[-1],
                'entry_price': pos['entry_price'],
                'exit_price': last_price,
                'side': pos['side'],
                'symbol': symbol,
                'quantity': pos['quantity'],
                'pnl': pnl_total,
                'pnl_net': pnl_net,
                'pnl_percent': pnl_percent,
                'pnl_net_percent': pnl_net_percent,
                'fees': fees
            })

        # 3. Calculate and print performance summary
        return self.print_summary(trades)

    def print_summary(self, trades):
        if not trades:
            print("\n--- Backtest Summary ---")
            print("No trades executed.")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl_net_percent': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }

        df_trades = pd.DataFrame(trades)
        
        total_trades = len(trades)
        winning_trades = df_trades[df_trades['pnl_net'] > 0]
        win_rate = (len(winning_trades) / total_trades) * 100
        
        total_pnl_gross = df_trades['pnl_percent'].sum()
        total_pnl_net = df_trades['pnl_net_percent'].sum()
        avg_pnl_net = df_trades['pnl_net_percent'].mean()
        total_fees = df_trades['fees'].sum()

        # Calculation of Sharpe Ratio (Simplified Daily)
        # We group by date to get daily returns
        df_trades['date'] = pd.to_datetime(df_trades['timestamp']).dt.date
        daily_returns = df_trades.groupby('date')['pnl_net_percent'].sum()
        
        sharpe_ratio = 0
        if len(daily_returns) > 1:
            std = daily_returns.std()
            if std != 0:
                sharpe_ratio = (daily_returns.mean() / std) * (252**0.5)

        # Max Drawdown calculation
        cumulative_returns = (1 + daily_returns / 100).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min() * 100

        print("\n" + "="*40)
        print("   BACKTEST PERFORMANCE (NET OF FEES)")
        print("="*40)
        print(f"Total Trades       : {total_trades}")
        print(f"Win Rate           : {win_rate:.2f}%")
        print(f"Total Fees         : {total_fees:.2f}")
        print(f"Total PnL Gross (%) : {total_pnl_gross:.2f}%")
        print(f"Total PnL Net (%)   : {total_pnl_net:.2f}%")
        print(f"Average PnL Net (%) : {avg_pnl_net:.2f}%")
        print(f"Sharpe Ratio (Ann) : {sharpe_ratio:.2f}")
        print(f"Max Drawdown (%)   : {max_drawdown:.2f}%")
        print("="*40 + "\n")

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_fees': total_fees,
            'total_pnl_gross_percent': total_pnl_gross,
            'total_pnl_net_percent': total_pnl_net,
            'avg_pnl_net_percent': avg_pnl_net,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }


    def stop(self):
        self.IsRunning = False
        print(f"BacktestEngine stopped.")
