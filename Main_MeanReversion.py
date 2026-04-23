import traceback
from Connections.DataManager import DataManager
from Strategies.MeanReversionStrategy import MeanReversionStrategy
from Engines.BacktestEngine import BacktestEngine

def main():
    print("Démarrage du backtest Mean Reversion...")
    
    try:
        strategy = MeanReversionStrategy()
        engine = BacktestEngine()
        dm = DataManager()

        training_data = dm.collect_data_yfinance("AAPL", 50, 5)
        backtest_data = dm.collect_data_yfinance("AAPL", 5, 0)

        engine.set_training_data(training_data)
        engine.set_backtest_data(backtest_data)
        engine.add_strategy(strategy)
        
        engine.train_strategies()
        engine.run()
        
    except Exception as e:
        print(f"Erreur : {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
