import traceback
from Connections.DataManager import DataManager
from Strategies.ArimaStrategy import ArimaStrategy
from Engines.BacktestEngine import BacktestEngine

def main():
    print("Démarrage du backtest ARIMA...")
    
    try:
        arima_strategy = ArimaStrategy()
        BacktestingEngine = BacktestEngine()
        dataManager = DataManager()

        print("Collecte des données...")
        training_data = dataManager.collect_data_yfinance("AAPL", 50, 5)
        backtest_data = dataManager.collect_data_yfinance("AAPL", 5, 0)

        BacktestingEngine.set_training_data(training_data)
        BacktestingEngine.set_backtest_data(backtest_data)
        BacktestingEngine.add_strategy(arima_strategy)
        
        print("Entraînement...")
        BacktestingEngine.train_strategies()
        
        print("Lancement du backtest...")
        BacktestingEngine.run()
        
        print("Backtest terminé.")

    except Exception as e:
        print(f"Erreur : {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
