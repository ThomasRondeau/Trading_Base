import time
import threading
import traceback

from Connections.TradingApp import TradingApp
from Connections.DataManager import DataManager
from Strategies.CascadeStrategy import CascadeStrategy
from Engines import BacktestEngine

def main():
    print("Démarrage du backtest via YFinance...")
    
    try:
        cascadeStrategy = CascadeStrategy()
        BacktestingEngine = BacktestEngine()
        dataManager = DataManager()

        print("Collecte des données d'entraînement...")
        training_data = dataManager.collect_data_yfinance("AAPL", 50, 5)
        
        print("Collecte des données de backtest...")
        backtest_data = dataManager.collect_data_yfinance("AAPL", 5, 0)

        BacktestingEngine.set_training_data(training_data)
        BacktestingEngine.set_backtest_data(backtest_data)

        BacktestingEngine.add_strategy(cascadeStrategy)
        
        print("Entraînement des stratégies en cours...")
        BacktestingEngine.train_strategies()
        BacktestingEngine.test_strategies()
        
        
        print("Lancement du backtest...")
        BacktestingEngine.run()
        
        print("Backtest terminé avec succès !")
        
        

    except Exception as e:
        print(f"Erreur lors de l'exécution du backtest : {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
