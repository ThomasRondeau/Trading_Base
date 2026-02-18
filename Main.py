import time
import threading
import traceback

from Connections.TradingApp import TradingApp
from Connections.DataManager import DataManager
from Strategies.CascadeStrategy import CascadeStrategy
from Engines import BacktestEngine


def main():
    app = TradingApp.get_instance()
    app.connect("127.0.0.1", 7497, clientId=5)

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    timeout = 10  # 10 secondes
    start_time = time.time()
    
    while not app.connected and (time.time() - start_time) < timeout:
        print("En attente de connexion...")
        time.sleep(1)

    if not app.connected:
        print("Not Connected to TWS/IB Gateway")
        return
    print("Connected to TWS/IB Gateway")


    #ICI QUE JE DOIS ECIRER LE CODE DE BACKTESTING
    try:
        cascadeStrategy = CascadeStrategy()
        BacktestingEngine = BacktestEngine()
        dataManager = DataManager()

        training_data = dataManager.collect_training_data("AAPL", 30, 10)
        backtest_data = dataManager.collect_training_data("AAPL", 10)
        
        BacktestingEngine.set_training_data(training_data)
        BacktestingEngine.set_backtest_data("AAPL", 10)
        

        BacktestingEngine.add_strategy(cascadeStrategy)
        BacktestingEngine.train_strategies()
        BacktestingEngine.run()

    except Exception as e:
        print(f"Erreur lors de l'exécution de la stratégie: {e}")
        print(traceback.format_exc())
    


    if app.connected:
        app.disconnect()
        print("Déconnexion")


if __name__ == "__main__":
    main()