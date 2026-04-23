import traceback
import pandas as pd
from Connections.DataManager import DataManager
from Strategies.PairTradingStrategy import PairTradingStrategy
from Engines.BacktestEngine import BacktestEngine

def prepare_pair_data(symbol_a, symbol_b, days, shift):
    dm = DataManager()
    data_a = dm.collect_data_yfinance(symbol_a, days, shift)
    data_b = dm.collect_data_yfinance(symbol_b, days, shift)
    
    # Fusionner sur l'index (timestamps)
    combined = pd.merge(data_a, data_b, left_index=True, right_index=True, suffixes=(f'_{symbol_a}', f'_{symbol_b}'))
    # Ajouter les colonnes génériques pour la compatibilité si besoin
    combined['close'] = combined[f'close_{symbol_a}'] 
    return combined

def main():
    print("Démarrage du backtest Pair Trading (AAPL vs MSFT)...")
    
    try:
        strategy = PairTradingStrategy()
        engine = BacktestEngine()

        print("Collecte et synchronisation des données...")
        training_data = prepare_pair_data("AAPL", "MSFT", 50, 5)
        backtest_data = prepare_pair_data("AAPL", "MSFT", 5, 0)

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
