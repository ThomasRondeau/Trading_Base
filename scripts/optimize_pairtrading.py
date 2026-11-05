import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ajout du chemin racine pour l'import des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Connections.DataManager import DataManager
from Strategies.PairTradingStrategy import PairTradingStrategy
from Engines.BacktestEngine import BacktestEngine

def prepare_pair_data(symbol_a, symbol_b, days, shift):
    dm = DataManager()
    data_a = dm.collect_data_yfinance(symbol_a, days, shift)
    data_b = dm.collect_data_yfinance(symbol_b, days, shift)
    combined = pd.merge(data_a, data_b, left_index=True, right_index=True, suffixes=(f'_{symbol_a}', f'_{symbol_b}'))
    combined['close'] = combined[f'close_{symbol_a}'] 
    return combined

def run_optimization():
    print("Optimisation PairTradingStrategy: Z-Score Threshold...")
    
    training_data = prepare_pair_data("AAPL", "MSFT", 100, 10)
    backtest_data = prepare_pair_data("AAPL", "MSFT", 10, 0)
    
    params = np.linspace(1.5, 3.0, 6)
    results = []
    
    for threshold in params:
        print(f"Test avec zscore_threshold={threshold:.2f}...")
        strategy = PairTradingStrategy(zscore_threshold=threshold)
        engine = BacktestEngine()
        engine.set_training_data(training_data)
        engine.set_backtest_data(backtest_data)
        engine.add_strategy(strategy)
        
        engine.train_strategies()
        metrics = engine.run()
        results.append(metrics['total_pnl_net_percent'])
        
    # Génération du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(params, results, marker='x', linestyle='-', color='orange')
    plt.title('Optimisation PairTrading : Seuil Z-Score')
    plt.xlabel('Z-Score Threshold')
    plt.ylabel('Total PnL Net (%)')
    plt.grid(True)
    
    output_path = os.path.join("report_assets", "opt_pairtrading.png")
    plt.savefig(output_path)
    print(f"Graphique sauvegardé dans {output_path}")

if __name__ == "__main__":
    if not os.path.exists("report_assets"):
        os.makedirs("report_assets")
    run_optimization()
