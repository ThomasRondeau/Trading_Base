import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Ajout du chemin racine pour l'import des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Connections.DataManager import DataManager
from Strategies.MLHybridStrategy import MLHybridStrategy
from Engines.BacktestEngine import BacktestEngine

def run_optimization():
    print("Optimisation MLHybridStrategy: XGBoost Max Depth...")
    
    dm = DataManager()
    # Collecte de données plus récente
    training_data = dm.collect_data_yfinance("AAPL", 30, 5) 
    backtest_data = dm.collect_data_yfinance("AAPL", 5, 0)
    
    params = [2, 3, 4, 5, 6, 7]
    results = []
    
    for depth in params:
        print(f"Test avec max_depth={depth}...")
        strategy = MLHybridStrategy(max_depth=depth)
        engine = BacktestEngine()
        engine.set_training_data(training_data)
        engine.set_backtest_data(backtest_data)
        engine.add_strategy(strategy)
        
        engine.train_strategies()
        metrics = engine.run()
        results.append(metrics['total_pnl_net_percent'])
        
    # Génération du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(params, results, marker='d', linestyle='-', color='purple')
    plt.title('Optimisation MLHybrid : Impact de la profondeur XGBoost')
    plt.xlabel('XGBoost Max Depth')
    plt.ylabel('Total PnL Net (%)')
    plt.grid(True)
    
    output_path = os.path.join("report_assets", "opt_mlhybrid.png")
    plt.savefig(output_path)
    print(f"Graphique sauvegardé dans {output_path}")

if __name__ == "__main__":
    if not os.path.exists("report_assets"):
        os.makedirs("report_assets")
    run_optimization()
