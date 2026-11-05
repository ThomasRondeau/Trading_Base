import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Ajout du chemin racine pour l'import des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Connections.DataManager import DataManager
from Strategies.ArimaStrategy import ArimaStrategy
from Engines.BacktestEngine import BacktestEngine

def run_optimization():
    print("Optimisation ArimaStrategy: Paramètre AR Lag (p)...")
    
    dm = DataManager()
    # Collecte de données plus large pour la robustesse
    training_data = dm.collect_data_yfinance("AAPL", 100, 10)
    backtest_data = dm.collect_data_yfinance("AAPL", 10, 0)
    
    params = [1, 2, 3, 4, 5, 6]
    results = []
    
    for p in params:
        print(f"Test avec p={p}...")
        strategy = ArimaStrategy(order=(p, 1, 1))
        engine = BacktestEngine()
        engine.set_training_data(training_data)
        engine.set_backtest_data(backtest_data)
        engine.add_strategy(strategy)
        
        engine.train_strategies()
        metrics = engine.run()
        results.append(metrics['total_pnl_net_percent'])
        
    # Génération du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(params, results, marker='o', linestyle='-', color='b')
    plt.title('Optimisation ArimaStrategy : Impact du Lag AR (p)')
    plt.xlabel('Lag AR (p)')
    plt.ylabel('Total PnL Net (%)')
    plt.grid(True)
    
    output_path = os.path.join("report_assets", "opt_arima.png")
    plt.savefig(output_path)
    print(f"Graphique sauvegardé dans {output_path}")

if __name__ == "__main__":
    if not os.path.exists("report_assets"):
        os.makedirs("report_assets")
    run_optimization()
