import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Ajout du chemin racine pour l'import des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Connections.DataManager import DataManager
from Strategies.MeanReversionStrategy import MeanReversionStrategy
from Engines.BacktestEngine import BacktestEngine

def run_optimization():
    print("Optimisation MeanReversionStrategy: Seuil RSI...")
    
    dm = DataManager()
    training_data = dm.collect_data_yfinance("AAPL", 100, 10)
    backtest_data = dm.collect_data_yfinance("AAPL", 10, 0)
    
    params = [20, 25, 30, 35, 40]
    results = []
    
    for rsi in params:
        print(f"Test avec RSI Threshold={rsi}...")
        strategy = MeanReversionStrategy(rsi_threshold=rsi)
        engine = BacktestEngine()
        engine.set_training_data(training_data)
        engine.set_backtest_data(backtest_data)
        engine.add_strategy(strategy)
        
        engine.train_strategies()
        metrics = engine.run()
        results.append(metrics['total_pnl_net_percent'])
        
    # Génération du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(params, results, marker='^', linestyle='-', color='r')
    plt.title('Optimisation MeanReversion : Seuil RSI Oversold')
    plt.xlabel('Seuil RSI')
    plt.ylabel('Total PnL Net (%)')
    plt.grid(True)
    
    output_path = os.path.join("report_assets", "opt_meanreversion.png")
    plt.savefig(output_path)
    print(f"Graphique sauvegardé dans {output_path}")

if __name__ == "__main__":
    if not os.path.exists("report_assets"):
        os.makedirs("report_assets")
    run_optimization()
