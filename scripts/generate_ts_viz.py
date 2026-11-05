import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def generate_timeseries_split_viz():
    print("Génération de la visualisation TimeSeriesSplit...")
    
    # Simulation de données
    n_samples = 100
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    plt.figure(figsize=(10, 6))
    
    for i, (train_index, test_index) in enumerate(tscv.split(np.arange(n_samples))):
        # Indices d'entraînement
        plt.scatter(train_index, [i + 1] * len(train_index), 
                    c='blue', marker='s', s=50, label='Train' if i == 0 else "")
        # Indices de test
        plt.scatter(test_index, [i + 1] * len(test_index), 
                    c='red', marker='s', s=50, label='Test' if i == 0 else "")
        
    plt.yticks(range(1, n_splits + 1), [f'Split {i+1}' for i in range(n_splits)])
    plt.xlabel('Index des données (Temps)')
    plt.title('Méthodologie de Validation Croisée Temporelle (TimeSeriesSplit)')
    plt.legend(loc='best')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    if not os.path.exists("report_assets"):
        os.makedirs("report_assets")
        
    output_path = os.path.join("report_assets", "timeseries_split.png")
    plt.savefig(output_path)
    print(f"Visualisation sauvegardée dans {output_path}")

if __name__ == "__main__":
    generate_timeseries_split_viz()
