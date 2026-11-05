import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Ajout du chemin racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Connections.DataManager import DataManager

def analyze_arima_diagnostics():
    print("Analyse diagnostique ARIMA pour AAPL...")
    dm = DataManager()
    # On prend une fenêtre de données récente
    df = dm.collect_data_yfinance("AAPL", 5, 0)
    prices = df['close']
    
    # 1. Test de Stationnarité (Augmented Dickey-Fuller)
    # Sur les prix
    result_raw = adfuller(prices)
    # Sur les rendements log (différence première)
    log_returns = np.log(prices).diff().dropna()
    result_diff = adfuller(log_returns)
    
    print(f"ADF Statistic (Raw): {result_raw[0]:.4f}, p-value: {result_raw[1]:.4f}")
    print(f"ADF Statistic (Diff): {result_diff[0]:.4f}, p-value: {result_diff[1]:.4f}")
    
    # 2. ACF / PACF Plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(log_returns, lags=20, ax=axes[0])
    plot_pacf(log_returns, lags=20, ax=axes[1])
    axes[0].set_title('Autocorrelation (ACF) - Rendements')
    axes[1].set_title('Partial Autocorrelation (PACF) - Rendements')
    
    output_acf_pacf = os.path.join("report_assets", "arima_diagnostics.png")
    plt.savefig(output_acf_pacf)
    print(f"Diagnostics sauvegardés dans {output_acf_pacf}")

    # 3. Grid Search AIC/BIC
    p_range = range(0, 4)
    q_range = range(0, 4)
    d = 1
    
    results = []
    for p in p_range:
        for q in q_range:
            try:
                model = ARIMA(prices, order=(p, d, q))
                model_fit = model.fit()
                results.append({'p': p, 'q': q, 'AIC': model_fit.aic, 'BIC': model_fit.bic})
            except:
                continue
    
    df_results = pd.DataFrame(results)
    best_aic = df_results.loc[df_results['AIC'].idxmin()]
    best_bic = df_results.loc[df_results['BIC'].idxmin()]
    
    print(f"Best Order (AIC): p={int(best_aic['p'])}, d=1, q={int(best_aic['q'])}")
    print(f"Best Order (BIC): p={int(best_bic['p'])}, d=1, q={int(best_bic['q'])}")
    
    # Graphique AIC/BIC
    plt.figure(figsize=(10, 6))
    for q in q_range:
        subset = df_results[df_results['q'] == q]
        plt.plot(subset['p'], subset['AIC'], label=f'q={q}', marker='o')
    
    plt.title('Critère AIC par paramètre (p, q)')
    plt.xlabel('Paramètre p (AR)')
    plt.ylabel('AIC')
    plt.legend()
    plt.grid(True)
    output_aic = os.path.join("report_assets", "arima_aic_grid.png")
    plt.savefig(output_aic)
    print(f"AIC Grid sauvegardé dans {output_aic}")

if __name__ == "__main__":
    if not os.path.exists("report_assets"):
        os.makedirs("report_assets")
    analyze_arima_diagnostics()
