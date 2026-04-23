# Stratégie ARIMA : Prédiction Statistique à Court Terme

Cette stratégie utilise la modélisation de séries temporelles classique pour exploiter l'autocorrélation des prix.

## 1. Modèle Statistique
- **Algorithme** : ARIMA (AutoRegressive Integrated Moving Average).
- **Configuration** : Ordre (2, 1, 0) privilégié pour la rapidité d'exécution en backtest intraday.
- **Rolling Forecast** : Le modèle est ré-entraîné sur les 50 dernières barres à chaque nouvelle bougie pour capturer la dynamique locale.

## 2. Logique de Signal
La stratégie repose sur l'écart entre la prévision (`forecast`) et le prix actuel :
- **Achat** : Si `forecast > current_price`.
- **Vente** : Si `forecast < current_price`.
- **Seuil** : Initialement filtré par l'ATR, le seuil a été réduit pour permettre une exploitation maximale du bruit de marché directionnel.

## 3. Gestion des Données
- Les prix de clôture sont convertis en type `float` pour garantir la compatibilité avec les algorithmes d'optimisation de `statsmodels`.
- Gestion des erreurs de convergence pour éviter l'arrêt du moteur lors de périodes de trop faible volatilité.

## 4. Performance
- **Résultat Backtest** : ~54% de Winrate.
- **Analyse** : Stratégie très active (haute fréquence de trades) qui compense un faible profit moyen par un nombre élevé d'occurrences gagnantes.
