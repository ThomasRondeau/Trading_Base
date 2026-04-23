# Stratégie ML Hybrid : Fusion Technique et Machine Learning

La stratégie ML Hybrid combine la robustesse des indicateurs de retour à la moyenne avec la puissance prédictive des modèles de classification.

## 1. Setup Technique (Déclencheur)
La stratégie utilise les déclencheurs de la **Mean Reversion** comme signaux candidats :
- Prix hors des Bandes de Bollinger.
- RSI en zone extrême.
- Ratio de volume élevé.

## 2. Confirmation ML (Filtre)
Chaque signal candidat est soumis à l'arbitrage d'un modèle **XGBoost** (`DirectionModel`) :
- Si le setup est "Achat" mais que le modèle ML prédit une direction "DOWN" sur les 5 prochaines barres, le trade est annulé.
- Le trade n'est exécuté que s'il y a **convergence** totale entre l'analyse technique et la prédiction ML.

## 3. Entraînement et Architecture
- Le modèle ML est entraîné sur le même `FeatureManager` que la stratégie Cascade, garantissant une richesse d'information (Z-scores, momentum, etc.).
- Utilisation d'un `training_data_shift` de 30 pour aligner les prédictions futures sans introduire de biais de look-ahead.

## 4. Performance
- **Résultat Backtest** : ~56.5% de Winrate.
- **Analyse** : L'ajout du ML permet d'augmenter légèrement le winrate par rapport à la Mean Reversion pure en filtrant les retournements qui se transforment en "breakouts" directionnels perdants.
