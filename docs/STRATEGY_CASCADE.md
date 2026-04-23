# Stratégie Cascade : Architecture Tri-Modèle ML

La stratégie Cascade est l'approche la plus complexe du système, reposant sur une hiérarchie de trois modèles de Machine Learning spécialisés pour filtrer les signaux.

## 1. Architecture des Modèles
La décision est prise en "cascade" à travers trois couches :
- **Modèle d'Opportunité (RandomForest)** : Filtre binaire déterminant si les conditions de marché sont propices à un mouvement significatif.
- **Modèle de Direction (XGBoost)** : Prédit la polarité du mouvement (HAUSSE/BAISSE) à un horizon de 5 barres.
- **Modèle d'Amplitude (RandomForest)** : Estime la force du mouvement prévu en pourcentage.

## 2. Feature Engineering Avancé
Pour maximiser l'edge, les modèles consomment des features stationnarisées :
- **Distances Normalisées** : Écart au prix SMA et aux Bandes de Bollinger divisé par l'ATR.
- **Momentum Multi-Horizons** : Taux de variation du prix sur 5, 10 et 20 barres.
- **Intensité du Volume** : Ratio du volume actuel par rapport à sa moyenne mobile.

## 3. Logique de Décision
Un ordre n'est généré que si les trois conditions suivantes sont réunies :
1. `OpportunityModel` prédit une opportunité réelle.
2. `DirectionModel` donne une direction claire.
3. `AmplitudeModel` prédit un mouvement supérieur au seuil configuré (ex: 0.05%).

## 4. Performance et Validation
- **Régularisation** : Utilisation de `max_depth` réduit et `min_samples_leaf` élevé pour éviter le surapprentissage observé initialement.
- **Résultat Backtest** : ~73% de Winrate sur AAPL (intervalle 2m), validé sur des données hors-échantillon.
