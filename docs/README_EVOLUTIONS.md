# Documentation des Évolutions du Système de Trading (Session Avril 2026)

Ce document récapitule les modifications majeures apportées au système pour atteindre un edge statistique (>50% winrate) et diversifier les approches de trading.

## 1. Amélioration du Moteur de Backtest (`Engines/`)

Le `BacktestEngine` a été transformé d'un simple émetteur de logs en un simulateur d'exécution fonctionnel :
- **Suivi des Positions** : Implémentation d'une gestion active des entrées et sorties.
- **Logique de Sortie** : Clôture automatique des positions après 5 barres (pour s'aligner sur l'horizon de prédiction des modèles ML).
- **Reporting** : Calcul automatique du Winrate, du PnL total et du PnL moyen par trade affiché en fin de run.

## 2. Raffinement des Modèles de Machine Learning (`Models/`)

### Lutte contre le Surapprentissage (Overfitting)
Les modèles présentaient initialement des scores de 1.0 sur le train et ~0.5 en validation. 
- **Régularisation** : Limitation de la profondeur des arbres (`max_depth=4` pour XGBoost, `5` pour RandomForest) et augmentation des échantillons minimums par feuille.
- **Features Continues** : Remplacement des indicateurs binaires (0/1) par des valeurs continues (Z-scores, distances normalisées par l'ATR) pour fournir plus de "nuances" aux modèles.

### Corrections de la Logique de Cible (Targets)
- **Target_Amplitude** : Suppression de la valeur absolue pour permettre au modèle de prédire la direction via le signe de l'amplitude.
- **Target_Opportunity** : Ajustement du seuil de 1.5% à 0.05% pour AAPL, rendant les opportunités détectables sur des bougies de 2 minutes.

## 3. Nouvelles Stratégies Implémentées

### A. Cascade Strategy (Améliorée)
La stratégie phare a été recalibrée avec les nouveaux modèles régularisés.
*   **Résultat Final** : ~73% Winrate sur AAPL (5 jours).

### B. ARIMA Strategy (`Strategies/ArimaStrategy.py`)
*   **Type** : Statistique (Séries Temporelles).
*   **Logique** : Prédiction du prix de la barre suivante via un modèle auto-régressif `ARIMA(2,1,0)`.
*   **Résultat Final** : ~54% Winrate sur AAPL (780 trades). Stratégie très active qui capture les micro-variations.

### C. Trend Following (`Strategies/TrendFollowingStrategy.py`)
*   **Type** : Suivi de tendance.
*   **Logique** : Crossover de moyennes mobiles (5/20 bars).
*   **Résultat Final** : ~42% Winrate. Cette stratégie souffre sur des intervalles courts (2m) sans filtre de tendance plus large.

### D. Mean Reversion Optimized (`Strategies/MeanReversionStrategy.py`)
*   **Type** : Retour à la moyenne.
*   **Logique** : Achat/Vente sur les bornes des Bandes de Bollinger.
*   **Filtres de Qualité** :
    - **Volume Spike** : Confirme l'épuisement du mouvement adverse.
    - **RSI Extreme** : Confirme l'état de sur-achat/sur-vente.
*   **Résultat Final** : ~56% Winrate.

### E. ML Hybrid (`Strategies/MLHybridStrategy.py`)
*   **Type** : Mixte (Technique + ML).
*   **Logique** : Utilise le setup technique de la Mean Reversion mais demande une confirmation de direction au modèle XGBoost avant d'entrer.
*   **Résultat Final** : ~56.5% Winrate.

## 4. Nouveaux Indicateurs (`Features/Base/`)

- **ADX (Average Directional Index)** : Mesure la force d'une tendance.
- **Z-Score** : Mesure l'écart statistique d'un prix ou d'un ratio par rapport à sa moyenne mobile.

## 5. Guide d'Exécution

Pour tester chaque stratégie avec les derniers réglages :
- `python3 Main_Yfinance.py` (Cascade)
- `python3 Main_MeanReversion.py`
- `python3 Main_Trend.py`
- `python3 Main_Hybrid.py`
