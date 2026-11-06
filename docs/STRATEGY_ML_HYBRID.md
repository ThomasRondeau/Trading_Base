# Stratégie ML Hybrid : signal technique confirmé par un modèle

`quantlab/strategies/ml_hybrid.py` · nom CLI : `ml_hybrid`

La stratégie combine un déclencheur de retour à la moyenne et un modèle XGBoost qui prédit la direction du prix. Elle ne trade que si les deux sont d'accord.

## 1. Déclencheur technique

Le déclencheur est proche de celui de la [Mean Reversion](STRATEGY_MEAN_REVERSION.md), en un peu plus large :

- **achat candidat** si les trois conditions sont réunies : close ≤ bande de Bollinger basse (20, 2), RSI(14) < 35, ratio de volume (20) > 1,1 ;
- **vente candidate** si les trois conditions sont réunies : close ≥ bande haute, RSI > 65, ratio de volume > 1,1.

## 2. Confirmation par le modèle

- **Modèle** : `XGBClassifier` avec 100 arbres, `max_depth` = 4, `learning_rate` = 0,05, `subsample` et `colsample_bytree` à 0,8, graine fixe. Il est enveloppé dans un `SklearnModel`.
- **Cible** : `ForwardTarget(5, "direction")`, qui vaut 1 si le close dans 5 barres est plus haut.
- **Entrées** : le jeu `DirectionFeatures` (10 colonnes `direction_*`, normalisées par l'ATR pour être comparables dans le temps) :
  - RSI ;
  - histogramme MACD / ATR ;
  - distance à la SMA 20 / ATR, pente de la SMA sur 3 barres / ATR ;
  - position dans les bandes ;
  - volume signé ;
  - ATR / prix ;
  - momentum sur 5, 10 et 20 barres.
- **Règle** :
  - un achat candidat n'est exécuté (10 actions par défaut) que si le modèle prédit la hausse ;
  - une vente candidate n'est exécutée que s'il prédit la baisse ;
  - sinon, le signal est abandonné.
- La prédiction est journalisée à chaque barre (`diagnostics = {"direction_up": …}`), ce qui permet de l'analyser après une session live.

## 3. Entraînement et protection contre le look-ahead

- `strategy.fit(history)` remplit le buffer de features, puis entraîne le modèle sur tout l'historique d'entraînement.
- Les 30 dernières barres sont exclues du jeu d'entraînement (`drop_last=30`, avec un horizon de 5). Aucun label n'utilise donc des prix postérieurs à la fin de l'historique.
- Les labels inconnus des dernières barres valent NaN et sont écartés, alors qu'ils valaient 0 avant la refonte.
- Les EWM (MACD) ne sont plus tronquées à 200 barres, et le buffer live démarre sans trou après l'entraînement.
- En live, il faut 10 jours d'historique IBKR pour l'entraînement (valeur par défaut de la CLI pour cette stratégie).

## 4. Paramètres

| Paramètre | Défaut | Exemple |
|---|---|---|
| `max_depth` | 4 | `--param max_depth=3` |
| `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`, `random_state` | 100, 0,05, 0,8, 0,8, 42 | `--param learning_rate=0.03` |
| `rsi_buy`, `rsi_sell`, `volume_threshold` | 35, 65, 1,1 | `--param rsi_buy=30 --param rsi_sell=70` |
| `horizon`, `training_margin` | 5, 30 | `--param horizon=10 --param training_margin=40` |
| fenêtres des indicateurs et des features composites | RSI 14, Bollinger 20 / 2, volume 20, MACD 12/26/9… | `--param rsi_window=21` |
| taille (`size_mode`, `quantity`, `size_value`) | fixed, 10, aucun | `--param size_mode=risk --param size_value=0.5` |

Tous les paramètres (indicateurs, seuils, taille, sorties) : `quantlab params --strategy ml_hybrid`, ou le formulaire de l'interface.

## 5. Résultats actuels

Sur la référence de non-régression (AAPL en 1 min, 4 000 barres d'entraînement et 1 000 de test, capital de 100 000 $) :

| Trades | Taux de réussite | Rendement | Profit factor | Drawdown max | Frais |
|---|---|---|---|---|---|
| 31 | 38,7 % | −0,125 % | 0,26 | −0,16 % | 17,18 $ |

L'ancien « ~56,5 % de réussite » reposait sur un moteur optimiste (exécution au close du signal) et sur un MACD dont les fenêtres étaient inversées (bug A1, corrigé). Il n'est plus valable.

## 6. Analyse et pistes

- Le modèle est entraîné sur **toutes** les barres, alors qu'il n'est consulté que sur les rares barres où le déclencheur technique s'allume. Entraîner (ou pondérer) le modèle sur les seules situations de retournement serait plus cohérent avec son usage.
- Comparer le score in-sample et le score en validation croisée chronologique (`model.cross_validate`) pour détecter le surapprentissage, puis valider en walk-forward sur plusieurs mois.
- Le paramètre `max_depth` se prête à `optimize` : `--grid max_depth=2,3,4,6`.
