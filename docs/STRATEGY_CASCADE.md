# Stratégie Cascade : trois modèles en chaîne

`quantlab/strategies/cascade.py` · nom CLI : `cascade`

C'est la stratégie la plus élaborée. Trois modèles spécialisés répondent successivement à trois questions, et un trade n'est pris que si les trois réponses concordent.

## 1. Les trois modèles

Les trois modèles regardent le même horizon de **5 barres**.

| Étape | Question | Modèle | Cible (`ForwardTarget`) | Score |
|---|---|---|---|---|
| 1. Opportunité | un mouvement significatif va-t-il se produire ? | `RandomForestClassifier` : 100 arbres, profondeur 5, ≥ 15 échantillons par feuille, classes équilibrées | `move` : \|rendement à 5 barres\| > 0,05 % | balanced accuracy |
| 2. Direction | hausse ou baisse ? | `XGBClassifier` : 100 arbres, profondeur 4, learning rate 0,05, subsample 0,8 | `direction` | accuracy |
| 3. Amplitude | de combien, en % ? | `RandomForestRegressor` : 100 arbres, profondeur 5, ≥ 10 échantillons par feuille | `return_pct` (signé) | R² |

La faible profondeur des arbres et le nombre minimal d'échantillons par feuille sont des choix de régularisation. Les premières versions sur-apprenaient : score de 1,0 en entraînement, environ 0,5 en validation.

## 2. Features

Chaque modèle a son propre jeu de features composites, construit à partir d'indicateurs **partagés** et calculés une seule fois : ATR(14), SMA(5), SMA(20), Bollinger(20, 2), RSI(14), MACD(12, 26, 9) et ratio de volume (14).

- **`opportunity_*`** (10 colonnes) :
  - largeur des bandes ;
  - régime de volatilité (ATR rapporté à sa moyenne sur 20 barres) et variation de l'ATR ;
  - ratio et tendance du volume ;
  - alignement du momentum (MACD, RSI, SMA rapide) ;
  - distances aux bandes et à la SMA ;
  - accélération du prix.
- **`direction_*`** (10 colonnes) : RSI, MACD et distances normalisées par l'ATR, pente de la SMA, position dans les bandes, volume signé, momentum sur 5, 10 et 20 barres.
- **`amplitude_*`** (7 colonnes) : volatilité relative, distance à la SMA, largeur des bandes, écart du RSI à 50, \|histogramme MACD\|, \|momentum sur 5 barres\|, ratio de volume.

Ces colonnes sont pour la plupart sans unité (rapportées au prix ou à l'ATR), donc comparables d'une période à l'autre.

## 3. Règle de décision

1. Le modèle d'opportunité doit prédire 1 ; sinon, rien.
2. On lit la direction (hausse ou baisse) et l'amplitude prédite, en %.
3. Il faut \|amplitude\| ≥ `min_amplitude_threshold`, soit 0,05 % par défaut, dans la même unité que les cibles.
4. Le signe de l'amplitude doit être cohérent avec la direction.
5. Si tout concorde, la stratégie achète (hausse) ou vend (baisse) avec une conviction min(|amplitude| / `conviction_cap_pct`, 1), le plafond valant 5 % par défaut. Le router en tire la taille : en mode `fixed`, `max(1, int(conviction × quantity))` actions, avec `quantity` = 10 par défaut.

Les trois prédictions sont journalisées à chaque barre (`opportunity`, `direction_up`, `amplitude_pct`). La sortie est gérée par l'`OrderRouter` : après 5 barres, ou au stop loss à −2 % ou au take profit à +5 %.

## 4. Entraînement et validation

- Entraînement sur l'historique en excluant les 30 dernières barres (`drop_last=30`, avec un horizon de 5), pour éviter tout look-ahead.
- `strategy.evaluate()` affiche, pour chaque modèle, le score in-sample et le score en validation croisée chronologique, ce qui sert à diagnostiquer le surapprentissage.
- **Latence** : c'est la stratégie la plus lourde (features, puis trois prédictions). Les prédictions sur une seule barre tournent sur un seul thread. Le test de latence impose moins de 100 ms en médiane par barre, et on est autour de 50 ms.
- En live, il faut 10 jours d'historique IBKR pour l'entraînement (valeur par défaut de la CLI).

## 5. Paramètres

| Paramètre | Défaut | Exemple |
|---|---|---|
| `min_amplitude_threshold` | 0,05 (%) | `--param min_amplitude_threshold=0.1` |
| `conviction_cap_pct` | 5 (%) | `--param conviction_cap_pct=2` |
| hyperparamètres (`amplitude_*`, `direction_*`, `opportunity_*`), `random_state` | voir §1 | `--param direction_max_depth=3` |
| cible (`horizon`, `opportunity_threshold_pct`, `training_margin`, `cv_folds`) | 5, 0,05, 30, 5 | `--param horizon=10 --param training_margin=40` |
| fenêtres des indicateurs et des features composites | voir §2 | `--param momentum_windows=3,7,30` |
| taille (`size_mode`, `quantity`, `size_value`) | fixed, 10, aucun | `--param size_mode=risk --param size_value=0.5` |

Tous les paramètres (indicateurs, seuils, taille, sorties) : `quantlab params --strategy cascade`, ou le formulaire de l'interface.

Changer `horizon` sans `holding_bars` donne un avertissement : les positions ne seraient pas gardées pendant la durée que les modèles prédisent. Le preset `presets/cascade-horizon-10.json` change les deux.

## 6. Résultats actuels

Sur la référence de non-régression (AAPL en 1 min, 4 000 barres d'entraînement et 1 000 de test, capital de 100 000 $) :

| Trades | Taux de réussite | Rendement | Profit factor | Drawdown max | Frais |
|---|---|---|---|---|---|
| 25 | 52,0 % | +0,0014 % | 1,22 | −0,005 % | 1,39 $ |

C'est la seule stratégie mono-symbole avec un profit factor supérieur à 1 sur cette référence. Mais l'échantillon (25 trades sur environ 2,5 jours) est bien trop petit pour conclure. L'ancien « ~73 % de réussite » venait d'un moteur qui exécutait au close du signal et d'un MACD dont les fenêtres étaient inversées. Il n'est plus valable.

## 7. Analyse et pistes

- **Taille des positions** : avec la formule actuelle, il faut une amplitude prédite d'au moins 1 % pour dépasser 1 action. En intraday, la stratégie trade donc presque toujours **1 action**, ce qui explique les montants minuscules. Une taille calibrée sur la volatilité ou sur un risque cible serait plus utile.
- Le seuil d'opportunité (0,05 %) est très bas pour des barres de 1 minute et ne couvre pas les coûts aller-retour (environ 4 bp avec frais et slippage). Le relever, ou l'exprimer en multiple des coûts.
- Prochaine étape : valider en walk-forward sur plusieurs mois de données Massive, puis en session shadow ou paper, et mesurer l'écart avec le backtest grâce à `quantlab replay`.
