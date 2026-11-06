# Stratégie ARIMA : prévision statistique à très court terme

`quantlab/strategies/arima.py` (stratégie) · `quantlab/models/arima.py` (`ArimaForecaster`) · nom CLI : `arima`

La stratégie exploite l'autocorrélation à court terme des prix avec un modèle de séries temporelles classique, ré-estimé à chaque barre.

## 1. Modèle

- **ARIMA(p, d, q)**, (2, 1, 0) par défaut, via `statsmodels`. Le paramètre `order` est réellement utilisé : avant la refonte, il était ignoré (bug A3).
- **Prévision glissante** : à chaque barre, le modèle est ré-estimé sur les `window` derniers closes (50 par défaut) et prévoit la variation du prochain close.
- Les barres brutes viennent de `self.feed.bars(window)`. La stratégie déclare donc `min_history = window` pour que le buffer live garde assez d'historique.
- En cas d'échec de convergence, la prévision vaut 0 : pas de trade, sans interrompre le moteur.

## 2. Règle de décision

La stratégie ne trade que si le mouvement prévu **couvre les coûts** :

- si \|variation prévue\| / prix < `min_edge_bps` (2 bp par défaut, soit les frais aller-retour de 2 × 1 bp), elle ne fait rien ;
- sinon, elle achète si la prévision est haussière, et vend si elle est baissière (5 actions par défaut : `quantity`, ou un autre mode de taille).

Avant la refonte, la stratégie tradait à **chaque** barre : le seuil n'était jamais appliqué. La prévision est maintenant journalisée à chaque barre (`diagnostics = {"forecast_change": …}`). La sortie est gérée par l'`OrderRouter` : après 5 barres, ou au stop loss à −2 % ou au take profit à +5 %.

## 3. Paramètres

| Paramètre | Défaut | Exemple |
|---|---|---|
| `order` | (2, 1, 0) | `--param order=1,1,1` |
| `min_edge_bps` | 2,0 | `--param min_edge_bps=4` |
| `window` | 50 | `--param window=100` |
| taille (`size_mode`, `quantity`, `size_value`) | fixed, 5, aucun | `--param size_mode=risk --param size_value=0.5` |

Tous les paramètres (indicateurs, seuils, taille, sorties) : `quantlab params --strategy arima`, ou le formulaire de l'interface.

## 4. Résultats actuels

Sur la référence de non-régression (AAPL en 1 min, 200 barres de test seulement, car la ré-estimation à chaque barre est lente) :

| Trades | Taux de réussite | Rendement | Profit factor | Drawdown max | Frais |
|---|---|---|---|---|---|
| 18 | 44,4 % | ≈ 0 % | 1,00 | −0,007 % | 4,97 $ |

La stratégie est juste à l'équilibre une fois déduits les frais (1 bp par côté) et le slippage simulé (1 bp). L'ancien « ~54 % sur 780 trades » venait de la version qui tradait à chaque barre, avec exécution au close du signal. Il n'est plus valable.

## 5. Coût de calcul

Ré-estimer un ARIMA à chaque barre est coûteux. C'est sans problème pour des barres d'une minute en live, mais ça rend les backtests longs : la référence de non-régression ne teste que 200 barres pour cette raison. Pour des recherches à grande échelle, envisager un filtre de Kalman ou une ré-estimation moins fréquente (tous les N barres), puis vérifier l'effet sur les signaux.

## 6. Pistes

- Exiger un avantage qui couvre aussi le slippage, par exemple `min_edge_bps=4`.
- Choisir l'ordre (p, d, q) et la fenêtre en walk-forward plutôt que de les fixer.
- Combiner la prévision avec un filtre de régime : l'autocorrélation à court terme varie selon la volatilité et l'heure de la journée.
