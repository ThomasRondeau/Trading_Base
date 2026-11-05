# Stratégie Trend Following : croisement de moyennes mobiles

`quantlab/strategies/trend_following.py` · nom CLI : `trend`

Une approche classique de suivi de tendance : on prend position dans le sens du croisement de deux moyennes mobiles simples.

## 1. Indicateurs

| Feature | Rôle |
|---|---|
| `SMA(sma_fast)`, soit `sma_5` par défaut | moyenne rapide, réactive aux variations récentes |
| `SMA(sma_slow)`, soit `sma_20` par défaut | moyenne lente, tendance de fond à court terme |

Les deux moyennes sont calculées à chaque barre sur le buffer live, ou précalculées en backtest (mêmes valeurs). Le warmup déclaré vaut `sma_slow − 1` barres. La stratégie ne décide rien avant que cet historique soit disponible.

## 2. Règles de décision

La stratégie compare les deux dernières lignes de features (`self.feed.latest(2)`) :

- **achat de 10 actions** quand la SMA rapide passe **au-dessus** de la SMA lente (elle était inférieure ou égale à la barre précédente) ;
- **vente de 10 actions** quand elle passe **en dessous**.

La stratégie n'émet que des entrées. Comme pour toutes les stratégies, la sortie est gérée par l'`OrderRouter` :

- clôture au marché après `holding_bars` = 5 barres ;
- ou, avant, par le stop loss à −2 % ou le take profit à +5 % placés chez le broker dès l'exécution de l'entrée.

Voir [FONCTIONNEMENT_TECHNIQUE.md §7](FONCTIONNEMENT_TECHNIQUE.md).

## 3. Paramètres

| Paramètre | Défaut | Exemple |
|---|---|---|
| `sma_fast` | 5 | `--param sma_fast=10` |
| `sma_slow` | 20 | `--param sma_slow=50` |

```bash
py -3.10 -m quantlab backtest --strategy trend --symbol AAPL --source yfinance --days 55 --test-days 5
py -3.10 -m quantlab optimize --strategy trend --symbol AAPL --grid sma_fast=5,10,20 --grid sma_slow=30,50,100 \
    --train-size 3000 --validation-size 1000 --test-size 1000 --embargo 30
```

## 4. Résultats actuels

Sur la référence de non-régression (`tests/fixtures/aapl_1m.parquet` : 4 000 barres d'1 min en entraînement, 1 000 en test, capital de 100 000 $) :

| Trades | Taux de réussite | Rendement | Profit factor | Drawdown max | Frais |
|---|---|---|---|---|---|
| 72 | 19,4 % | −0,096 % | 0,36 | −0,10 % | 39,86 $ |

Cette référence sert à détecter les régressions, pas à mesurer la performance : elle ne couvre qu'environ 2,5 jours. Le chiffre de « ~42 % de réussite » des anciennes versions venait d'un moteur qui exécutait au close de la barre du signal et sommait des pourcentages par trade. Il n'est plus valable.

## 5. Analyse et pistes

- Sur des barres de 1 à 2 minutes, le croisement de moyennes courtes réagit surtout au bruit : beaucoup de faux signaux, et des frais (1 bp par côté) et un slippage (1 bp) qui pèsent lourd.
- Les filtres testés auparavant (RSI pour éviter les fins de mouvement, ADX > 15 pour ne garder que les tendances fortes) n'ont pas été conservés. L'ADX existe toujours comme feature (`ADX(14)`) et peut être réintroduit.
- Pistes : horizons plus longs (barres de 15 min ou 1 h, en ajustant `holding_bars`), filtre de régime de tendance, sortie au croisement inverse plutôt qu'après un nombre fixe de barres. Évaluer chaque variante avec `optimize` (walk-forward) plutôt que sur un seul découpage.
