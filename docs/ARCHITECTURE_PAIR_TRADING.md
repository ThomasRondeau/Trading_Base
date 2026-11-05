# Pair Trading et support multi-symboles

`quantlab/strategies/pair_trading.py` · `quantlab/data/pairs.py` · nom CLI : `pair`

Le pair trading joue l'écart entre deux actifs corrélés, AAPL et MSFT par défaut. Ce document décrit la stratégie et la façon dont l'application gère plusieurs symboles, en backtest comme en live.

## 1. Données multi-symboles

### Format des barres

Une barre multi-symboles contient :

- pour **chaque** symbole, ses colonnes suffixées : `open_AAPL, high_AAPL, low_AAPL, close_AAPL, volume_AAPL, open_MSFT, …` ;
- les colonnes simples `open … volume`, qui reprennent celles du **premier** symbole, pour le code qui ne connaît qu'un seul actif.

### Synchronisation

- **Backtest** : chaque symbole est chargé séparément (yfinance, Massive, IBKR), puis `combine_symbols` les aligne par une **jointure interne** sur les horodatages. Une minute absente pour l'un des deux est écartée.
- **Live** : `IBKRBarFeed` agrège les barres de 5 s de chaque symbole et n'émet une barre que lorsque **tous** les symboles ont terminé la période. Un symbole sans transaction sur la période est complété avec son dernier close et un volume nul, pour ne pas bloquer le flux.

### Prix d'exécution par symbole

Le broker simulé et la valorisation du portefeuille lisent le prix de chaque ordre dans la colonne de son symbole (`bar_value(bar, "MSFT", "open")` lit `open_MSFT`), et se rabattent sur la colonne simple pour une barre mono-symbole. Un ordre sur MSFT s'exécute donc bien au prix de MSFT.

## 2. Features : l'actif synthétique

Il n'y a plus de colonne `ratio` précalculée ni de feature spéciale. Le ratio est lui-même une feature, et le pipeline enchaîne trois features composables :

```
Ratio(close_AAPL, close_MSFT)            -> ratio_close_AAPL_close_MSFT
  └─ ZScore(ratio, 30)                   -> zscore_30_ratio_close_AAPL_close_MSFT
       └─ SMA(3, source=zscore)          -> sma_3_zscore_30_ratio_...   (lissage sur 3 barres)
```

- Le pipeline résout ces dépendances, calcule chaque étage une seule fois et additionne les warmups : 0 + 29 + 2 barres.
- Le z-score mesure de combien d'écarts-types le ratio s'éloigne de sa moyenne mobile sur 30 barres. Le lissage sur 3 barres filtre les pics isolés.
- Ces features sont causales et testées comme les autres (causalité, warmup, parité batch / barre à barre).
- Le buffer live est **borné** au warmup plus une marge. Avant la refonte, l'historique de la stratégie grossissait sans limite (bug A11), ce qui rendait chaque barre de plus en plus lente.

## 3. Règles de décision

Soit `z`, le z-score lissé de la barre courante :

| Condition | Lecture | Ordres |
|---|---|---|
| `z ≥ zscore_threshold` (2,2) | AAPL est cher par rapport à MSFT | **vente** de 10 AAPL et **achat** de 10 MSFT |
| `z ≤ −zscore_threshold` | AAPL est bon marché par rapport à MSFT | **achat** de 10 AAPL et **vente** de 10 MSFT |

Chaque jambe est un ordre distinct, qui ouvre son propre **lot** dans l'`OrderRouter`. Chaque lot a sa paire d'ordres protecteurs (stop −2 %, objectif +5 %) et sa sortie au temps après 5 barres. Les deux jambes sont donc normalement fermées ensemble, après 5 barres, mais un stop peut fermer une seule jambe et laisser l'autre ouverte.

Comme la stratégie ne fait pas de gestion de position, un z-score qui reste au-delà du seuil génère une nouvelle paire à chaque barre. Les lots s'empilent, dans la limite des `RiskLimits` (`max_position_quantity`) en live.

Il n'y a pas de modèle ML ni de cible : la règle est purement statistique. Une version ML pourrait prédire la vitesse de convergence du ratio avec une `ForwardTarget` calculée sur la série du ratio.

## 4. Paramètres et utilisation

| Paramètre | Défaut | Exemple |
|---|---|---|
| `symbol_a`, `symbol_b` | AAPL, MSFT | les deux `--symbol` dans l'ordre |
| `zscore_threshold` | 2,2 | `--param zscore_threshold=2.0` |
| `zscore_window` | 30 | `--param zscore_window=60` |
| `smoothing_window` | 3 | `--param smoothing_window=5` |

```bash
py -3.10 -m quantlab backtest --strategy pair --symbol AAPL --symbol MSFT --source yfinance --days 55
py -3.10 -m quantlab paper --strategy pair --symbol AAPL --symbol MSFT --session pair_aapl_msft
```

## 5. Résultats actuels

Sur la référence de non-régression (`tests/fixtures/aapl_msft_2m.parquet` : 1 500 barres de 2 min en entraînement, 450 en test, capital de 100 000 $) :

| Trades (jambes) | Taux de réussite | Rendement | Profit factor | Drawdown max | Frais |
|---|---|---|---|---|---|
| 56 | 42,9 % | +0,081 % | 1,36 | −0,10 % | 48,44 $ |

Le profit factor est supérieur à 1, mais l'échantillon (environ 2 jours) est trop court pour conclure.

## 6. Limites et pistes

- **Pas de neutralité en dollars** : 10 actions de chaque côté ne représentent pas le même montant quand les prix diffèrent. Calibrer les quantités sur le ratio des prix, ou sur un coefficient de couverture estimé par régression.
- **Sortie au temps** : une sortie au retour du z-score vers 0 serait plus naturelle. Elle demanderait que la stratégie gère elle-même ses sorties.
- **Stabilité de la relation** : vérifier la cointégration de la paire sur la période (test d'Engle-Granger) avant de trader. Une corrélation qui se casse transforme la stratégie en double pari directionnel.
- **Empilement** : limiter à une paire ouverte à la fois via `on_fill` et l'état de la stratégie.
