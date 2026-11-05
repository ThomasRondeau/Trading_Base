# Stratégie Mean Reversion : retour à la moyenne filtré

`quantlab/strategies/mean_reversion.py` · nom CLI : `mean_reversion`

La stratégie joue les excès de prix temporaires hors des bandes de Bollinger. Elle n'entre que si plusieurs signes d'épuisement du mouvement sont réunis.

## 1. Indicateurs

| Feature | Rôle |
|---|---|
| `BollingerBands(20, 2)` (`bb_upper_20_2`, `bb_lower_20_2`) | enveloppe statistique du mouvement « normal » |
| `ATR(14)` | volatilité : mesure de combien le prix sort des bandes |
| `RSI(14)` | état de surachat ou de survente |
| `VolumeRatio(20)` | volume courant divisé par sa moyenne sur 20 barres |

## 2. Règles de décision

Il faut que le prix **dépasse franchement** la bande, et pas seulement qu'il la touche :

- **achat de 10 actions** si les trois conditions sont réunies :
  - close < bande basse − 0,2 × ATR ;
  - RSI < `rsi_threshold` (30) ;
  - ratio de volume > 1,2 ;
- **vente de 10 actions** si les trois conditions sont réunies :
  - close > bande haute + 0,2 × ATR ;
  - RSI > 100 − `rsi_threshold` (70) ;
  - ratio de volume > 1,2.

Le pic de volume sert d'indice d'un climax d'achat ou de vente. La sortie ne se fait pas sur la moyenne centrale : c'est l'`OrderRouter` qui ferme la position. Il le fait après 5 barres (`holding_bars`), ou plus tôt si le stop loss à −2 % ou le take profit à +5 %, placés chez le broker, est touché.

## 3. Paramètres

| Paramètre | Défaut | Exemple |
|---|---|---|
| `rsi_threshold` | 30 | `--param rsi_threshold=25` |

Les fenêtres des indicateurs, le multiplicateur d'ATR (0,2) et le seuil de volume (1,2) sont fixés dans le code.

## 4. Résultats actuels

Sur la référence de non-régression (AAPL en 1 min, 1 000 barres de test, capital de 100 000 $) :

| Trades | Taux de réussite | Rendement | Profit factor | Drawdown max | Frais |
|---|---|---|---|---|---|
| 21 | 42,9 % | −0,056 % | 0,35 | −0,08 % | 11,64 $ |

L'ancien « ~56 % de réussite » a été obtenu avec un moteur qui exécutait au close de la barre du signal. Ce n'est pas réalisable : on ne connaît ce close qu'une fois la barre terminée. Depuis la refonte, l'exécution a lieu à l'open de la barre suivante, avec slippage. Sur une stratégie de retournement, ce délai coûte justement une partie du rebond.

## 5. Analyse et pistes

- La stratégie trade peu (21 trades sur environ 2,5 jours), ce qui la rend sensible à l'échantillon. Il faut la valider en walk-forward sur plusieurs mois de données Massive (`--source massive --days 180`).
- Pistes : sortie au retour vers la moyenne centrale (`sma_20`) plutôt qu'après un nombre fixe de barres, ordres limites à l'entrée pour ne pas payer le slippage, seuil d'ATR et de volume à optimiser.
- La version **ML Hybrid** ([STRATEGY_ML_HYBRID.md](STRATEGY_ML_HYBRID.md)) reprend ce déclencheur et le filtre par un modèle de direction.
