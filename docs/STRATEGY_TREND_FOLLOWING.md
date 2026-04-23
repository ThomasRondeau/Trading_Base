# Stratégie Trend Following : Suivi de Tendance Réactif

Approche classique basée sur le momentum des prix et le croisement de moyennes mobiles lissées.

## 1. Indicateurs de Tendance
- **Fast SMA (5 bars)** : Capture les changements de prix rapides.
- **Slow SMA (20 bars)** : Définit la tendance de fond à court terme.
- **Support des Données** : Nécessite un historique minimum de 20 barres avant de générer le premier signal.

## 2. Mécanisme de Crossover
- **Signal d'Achat** : Se déclenche lorsque la SMA rapide croise à la hausse la SMA lente.
- **Signal de Vente** : Se déclenche lorsque la SMA rapide croise à la baisse la SMA lente.

## 3. Optimisations Itératives
Plusieurs filtres ont été testés pour améliorer le winrate initial de 41% :
- **Filtre RSI** : Tentative d'évitement des entrées en fin de mouvement (sur-achat/sur-vente).
- **Filtre ADX** : Tentative de restriction aux tendances fortes (ADX > 15).
*Note : Sur un intervalle de 2 minutes, cette stratégie reste sensible au bruit (choppiness).*

## 4. Performance
- **Résultat Backtest** : ~42% de Winrate.
- **Conclusion** : Cette stratégie nécessite des horizons de temps plus longs (15m ou 1h) pour s'affranchir du bruit microstructurel de l'intervalle 2m.
