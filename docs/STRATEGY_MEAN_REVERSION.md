# Stratégie Mean Reversion : Retour à la Moyenne Optimisé

Cette stratégie exploite les excès de prix temporaires par rapport à une moyenne statistique, renforcée par des filtres de conviction.

## 1. Enveloppe de Prix
- **Bandes de Bollinger (20, 2)** : Définit les limites statistiques du mouvement "normal".
- **Filtre ATR** : Pour entrer, le prix ne doit pas juste toucher la bande, mais la dépasser d'un multiple de la volatilité moyenne (0.2 * ATR), assurant une extension réelle du prix.

## 2. Filtres de Conviction
La stratégie ne parie pas aveuglément sur le retournement, elle exige des preuves d'épuisement :
- **Volume Spike (> 1.2)** : Un pic de volume relatif suggère un climax d'achat ou de vente.
- **RSI Extrême** : RSI < 30 pour l'achat (sur-vente) et RSI > 70 pour la vente (sur-achat).

## 3. Logique de Sortie
- Contrairement à une sortie sur la moyenne mobile centrale, cette version utilise la clôture temporelle du moteur (5 barres) pour rester cohérente avec les mesures de performance globales.

## 4. Performance
- **Résultat Backtest** : ~56% de Winrate.
- **Analyse** : Les filtres de volume et d'ATR ont permis de passer d'un winrate aléatoire (50%) à un edge statistiquement significatif.
