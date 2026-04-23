# Architecture du Pair Trading et Support Multi-Symboles

L'implémentation de la stratégie de Pair Trading a nécessité une évolution de la gestion des données et du moteur d'exécution pour passer d'un paradigme "Mono-Symbole" à un paradigme "Multi-Symboles / Cross-Asset".

## 1. Pipeline de Données et Synchronisation
Le Pair Trading repose sur la corrélation entre deux actifs (ex: AAPL et MSFT). La difficulté réside dans la synchronisation parfaite des timestamps.

- **Mécanisme de Fusion** : Les données sont récupérées individuellement via `DataManager`, puis fusionnées à l'aide d'une jointure interne (`pandas.merge` sur l'index) dans le script `Main_PairTrading.py`.
- **Suffixage des Colonnes** : Pour éviter les collisions, chaque symbole reçoit un suffixe (ex: `close_AAPL`, `close_MSFT`).
- **Actif Synthétique** : Avant d'être envoyé aux managers, une colonne `ratio` est calculée. Elle représente "l'actif synthétique" sur lequel la stratégie va réellement travailler.

## 2. Gestion des Features (FeatureManager)
Le `FeatureManager` a été utilisé pour traiter ce DataFrame "large" (multi-colonnes) sans modification de son code source :
- **Indicateurs Cross-Asset** : L'indicateur `ZScoreFeature` a été conçu pour ignorer les colonnes de prix standards et cibler spécifiquement la colonne `ratio`.
- **Calcul glissant** : Le manager calcule la moyenne et l'écart-type mobiles du ratio, permettant de déterminer en temps réel si l'écart entre les deux actions est statistiquement anormal.

## 3. Gestion des Cibles (TargetManager)
Dans cette version **statistique** du Pair Trading, le `TargetManager` a été volontairement **omis**.
- **Logique déterministe** : Contrairement aux modèles ML qui cherchent à apprendre un pattern, le Pair Trading utilise une règle mathématique fixe (Mean Reversion). 
- **Évolutivité ML** : Si nous souhaitions créer une version ML (ex: prédire si le ratio va converger plus vite), nous utiliserions le `TargetManager` pour labéliser le futur du `ratio`.

## 4. Évolution du Moteur de Backtest (`BacktestEngine`)
C'est le changement le plus critique. Le moteur est passé d'une lecture aveugle à une lecture intelligente des prix :
- **Symbol-Aware Execution** : Lors d'un ordre, le moteur vérifie si une colonne spécifique au symbole existe (ex: `close_AAPL`). Si oui, il exécute l'ordre à ce prix. Sinon, il se rabat sur la colonne `close` par défaut.
- **Gestion des Positions Simultanées** : Le moteur peut désormais porter simultanément une position longue sur l'actif A et une position courte sur l'actif B, en calculant le PnL individuel de chaque "jambe" de la paire.

## 5. Logique de Décision (Z-Score)
La stratégie utilise un Z-Score lissé sur 3 barres pour filtrer le bruit :
- **Entrée Vente Paire (Z > 2.2)** : Vente AAPL (sur-évalué) + Achat MSFT (sous-évalué).
- **Entrée Achat Paire (Z < -2.2)** : Achat AAPL (sous-évalué) + Vente MSFT (sur-évalué).
- **Sortie** : Les positions sont clôturées par le moteur après un horizon fixe (5 barres), ou pourraient l'être lors du retour du Z-Score à zéro (non implémenté pour rester cohérent avec le reste du système).
