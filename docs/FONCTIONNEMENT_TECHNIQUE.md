# Fonctionnement technique de quantlab

Ce document explique **comment l'application fonctionne** : les modes d'exécution, le trajet d'une barre de marché, la simulation des ordres, la gestion du risque, le trading live avec IBKR et la persistance des sessions. Pour l'organisation du code et la façon de l'étendre, voir [GUIDE_DU_CODE.md](GUIDE_DU_CODE.md).

---

## 1. Objectif et principe directeur

quantlab est un outil de recherche de stratégies de trading intraday fondées sur des indicateurs techniques et du machine learning. Il sert à les **backtester**, les **optimiser** et les **faire tourner en paper trading puis en live** sur Interactive Brokers (IBKR).

Le principe directeur : **un seul code pour le backtest et le live**. Features, modèles, stratégies, gestion du risque et routage des ordres sont identiques dans tous les modes. Seuls deux composants changent :

| Composant | Backtest | Shadow | Paper | Live |
|---|---|---|---|---|
| Source des barres (`BarFeed`) | `HistoricalBarFeed` (rejeu d'un DataFrame) | `IBKRBarFeed` (temps réel) | `IBKRBarFeed` | `IBKRBarFeed` |
| Exécution (`Broker`) | `SimulatedBroker` | `SimulatedBroker` | `IBKRBroker` (compte DU…) | `IBKRBroker` (compte réel) |
| Ordres envoyés à IBKR | non | **non** | oui | oui |

- **Shadow** : les stratégies tournent sur les vraies barres live, mais les fills sont simulés. C'est le mode par défaut tant que `IBKR_READ_ONLY=true`.
- **Paper** : ordres réels sur un compte de démonstration IBKR (identifiant `DU…`).
- **Live** : compte réel. Il exige `IBKR_READ_ONLY=false`, l'option `--live` et la saisie de l'identifiant du compte en confirmation.

Toute optimisation propre au backtest, comme le précalcul des features, doit être **prouvée identique** au chemin live par un test de parité.

On pilote l'application soit par la ligne de commande (`py -3.10 -m quantlab …`), soit par l'**interface graphique** (`py -3.10 -m quantlab ui`), qui s'ouvre dans le navigateur. Les deux font exactement la même chose, avec le même code. L'interface ajoute des graphiques : courbe de capital, drawdown, P&L par trade, scores par fold, suivi des sessions live en direct et écarts du replay.

---

## 2. Vue d'ensemble

```
                 ┌──────────────────────────────────────────────────────────────┐
  Données        │ yfinance · Massive · IBKR historique · fichiers parquet/csv  │
  historiques    └───────────────┬──────────────────────────────────────────────┘
                                 │ validate_ohlcv (UTC, float, trié, sans doublons)
                                 ▼
                       strategy.fit(history)
                       ├─ warm : remplit le buffer de barres (FeatureFeed)
                       └─ train : entraîne les modèles ML (s'il y en a)
                                 │
                                 ▼
 ┌──────────── StrategyRunner : une boucle, un thread, identique partout ────────────┐
 │  pour chaque barre du BarFeed :                                                     │
 │   1. broker.on_bar(bar)        simulateur : exécute les ordres en attente          │
 │   2. broker.process_events()   live : livre les fills reçus du réseau              │
 │   3. portfolio.mark(prix)      valorisation au close                               │
 │   4. router.on_bar_close()     sorties au temps, coupe-circuit                     │
 │   5. strategy.on_bar(bar)      features → décision → ordres d'entrée               │
 │   6. router.submit_entry()     contrôle de risque → broker.submit                  │
 │   7. courbe de capital, latence, journal, état persistant                          │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

Un **backtest**, c'est ce runner alimenté par un `HistoricalBarFeed` et branché sur un `SimulatedBroker`. Une **session live**, c'est le même runner avec un `IBKRBarFeed` et, selon le mode, un `IBKRBroker` ou un `SimulatedBroker`. Il n'existe pas de moteur de backtest séparé.

---

## 3. Données

### 3.1 Sources

| Source | Résolution | Profondeur | Prérequis |
|---|---|---|---|
| yfinance | 2 min | ~60 jours | aucun |
| Massive | 1 min | plusieurs mois | `MASSIVE_API_KEY` dans `.env` |
| IBKR | 1 min | une requête par jour, mise en cache CSV dans `data_cache/` | TWS ou IB Gateway avec l'API activée |
| Fichier | libre | libre | parquet ou csv |

Toutes les sources suivent la même convention de fenêtre, `(symbole, days_back, until)` : depuis il y a `days_back` jours jusqu'à il y a `until` jours.

### 3.2 Format unique (`validate_ohlcv`)

Chaque source renvoie un DataFrame normalisé :

- un index `DatetimeIndex` **UTC** nommé `Datetime`, trié et sans doublons (un horodatage sans fuseau est considéré comme UTC) ;
- des colonnes `open, high, low, close, volume` en minuscules et en `float` ;
- aucune ligne sans close.

Tout est en UTC en interne. Seul le découpage en séances, pour la VWAP et les rendements journaliers, utilise l'heure de New York.

### 3.3 Plusieurs symboles

Les stratégies multi-actifs (pair trading) reçoivent des barres alignées sur les horodatages communs (jointure interne). Chaque symbole y a ses colonnes `<champ>_<SYMBOLE>` (`close_AAPL`, `open_MSFT`…), et les colonnes simples `open…volume` sont celles du premier symbole. `combine_symbols` produit ce format en backtest, et le flux live IBKR produit le même.

---

## 4. Features

### 4.1 Features sans état et causales

Une feature est une **fonction pure des barres passées** : pour chaque ligne, `compute(data, computed)` n'utilise que les lignes jusqu'à celle-ci. Chaque feature déclare :

- un **nom dérivé de ses paramètres** (`sma_20`, `macd_hist_12_26_9`, `bb_upper_20_2`). Deux features de même nom calculent forcément la même chose, donc un indicateur partagé n'est calculé qu'une fois ;
- ses **entrées** (`inputs()`), c'est-à-dire les autres features qu'elle lit. Exemple : un `SMA(3)` lissant un `ZScore` du `Ratio` de deux prix ;
- son **warmup** : le nombre de barres d'historique nécessaires pour que sa valeur soit exacte. Pour une moyenne mobile sur `w` barres, c'est `w-1`. Pour une moyenne exponentielle, c'est le nombre de barres au-delà duquel le poids de l'historique oublié passe sous 1e-10 (`ewm_warmup`).

Indicateurs disponibles : SMA, RSI, ATR, MACD, bandes de Bollinger, ratio de volume, ratio de deux séries, z-score, ADX, et VWAP de séance avec ses bandes. S'y ajoutent trois **jeux composites** pour les modèles (`amplitude_*`, `direction_*`, `opportunity_*`), qui reçoivent leurs indicateurs de base en paramètres.

Les supports et résistances (`levels.py`) **regardent le futur** : ils détectent les extremums sur une fenêtre centrée. Ce sont de simples fonctions, réservées aux graphiques et exclues des features.

### 4.2 Le pipeline

`FeaturePipeline` résout le graphe des dépendances (parcours topologique, détection des cycles et des collisions de noms). Il calcule chaque feature une seule fois et additionne les warmups le long des dépendances.

### 4.3 Deux chemins de calcul, mêmes valeurs

`FeatureFeed` fournit les features à une stratégie :

- **Chemin barre à barre (live, et référence)** : chaque nouvelle barre entre dans un `BarBuffer`, un tampon circulaire préalloué sans `pd.concat`. Sa taille vient du warmup déclaré (warmup total + 1, ou plus si la stratégie lit elle-même des barres brutes, plus une marge de 32). Les features sont calculées sur ce buffer.
- **Mode batch (accélération du backtest)** : `precompute(barres_à_venir)` calcule tout en une fois. Chaque barre ne fait ensuite qu'avancer un curseur. C'est 3 à 9 fois plus rapide. Si une barre ne correspond pas aux données précalculées, le flux repasse automatiquement en mode barre à barre.

Trois familles de tests garantissent que ces deux chemins sont équivalents :

1. **Causalité** : la valeur en `t` calculée sur tout l'historique est égale à celle calculée sur l'historique tronqué en `t`.
2. **Warmup suffisant** : avec exactement le warmup déclaré, la valeur est exacte. C'est ce qui garantit que le buffer live est assez grand.
3. **Parité** : les modes batch et barre à barre donnent les mêmes features, et chaque stratégie produit les mêmes métriques de backtest dans les deux modes.

---

## 5. Modèles et entraînement

### 5.1 Cibles (`ForwardTarget`)

Une cible regarde `horizon` barres dans le futur :

| `kind` | Valeur |
|---|---|
| `price` | close futur |
| `return_pct` | rendement futur, en % |
| `direction` | 1 si le close futur est plus haut, sinon 0 |
| `move` | 1 si \|rendement futur\| dépasse `threshold_pct` (en %) |

Les labels des `horizon` dernières barres valent NaN : ils sont inconnus et n'entrent jamais dans un jeu d'entraînement.

### 5.2 Protection contre le look-ahead

`build_dataset(..., drop_last=n)` retire les `n` dernières barres de l'historique d'entraînement, et `n` doit être au moins égal à l'horizon. Les stratégies ML utilisent `n = 30` pour un horizon de 5. Aucun label ne dépend donc de barres postérieures à la fin de l'historique d'entraînement.

### 5.3 `SklearnModel`

C'est un estimateur compatible scikit-learn (RandomForest, XGBoost…) associé à ses colonnes d'entrée, à sa cible et à sa métrique de score :

- l'entraînement utilise le `n_jobs` configuré ;
- la **prédiction sur une seule barre tourne sur un seul thread**. Lancer un pool de threads à chaque barre coûtait plus cher que la prédiction elle-même (71 ms contre 19 ms pour les trois modèles de Cascade) ;
- la validation croisée est chronologique (`TimeSeriesSplit`) et entraîne des clones de l'estimateur configuré ;
- le score in-sample ne sert qu'à diagnostiquer le surapprentissage, en comparaison avec le score de validation croisée (`CascadeStrategy.evaluate()`).

`ArimaForecaster` ré-estime un ARIMA(p, d, q) à chaque barre sur une fenêtre glissante de closes.

### 5.4 Budget de latence

Un test vérifie que le `on_bar` live de la stratégie la plus lourde (Cascade : features, puis trois prédictions) reste **sous 100 ms en médiane**. On est aujourd'hui autour de 50 ms.

---

## 6. Stratégies

Une stratégie déclare ses features, entraîne éventuellement ses modèles, et **émet uniquement des ordres d'entrée**. Les sorties ne sont pas de son ressort (voir §7).

| Nom CLI | Classe | Logique |
|---|---|---|
| `trend` | `TrendFollowingStrategy` | croisement de deux SMA (5/20 par défaut) |
| `mean_reversion` | `MeanReversionStrategy` | prix nettement hors des bandes de Bollinger, RSI extrême et pic de volume, puis position contraire |
| `ml_hybrid` | `MLHybridStrategy` | signal de retour à la moyenne confirmé par un XGBoost qui prédit la direction à 5 barres |
| `cascade` | `CascadeStrategy` | trois modèles en cascade (opportunité ? → direction ? → amplitude ?) ; trade seulement s'ils concordent et si l'amplitude prédite est suffisante |
| `arima` | `ArimaStrategy` | trade si le mouvement prévu par l'ARIMA dépasse les frais aller-retour |
| `pair` | `PairTradingStrategy` | z-score lissé du ratio de deux prix : vend l'actif surperformant et achète l'autre au-delà d'un seuil |

Chaque stratégie expose `holding_bars` (durée de détention, 5 barres par défaut). Elle peut aussi remplir `diagnostics` (par exemple ses prédictions), qui est enregistré dans le journal à chaque barre.

---

## 7. Exécution des ordres

### 7.1 Ordres et lots

Chaque ordre d'entrée émis par une stratégie ouvre un **lot** qui appartient à cette stratégie. L'`OrderRouter` gère son cycle de vie :

```
 signal (close de t) ──► contrôle de risque ──► ordre MARKET ──► fill (open de t+1)
                                                                    │
                       paire OCA placée chez le broker ◄────────────┘
                       ├─ STOP  au prix d'entrée × (1 − 2 %)   (stop loss)
                       └─ LIMIT au prix d'entrée × (1 + 5 %)   (take profit)
                                    │
         ┌──────────────────────────┼────────────────────────────────────┐
         ▼                          ▼                                    ▼
  le stop est touché        l'objectif est touché            holding_bars écoulées :
  (l'objectif est annulé)   (le stop est annulé)             annulation de la paire OCA,
                                                             sortie au marché
```

- **OCA** (« one cancels all ») : quand l'un des deux ordres s'exécute, le broker annule l'autre. Chez IBKR, ces ordres protecteurs sont en GTC, donc **ils restent actifs même si l'application plante**.
- Les prix protecteurs partent du **prix de fill réel** et sont arrondis au tick (0,01).
- Les fills sont attribués aux lots par identifiant d'ordre. Les fills partiels sont gérés : la protection n'est posée qu'une fois l'entrée complète ou close.
- Plusieurs stratégies peuvent détenir des lots sur le même symbole. La position chez le broker est leur somme nette, et la sortie d'un lot la modifie exactement de la quantité de ce lot. Il n'y a pas de compensation (« netting ») des ordres entre stratégies : la sortie d'un lot reste exacte, au prix de quelques frais en plus.
- **Course critique en live** : si le stop s'exécute pendant qu'une sortie au temps est en vol, le lot est « sur-fermé ». L'excédent devient un lot de correction, refermé immédiatement.

### 7.2 Le broker simulé

Le `SimulatedBroker` sert au backtest et au mode shadow. Ses règles sont réalistes et prudentes :

- un ordre décidé sur le close de la barre `t` n'est **jamais** exécuté à ce close. Un ordre au marché s'exécute à l'**open de `t+1`**, plus le slippage (1 bp par défaut) défavorable à l'ordre ;
- un **stop** se déclenche si le high/low de la barre le traverse. Il s'exécute au prix du stop, ou à l'open si la barre a ouvert au-delà (gap), plus le slippage ;
- une **limite** s'exécute à son prix, ou à un open plus favorable ;
- les ordres protecteurs posés au fill de l'open sont testés sur le reste de cette même barre ;
- si une barre touche **à la fois** le stop et l'objectif, c'est le **stop qui est retenu** (choix prudent) ;
- les **frais** valent 1 bp du notionnel par côté.

### 7.3 Portefeuille

Le `Portfolio` suit le cash, la position nette et le prix moyen par symbole, ainsi que les frais payés. L'**equity** vaut cash plus valeur de marché des positions au dernier close, et elle est enregistrée à chaque barre : c'est la courbe de capital.

---

## 8. Gestion du risque

Le `RiskManager` contrôle **chaque ordre d'entrée** avant son envoi. Les sorties (stops, objectifs, sorties au temps, liquidation) ne sont jamais bloquées.

| Limite (`RiskLimits`) | Effet |
|---|---|
| `stop_loss_pct`, `take_profit_pct` | niveaux des ordres protecteurs (`None` = pas d'ordre) |
| `max_order_quantity` | quantité de l'ordre **réduite** à ce plafond |
| `max_position_quantity` | position nette par symbole, ordres d'entrée en attente compris : l'ordre est réduit, ou refusé s'il n'y a plus de marge |
| `max_daily_loss_pct` | **coupe-circuit** : si l'equity tombe de ce pourcentage sous celle du début de journée, le trading s'arrête, les entrées en attente sont annulées et, par défaut (`flatten_on_halt`), toutes les positions sont fermées |

En session live, la CLI applique par défaut 10 actions par ordre, 50 par position et un coupe-circuit à 1 % de perte journalière.

---

## 9. Mesure de la performance

### 9.1 Métriques (`compute_metrics`)

Elles sont calculées sur la **courbe de capital**, pas sur une somme de pourcentages par trade :

- `total_return_pct` : rendement de l'equity finale par rapport au capital initial ;
- `sharpe_ratio`, `sortino_ratio` : sur les rendements journaliers (séances de New York), annualisés par √252 ;
- `max_drawdown_pct` : plus forte baisse entre un sommet et un creux de l'equity ;
- statistiques par trade : `total_trades`, `win_rate`, `profit_factor`, `avg_trade_return_pct`, `total_pnl_net`, `total_fees`.

Les positions encore ouvertes à la fin d'un backtest sont valorisées au dernier close et comptées comme des trades marqués `open`.

### 9.2 Optimisation walk-forward

Chaque fold du `WalkForwardOptimizer` suit ce découpage :

```
[ train ][ embargo ][ validation ][ embargo ][ test ]
```

1. Chaque combinaison de la grille est entraînée sur *train* et évaluée sur *validation*.
2. La meilleure combinaison est ré-entraînée sur *train + validation*.
3. Elle est évaluée **une seule fois** sur *test*, qui n'a jamais servi à la choisir.

L'embargo retire des barres entre les blocs pour que les cibles construites sur des prix futurs ne fuient pas d'un bloc à l'autre. Le résultat donne le score hors échantillon par fold et la **stabilité des paramètres choisis** : un choix qui change à chaque fold trahit du bruit.

---

## 10. Trading live avec IBKR

### 10.1 Connexion et sécurité

Les réglages se trouvent dans `.env` ou dans l'environnement (voir `.env.example`) :

| Variable | Défaut | Rôle |
|---|---|---|
| `IBKR_HOST` / `IBKR_CLIENT_ID` | 127.0.0.1 / 5 | adresse de TWS ou IB Gateway |
| `IBKR_ACCOUNT` | `paper` | session utilisée par défaut : `paper` ou `live` (option `--account` de la CLI, page Configuration de l'interface) |
| `IBKR_PORT_PAPER` / `IBKR_PORT` | 7497 / 7496 | ports des sessions paper et réelle (IB Gateway : 4002 / 4001) |
| `IBKR_READ_ONLY` | `true` | aucun ordre ne peut partir (mode shadow) |
| `IBKR_ALLOW_LIVE` | `false` | autorise les ordres sur un compte réel |

Les protections s'empilent sur plusieurs niveaux :

1. Choisir le compte paper garantit un compte paper : si le port renvoie un compte réel, la connexion est refusée.
2. Le client refuse `place_order` et `cancel_order` en lecture seule.
3. Le socket lui-même (`GuardedEClient`) refuse `placeOrder`, `cancelOrder` et `reqGlobalCancel` en lecture seule.
4. Sur un compte réel (identifiant autre que `DU…`), les ordres sont refusés sans `IBKR_ALLOW_LIVE`, et la CLI demande en plus `--live` et la saisie de l'identifiant du compte.
5. `IBKRBroker` refuse même de se construire si la connexion est en lecture seule.

### 10.2 Threads

```
 thread réseau ibapi ──► callbacks EWrapper ──► files d'attente (aucun calcul)
                                                     │
 thread stratégie (StrategyRunner) ◄─────────────────┘
   └─ broker.process_events() entre deux barres et à chaque heartbeat (1 s)
```

Le thread réseau ne fait que stocker les données et mettre les événements en file. Il n'est donc jamais bloqué par un calcul lent. Toute la logique de trading tourne sur un **seul** thread, ce qui évite les problèmes de concurrence dans le router et le portefeuille.

### 10.3 Le client (`IBKRClient`)

- Les requêtes bloquantes (historique, positions, ordres ouverts, exécutions) attendent un `Event` propre à chaque requête, avec timeout. Une erreur IBKR est convertie en `IBKRRequestError`.
- Les identifiants de requête et d'ordre sont alloués sous verrou, dans deux plages séparées.
- **Reconnexion automatique** : si la connexion tombe sans qu'on l'ait demandé, le client réessaie avec un délai croissant (1, 2, 5, 10, puis 30 s). Il rétablit ensuite les abonnements aux barres temps réel et prévient le broker, qui redemande les exécutions de la journée.
- Si TWS refuse un abonnement aux barres temps réel (par exemple l'erreur 10089, faute d'abonnement aux données de marché pour l'API), le flux **lève immédiatement une erreur** au lieu d'attendre en silence.

### 10.4 Le flux de barres (`IBKRBarFeed`)

IBKR ne diffuse en temps réel que des barres de 5 secondes. Le flux les agrège en barres de stratégie (1 minute par défaut), alignées sur les multiples de la période et horodatées par leur début, comme les barres historiques. Une barre est émise **dès que son dernier bloc de 5 s arrive**, sans attendre la minute suivante. En multi-symboles, une barre n'est émise que lorsque tous les symboles sont complets. Un symbole sans transaction sur la période est complété avec son dernier close et un volume nul.

`history()` charge l'historique de préchauffage dans le même format. La dernière barre, peut-être encore incomplète, est écartée et sera livrée complète par le flux live.

### 10.5 Le broker IBKR (`IBKRBroker`)

- Notre identifiant d'ordre est transmis dans le champ `orderRef` d'IBKR, ce qui permet de retrouver les ordres après un redémarrage.
- Les ordres au marché sont en `DAY`. Les ordres protecteurs sont en `GTC`, avec un groupe OCA de type 2 : une exécution partielle de l'un réduit la quantité de l'autre.
- Un fill n'est émis qu'à réception du rapport de commission, pour qu'il porte les frais réels, ou après 2 s si le rapport ne vient pas. Les doublons sont écartés par identifiant d'exécution.
- Les statuts terminaux (`Cancelled`, `Inactive`) et les rejets sont remontés au router.
- `reconcile()` compare les positions IBKR à celles du carnet de la session.

---

## 11. Sessions live : journal, reprise, replay

### 11.1 Contenu d'une session

Chaque session a son dossier `sessions/<nom>/` :

| Fichier | Contenu |
|---|---|
| `session.json` | mode, symboles, capital, limites de risque, coûts simulés |
| `strategies.pkl` | les stratégies **juste après l'entraînement** : modèles et buffers |
| `warmup.parquet` | l'historique d'entraînement |
| `journal.sqlite` | tout ce qui s'est passé (voir ci-dessous) |
| `state.json` | l'état du runner après la dernière barre |

### 11.2 Le journal

Le journal est une base SQLite en mode WAL : on n'y fait qu'ajouter des lignes, elle résiste aux plantages et se lit pendant que la session tourne. Elle contient :

- la table `bars` : chaque barre vue (une colonne `catch_up` signale celles rattrapées après un redémarrage) ;
- la table `events`, une ligne par événement, avec la barre concernée, l'heure réelle et un contenu JSON :
  - `features` : la ligne de features de chaque stratégie ;
  - `decision` : ses diagnostics, comme les prédictions ;
  - `signal` : l'ordre émis et la quantité que le risque a laissé passer ;
  - `order`, `cancel`, `fill`, `status` : enregistrés à la frontière du broker ;
  - `trade` : les trades clôturés ;
  - `equity` : equity, cash, positions, état du coupe-circuit, latence ;
  - `session` : démarrage, reprise, arrêt.

### 11.3 État persistant et reprise

Après **chaque barre**, l'état complet est écrit de façon atomique : écriture dans un fichier temporaire, puis renommage, pour qu'un plantage ne laisse jamais de fichier à moitié écrit. Il couvre le portefeuille, les lots ouverts et leurs ordres protecteurs, les ordres en cours, le risque (coupe-circuit, equity de début de journée), le broker et les stratégies.

Une reprise (`--resume`) se déroule ainsi :

1. Les **mêmes modèles** sont rechargés depuis `strategies.pkl`, sans ré-entraînement.
2. Les buffers de features sont reconstruits à partir de l'historique IBKR frais (`strategy.warm`).
3. L'état sauvegardé est restauré.
4. Côté IBKR : les ordres encore actifs sont retrouvés grâce à `orderRef`, les exécutions manquées sont rejouées (les doublons sont ignorés), et les ordres terminés pendant l'arrêt sont signalés comme annulés.
5. **Rattrapage** des barres manquées : le carnet avance (le simulateur exécute ce qui était en attente, les durées de détention progressent, les sorties au temps partent), mais aucune nouvelle décision n'est prise sur ces barres passées.
6. Les positions IBKR sont comparées au carnet : en cas d'écart, la session refuse de démarrer.

Un test vérifie que **couper une session en plein trade puis la reprendre donne exactement les mêmes trades et le même capital** qu'une exécution sans interruption.

### 11.4 Replay : écart entre live et backtest

`quantlab replay --session <nom>` recharge les stratégies telles qu'elles étaient avant la première barre live, les fait tourner sur les barres journalisées avec le broker simulé, puis compare les deux journaux :

- **accord des signaux** : pour chaque (barre, stratégie, symbole, sens), le signal est-il présent en live, en replay ou dans les deux ? Les barres rattrapées sont signalées à part, puisque le live ne pouvait pas décider sur elles ;
- **slippage d'entrée** : prix live comparé au prix simulé, en bp, positif quand le live est moins bon ;
- **écarts de features** entre live et replay ; ils devraient être nuls, puisque c'est le même code sur les mêmes barres ;
- PnL et nombre de trades des deux côtés, latence p50 et p95.

`bar_differences` compare aussi les barres live aux barres historiques téléchargées après coup, pour repérer les corrections de données faites par le fournisseur.

---

## 12. Limites connues

- La validation croisée des modèles (`TimeSeriesSplit`) n'a pas d'écart (`gap`) entre l'entraînement et la validation. Les labels du bord peuvent donc légèrement chevaucher. Le walk-forward, avec son embargo, n'a pas ce défaut.
- Seuls les actions (STK) et les ordres MKT, LMT et STP sont gérés. Pas de bracket natif IBKR : la paire OCA est posée après le fill de l'entrée.
- Les barres temps réel IBKR exigent un abonnement aux données de marché **pour l'API**, sinon erreur 10089.
- Pas de compensation des ordres entre stratégies.
- Les stratégies sont sauvées avec `pickle` : ne recharger que des sessions qu'on a produites soi-même.
