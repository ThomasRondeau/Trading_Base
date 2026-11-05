# Historique des évolutions

Ce document retrace les grandes évolutions du projet, de la version scolaire à l'outil actuel. Pour le fonctionnement actuel, voir [FONCTIONNEMENT_TECHNIQUE.md](FONCTIONNEMENT_TECHNIQUE.md) ; pour le code, voir [GUIDE_DU_CODE.md](GUIDE_DU_CODE.md).

---

## Novembre 2026 : refonte vers un outil de recherche et de trading de niveau production

Le projet est passé d'un rendu scolaire à un outil de recherche de stratégies ML **capable de trader en paper et en live sur IBKR avec le même code que le backtest**. La refonte s'est faite en sept étapes. Chacune laissait le projet fonctionnel, avec des tests verts.

### Étape 0 : filet de sécurité

- Outillage : `pyproject.toml`, ruff, mypy, pytest.
- Fixtures de données figées (`tests/fixtures/*.parquet`) et **tests de caractérisation** : chaque stratégie est backtestée et ses métriques comparées à une référence (`baseline_metrics.json`).

### Étape 1 : correction des bugs qui faussaient les résultats

| # | Bug | Effet |
|---|---|---|
| A1 | fenêtres du MACD inversées (`long=12`, `short=26`) | le MACD avait le signe opposé dans Cascade et ML Hybrid |
| A2 | l'ADX perdait l'index des données | indicateur désaligné par rapport aux prix |
| A3 | ARIMA : ordre ignoré, seuil jamais appliqué | la stratégie tradait à **chaque** barre |
| A4 | yfinance : `reset_index()` | index entier, Sharpe et drawdown calculés sur des dates de 1970 |
| A5 | Massive : appel à un attribut inexistant | plantage systématique |
| A6 | moteur : exécution au close de la barre du signal, PnL en somme de % par trade, sortie codée en dur | résultats trop optimistes (corrigé à l'étape 4) |
| A7 | la validation croisée ignorait `max_depth` | hyperparamètres dupliqués et incohérents |
| A8 | seuil d'opportunité défini mais jamais utilisé, unités incohérentes | seuil réel différent du seuil affiché |
| A9 | VWAP cumulée sur tout l'historique | fausse en intraday ; remise à zéro à chaque séance de New York |
| A10 | identifiants de features qui ne correspondaient pas à leur fenêtre | `SMA_5` contenait une SMA de `sma_fast` barres |
| A11 | historique du pair trading sans limite | coût par barre croissant (O(n²) sur un backtest) |
| A12 | singleton IBKR construit directement | plantage à la deuxième instance |
| A13 | supports et résistances calculés en regardant le futur | exclus des features, réservés aux graphiques |

Une fonction `validate_ohlcv` unique normalise désormais toutes les sources : index UTC, colonnes en float, tri, pas de doublons.

### Étape 2 : package `quantlab/`

- Packages en snake_case, imports explicites (fin des imports circulaires), code entièrement en anglais, `logging` au lieu de `print`.
- Configuration typée (`IBKRSettings`), secrets dans `.env`. La clé API Massive, auparavant écrite en dur, a été retirée du code. Elle reste visible dans l'historique git, d'où la nécessité de la révoquer.

### Étape 3 : un cœur pensé pour le live

- **Features sans état et causales**, nommées d'après leurs paramètres, avec un warmup déclaré. Un `FeaturePipeline` résout les dépendances. Les 13 classes recopiées et l'enum `FeatureID` disparaissent.
- **`BarBuffer` et `FeatureFeed`** : un seul chemin de calcul, celui du live, avec un précalcul batch pour accélérer le backtest. Des tests de causalité, de warmup et de parité prouvent que les deux donnent les mêmes valeurs.
- Une seule `ForwardTarget` au lieu de 7 classes de cibles, et un seul `SklearnModel` au lieu de 3 classes de modèles. Les prédictions sur une seule barre tournent sur un seul thread : latence de Cascade divisée par environ 2.
- Une classe de base `Strategy` commune ; `Order` et `Fill` deviennent des objets figés.

### Étape 4 : couche d'exécution commune

- `StrategyRunner` remplace `BacktestEngine` : la même boucle sert au backtest et au live.
- `SimulatedBroker` réaliste : exécution à l'open suivant, slippage, frais, stops et limites testés sur le high et le low, gaps.
- `OrderRouter` : lots par stratégie, stop et objectif placés chez le broker en paire OCA, sortie au temps paramétrable (`holding_bars`), fills partiels, sur-exécutions.
- `RiskManager` branché : taille maximale par ordre et par position, coupe-circuit sur la perte journalière.
- Métriques calculées sur une **vraie courbe de capital**.
- Client IBKR réécrit : requêtes avec timeout, identifiants protégés par verrou, reconnexion automatique, thread réseau séparé, garde-fous de lecture seule et de compte réel. S'y ajoutent `IBKRBroker` et `IBKRBarFeed`, testés avec un faux client sans TWS.

### Étape 5 : sessions live

- Mode **shadow** (barres live, exécutions simulées), paper, et live avec double confirmation.
- Journal SQLite de chaque barre, feature, prédiction, ordre et exécution.
- État persistant après chaque barre, et reprise après plantage (récupération des ordres et exécutions IBKR, rattrapage des barres manquées).
- **Replay** d'une session dans le backtester pour mesurer l'écart entre live et backtest.

### Étape 6 : points d'entrée et outillage

- Une CLI unique (`python -m quantlab backtest | optimize | paper | replay | plot`) remplace les douze scripts `Main_*.py` et `UI_App.py`.
- mypy strict sans erreur, hooks pre-commit (ruff et mypy).
- Correction d'une fuite de la clé Massive dans les logs d'erreur, et ajout d'un timeout aux requêtes.

### Effet sur les résultats

Les taux de réussite publiés en avril 2026 (voir ci-dessous) **ne sont plus valables**. Ils reposaient sur une exécution au close de la barre du signal, impossible en pratique, sur un MACD inversé et, pour l'ARIMA, sur un trade à chaque barre. Sur la référence actuelle (AAPL en 1 min, environ 2,5 jours de test, ce qui est trop court pour conclure) :

| Stratégie | Trades | Taux de réussite | Profit factor | Rendement |
|---|---|---|---|---|
| Cascade | 25 | 52,0 % | 1,22 | +0,0014 % |
| Pair trading (AAPL/MSFT en 2 min) | 56 | 42,9 % | 1,36 | +0,081 % |
| ARIMA | 18 | 44,4 % | 1,00 | ≈ 0 % |
| Mean reversion | 21 | 42,9 % | 0,35 | −0,056 % |
| ML Hybrid | 31 | 38,7 % | 0,26 | −0,125 % |
| Trend following | 72 | 19,4 % | 0,36 | −0,096 % |

Prochaine étape côté recherche : valider en walk-forward sur plusieurs mois (Massive), puis en session shadow ou paper avec replay.

---

## Avril 2026 : premières stratégies (version scolaire)

*Section conservée pour l'historique. Les chiffres ci-dessous ont été obtenus avec l'ancien moteur et ne sont pas fiables (voir plus haut).*

- **Moteur de backtest** : premier suivi des positions, sortie automatique après 5 barres pour correspondre à l'horizon des modèles, calcul du taux de réussite et du PnL.
- **Modèles ML** : régularisation contre le surapprentissage (`max_depth` = 4 pour XGBoost et 5 pour RandomForest, plus d'échantillons minimum par feuille), features continues (z-scores, distances normalisées par l'ATR) à la place d'indicateurs binaires.
- **Cibles** :
  - amplitude signée, pour que le modèle porte aussi la direction ;
  - seuil d'opportunité ramené de 1,5 % à 0,05 % pour des barres de 2 minutes.
- **Nouvelles stratégies** : ARIMA, Trend Following, Mean Reversion filtrée (volume, RSI) et ML Hybrid, en plus de Cascade.
- **Nouveaux indicateurs** : ADX et z-score.
- Taux de réussite annoncés à l'époque : Cascade ~73 %, ML Hybrid ~56,5 %, Mean Reversion ~56 %, ARIMA ~54 %, Trend Following ~42 %.
