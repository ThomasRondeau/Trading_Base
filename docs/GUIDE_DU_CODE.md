# Guide du code de quantlab

Ce document décrit **l'organisation du code** : les modules, les classes principales, les conventions, les tests, et comment ajouter une feature, une stratégie, un modèle, une source de données ou un broker. Pour le fonctionnement d'ensemble (modes, simulation, risque, live), voir [FONCTIONNEMENT_TECHNIQUE.md](FONCTIONNEMENT_TECHNIQUE.md).

---

## 1. Démarrage rapide

```bash
py -3.10 -m pip install -e .[dev]        # dépendances (Python 3.10 ; sur cette machine `python` est la 3.8)
py -3.10 -m quantlab --help              # CLI (ou `quantlab --help` après l'installation)
py -3.10 -m quantlab ui                  # interface graphique dans le navigateur (Streamlit)
py -3.10 -m pytest                       # tous les tests (~2 min)
py -3.10 -m pytest -m "not slow"         # sans les backtests de bout en bout (~15 s)
py -3.10 -m ruff check . && py -3.10 -m ruff format .
py -3.10 -m mypy quantlab                # mypy strict, doit rester à zéro erreur
py -3.10 -m pre_commit install           # hooks ruff + mypy avant chaque commit
```

Exemples de commandes :

```bash
py -3.10 -m quantlab backtest --strategy trend --symbol AAPL --source yfinance --days 55 --test-days 5
py -3.10 -m quantlab backtest --strategy pair --symbol AAPL --symbol MSFT --param zscore_threshold=2.0
py -3.10 -m quantlab optimize --strategy trend --symbol AAPL --grid sma_fast=5,10 --grid sma_slow=30,50 \
    --train-size 3000 --validation-size 1000 --test-size 1000
py -3.10 -m quantlab paper --strategy trend --symbol AAPL --session essai     # shadow tant que read-only
py -3.10 -m quantlab replay --session essai
```

Les secrets et les réglages IBKR se mettent dans `.env` (modèle : `.env.example`, et `.env` est ignoré par git).

---

## 2. Conventions

- **Tout le code est en anglais** : noms, docstrings, logs. La documentation peut être en français.
- **`logging` partout, jamais de `print`**. Seule la CLI imprime ses tableaux de résultats.
- **Les `__init__.py` sont vides** : on importe toujours depuis le module qui définit l'objet (`from quantlab.features.technical import SMA`). Cela évite les imports circulaires et les masquages de modules.
- **Typage strict** : `mypy --strict` passe sans erreur, et le hook pre-commit y veille. Les bibliothèques non typées (ibapi, sklearn, xgboost) sont tolérées via `ignore_missing_imports`.
- **ruff** : lignes de 120 caractères, règles `E F W I B UP SIM RUF`.
- **Pas de raccourci réservé au backtest** : tout ce qui décide ou exécute doit fonctionner barre par barre, en live.
- **Horodatages** : UTC en interne. Une barre est une `pd.Series` dont le `name` est son horodatage, comme le renvoie `DataFrame.iterrows()`.
- **Sécurité IBKR** : ne jamais affaiblir les garde-fous de lecture seule et de compte réel (voir §8).

---

## 3. Arborescence

```
quantlab/
├── cli.py                  point d'entrée : backtest | optimize | paper | replay | plot | ui
├── __main__.py             `python -m quantlab`
├── config.py               IBKRSettings.from_env() (IBKR_HOST/PORT/CLIENT_ID/READ_ONLY/ALLOW_LIVE)
├── targets.py              ForwardTarget (cibles des modèles)
├── data/
│   ├── schema.py           validate_ohlcv, bar_value, bar_time
│   ├── pairs.py            combine_symbols (format multi-symboles)
│   ├── loading.py          load_bars, split, ibkr_client (partagés par la CLI et l'interface)
│   ├── data_manager.py     DataManager : yfinance, Massive, IBKR (+ cache CSV)
│   └── massive.py          MassiveAPIClient
├── features/
│   ├── base.py             Feature (classe abstraite), Source, ewm_warmup
│   ├── pipeline.py         FeaturePipeline (dépendances, warmup total)
│   ├── technical.py        SMA, RSI, ATR, MACD, BollingerBands, VolumeRatio, Ratio, ZScore, ADX, VWAP
│   ├── composite.py        AmplitudeFeatures, DirectionFeatures, OpportunityFeatures
│   └── levels.py           support_lines / resistance_lines (NON causal, graphiques seulement)
├── models/
│   ├── sklearn_model.py    SklearnModel, build_dataset
│   └── arima.py            ArimaForecaster
├── strategies/
│   ├── base.py             Strategy (contrat commun backtest / live)
│   ├── registry.py         STRATEGIES, build_strategy (noms utilisés par la CLI)
│   └── trend_following.py, mean_reversion.py, ml_hybrid.py, cascade.py, arima.py, pair_trading.py
├── runtime/                cœur commun aux deux modes
│   ├── buffer.py           BarBuffer (tampon circulaire préalloué)
│   ├── feature_feed.py     FeatureFeed (barre à barre ou précalcul)
│   ├── bar_feed.py         BarFeed (protocole), HistoricalBarFeed
│   ├── runner.py           StrategyRunner (LA boucle de trading)
│   ├── journal.py          Journal SQLite, JournalingBroker, read_journal
│   ├── state.py            StateStore (JSON atomique)
│   └── session.py          SessionFiles (dossier d'une session live)
├── execution/
│   ├── orders.py           Side, OrderType, OrderStatus, Order, Fill
│   ├── broker.py           Broker (protocole)
│   ├── portfolio.py        Portfolio (cash, positions, equity)
│   ├── risk.py             RiskLimits, RiskManager (contrôles, coupe-circuit)
│   ├── router.py           OrderRouter, Lot, Trade
│   ├── simulated.py        SimulatedBroker
│   └── ibkr/
│       ├── client.py       IBKRClient, GuardedEClient, stock(), erreurs
│       ├── broker.py       IBKRBroker, to_ib_order
│       ├── bar_feed.py     IBKRBarFeed, BarAggregator
│       └── live.py         LiveConfig, open_live_session, LiveSession, run_live_session
├── backtest/
│   ├── run.py              backtest() -> BacktestResult
│   ├── metrics.py          compute_metrics, daily_returns, max_drawdown
│   └── walk_forward.py     WalkForwardOptimizer, make_folds, expand_grid
└── analysis/
    ├── replay.py           replay_session, compare_journals, bar_differences
    └── plot.py             session_bars, levels_figure, plot_levels (VWAP + supports et résistances)
└── ui/                     interface graphique Streamlit (libellés en français)
    ├── app.py              les 5 pages : Backtest, Optimisation, Session live, Sessions & replay, Graphique
    ├── charts.py           graphiques Altair (capital, drawdown, P&L par trade, barres, courbes)
    └── live_runner.py      LiveSessions : sessions live en tâche de fond (statut, erreurs, arrêt)
```

---

## 4. Les contrats principaux

### 4.1 `Feature` (`features/base.py`)

```python
class Feature(ABC):
    required_columns: tuple[str, ...] = ()   # colonnes brutes lues (close, volume…)
    causal: bool = True
    name: str                                # propriété abstraite ; nom unique dérivé des paramètres
    outputs: tuple[str, ...]                 # (name,) par défaut ; plusieurs colonnes possibles
    def inputs(self) -> Sequence[Feature]    # autres features lues dans `computed`
    warmup: int                              # propriété abstraite ; historique propre nécessaire
    def compute(self, data, computed) -> pd.Series | pd.DataFrame
```

Une `Source` est soit le nom d'une colonne brute (`"close"`), soit une feature à une seule sortie. Les helpers `source_name`, `source_inputs`, `source_columns` et `source_series` évitent de traiter les deux cas à la main.

### 4.2 `FeaturePipeline` et `FeatureFeed`

- `FeaturePipeline(features)` : `.compute(data) -> DataFrame` (toutes les sorties), `.warmup`, `.columns`, `.required_columns`, `.causal`.
- `FeatureFeed(pipeline, min_history=0)` :
  - pour alimenter : `warm_start(history)`, `update(bar)` (chemin live), `precompute(barres)` (mode batch) ;
  - pour lire : `ready` (le warmup est atteint), `row()` (features de la barre courante), `latest(n)` (n dernières lignes), `bars(n)` (n dernières barres brutes), `batch_mode`.

### 4.3 `Strategy` (`strategies/base.py`)

| À implémenter | Rôle |
|---|---|
| `features()` | features lues par `decide` et par les modèles |
| `decide(bar) -> list[Order]` | **obligatoire** : ordres d'entrée pour la barre courante (`self.feed` la contient déjà) |
| `train(history)` | optionnel : entraînement des modèles |
| `on_fill(fill)` | optionnel : réaction à une exécution |

| Attribut | Rôle |
|---|---|
| `holding_bars` | durée de détention avant la sortie au marché (5 par défaut ; `None` = seulement stop / objectif) |
| `min_history` | barres brutes lues via `feed.bars(n)` en plus du warmup des features |
| `diagnostics` | dictionnaire remis à zéro à chaque barre ; journalisé (ex. prédictions) |

Fourni par la base :

- `fit(history)` = `warm(history)` puis `train(history)`. `warm` reconstruit le buffer sans ré-entraîner, ce qui sert à la reprise d'une session ;
- `prepare(barres)` : précalcul pour le backtest ;
- `start()` et `stop()` ; `on_bar(bar)` filtre arrêt et warmup puis appelle `decide` ;
- `get_state()` et `set_state()` ;
- `order(side, qty, symbol=None)` construit un `Order` attribué à la stratégie.

Un `strategy_id` vaut le nom de la classe par défaut. Il doit être unique dans un runner : passer `strategy_id=…` pour faire tourner deux instances.

### 4.4 Ordres (`execution/orders.py`)

- `Order(symbol, side, quantity, strategy_id, order_type=MARKET, limit_price=None, stop_price=None, oca_group=None, order_id=<uuid>)` : objet figé, sérialisable avec `to_dict` / `from_dict`.
- `Fill(order, price, quantity, timestamp, fees=0.0, exec_id=None)`, avec `signed_quantity`.
- `OrderStatus` : `CANCELLED` ou `REJECTED`, les statuts terminaux remontés par les brokers.

### 4.5 `Broker` (`execution/broker.py`)

```python
class Broker(Protocol):
    def set_handlers(self, on_fill, on_status) -> None
    def submit(self, order) -> None
    def cancel(self, order_id) -> None
    def positions(self) -> dict[str, int]
    def process_events(self) -> None      # livre les événements en file sur le thread appelant
    def on_bar(self, bar) -> None         # le simulateur exécute ici ; no-op en live
    def get_state(self) -> dict ; def set_state(self, state) -> None
```

Implémentations : `SimulatedBroker`, `IBKRBroker`, et `JournalingBroker`, une enveloppe qui enregistre tout dans le journal.

### 4.6 `OrderRouter` (`execution/router.py`)

- Pour envoyer : `submit_entry(order, holding_bars) -> Order | None` (passe par le risque et ouvre un `Lot`).
- Callbacks branchés sur le broker : `on_fill`, `on_status`.
- À chaque barre : `on_bar_close(ts)` gère les sorties au temps et le coupe-circuit ; `flatten(reason)` liquide tout.
- Lecture : `lots` (ouverts), `trades` (`Trade` clôturés : `pnl`, `pnl_net`, `return_pct`), `open_trades(prix, ts)`, `exposure(symbole)`.
- Persistance : `get_state()`, `set_state()`, `orders()`.

### 4.7 `StrategyRunner` (`runtime/runner.py`)

```python
runner = StrategyRunner(strategies, broker, RiskManager(limits), capital=100_000, journal=None, state_store=None)
runner.run(feed)  # bloquant ; runner.stop() depuis un autre thread
runner.step(bar)  # une barre
runner.catch_up(bars)  # rattrapage après redémarrage (pas de décision)
runner.equity, runner.latencies_ms, runner.portfolio, runner.router
runner.get_state() / runner.set_state(state)
```

### 4.8 Points d'entrée de haut niveau

- `backtest(strategies, training, data, capital=…, fee_bps=1, slippage_bps=1, risk_limits=None, precompute_features=True) -> BacktestResult` (`metrics`, `equity`, `trades`, `trades_frame()`).
- `WalkForwardOptimizer(strategy_factory, param_grid, train_size, validation_size, test_size, objective="total_return_pct", embargo=…)`, puis `.run(data)` qui renvoie `WalkForwardResult` (`summary()`, `param_stability()`, `mean_test_score`). La fabrique doit renvoyer une **nouvelle** instance à chaque appel.
- `open_live_session(client, make_strategies, dossier, LiveConfig(...), resume=False)`, qui renvoie `LiveSession`, puis `.run()` / `.stop()`.
- `replay_session(dossier)`, qui renvoie `ReplayReport` (`summary`, `signals`, `entry_fills`, `feature_diffs`).

---

## 5. Étendre l'application

### 5.1 Ajouter une feature

Dans `features/technical.py` :

```python
class EMA(Feature):
    """Exponential moving average."""

    def __init__(self, span: int = 20, source: Source = "close") -> None:
        self.span = span
        self.source = source
        self.required_columns = source_columns(source)

    @property
    def name(self) -> str:
        return f"ema_{self.span}{_suffix(self.source)}"  # nom = paramètres

    def inputs(self) -> Sequence[Feature]:
        return source_inputs(self.source)

    @property
    def warmup(self) -> int:
        return ewm_warmup(self.span)  # EWM : convergence à 1e-10

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.Series:
        return source_series(self.source, data, computed).ewm(span=self.span, adjust=False).mean()
```

Règles à respecter :

- **causale** : `rolling`, `shift(positif)`, `ewm` et `cumsum` conviennent ; `shift(-n)`, une fenêtre centrée ou une normalisation sur toute la série sont interdits ;
- **warmup exact ou majoré** : pour une moyenne mobile sur `w` barres, `w-1` ; un `diff` ou `shift(1)` ajoute 1 ; les warmups des entrées sont additionnés par le pipeline, ne les recompte pas ;
- **nom unique** dérivé de tous les paramètres qui changent le résultat ;
- **ajouter la feature à `PIPELINES` dans `tests/test_features.py`**. Les tests de causalité, de warmup et de parité s'appliquent alors automatiquement.

### 5.2 Ajouter une stratégie

Dans `strategies/breakout.py` :

```python
class BreakoutStrategy(Strategy):
    """Buy when the close breaks above its moving average by k ATRs (and vice versa)."""

    holding_bars = 10

    def __init__(self, window: int = 20, k: float = 2.0, symbol: str = "AAPL", **kwargs: str) -> None:
        super().__init__(symbol=symbol, **kwargs)
        self.k = k
        self.sma, self.atr = SMA(window), ATR(14)

    def features(self) -> Sequence[Feature]:
        return (self.sma, self.atr)

    def decide(self, bar: pd.Series) -> list[Order]:
        row = self.feed.row()
        mid, band = row[self.sma.name], self.k * row[self.atr.name]
        if bar["close"] > mid + band:
            return [self.order(Side.BUY, 10)]
        if bar["close"] < mid - band:
            return [self.order(Side.SELL, 10)]
        return []
```

Ensuite :

1. L'enregistrer dans `strategies/registry.py` : `"breakout": StrategySpec(BreakoutStrategy)`. Ajouter `n_symbols=2` pour une stratégie à deux actifs, et `live_history="10 D"` si elle entraîne un modèle.
2. Elle est alors disponible partout : `quantlab backtest --strategy breakout --param k=1.5`, `optimize`, `paper`.
3. Ajouter un cas dans `tests/test_characterization.py` (puis `UPDATE_BASELINE=1`, voir §7) et ses features dans `tests/test_features.py`.

Points d'attention :

- n'utiliser que `self.feed`, c'est-à-dire les données jusqu'à la barre courante ; jamais de DataFrame global ;
- ne pas gérer les sorties : elles relèvent de `holding_bars` et des `RiskLimits` ;
- pour un modèle ML, implémenter `train(history)` comme dans `MLHybridStrategy`, avec `drop_last` au moins égal à l'horizon de la cible ;
- pour la latence live, garder des prédictions sur une seule ligne (`SklearnModel.predict_row`) ; budget : 100 ms par barre.

### 5.3 Ajouter un modèle

Tout estimateur compatible scikit-learn (avec `fit` et `predict`) peut être enveloppé :

```python
model = SklearnModel(
    GradientBoostingClassifier(max_depth=3), features.outputs, ForwardTarget(5, "direction"), accuracy_score
)
X, y = model.dataset(history, self.feed.pipeline.compute(history), drop_last=30)
model.fit(X, y)
model.predict_row(self.feed.row())
model.cross_validate(X, y)  # validation croisée chronologique sur des clones
```

L'estimateur doit pouvoir être sauvé avec `pickle`, car les sessions live en ont besoin.

### 5.4 Ajouter une source de données

Ajouter une méthode `collect_data_<source>(symbol, days_back, until)` à `DataManager` qui **renvoie `validate_ohlcv(...)`**. Ajouter ensuite son nom dans `SOURCES` et dans `load_bars` de `data/loading.py` (la CLI et l'interface en profitent toutes les deux), et écrire un test avec une réponse simulée, sans réseau, comme pour Massive dans `test_bugfixes.py`.

### 5.5 Ajouter un broker

Implémenter le protocole `Broker` (§4.5) :

- appeler `on_fill` et `on_status` **sur le thread stratégie**. Si l'API notifie depuis un autre thread, mettre les événements en file et les livrer dans `process_events()`, comme le fait `IBKRBroker` ;
- ignorer les doublons d'exécution ;
- fournir `get_state` et `set_state` pour la reprise.

---

### 5.6 L'interface graphique

L'interface (`quantlab/ui/app.py`) n'a pas de logique propre : chaque page appelle les mêmes fonctions que la CLI. Pour ajouter une fonctionnalité, l'écrire dans la bibliothèque, puis brancher la commande CLI **et** la page.

- Les paramètres d'une stratégie sont lus automatiquement dans la signature de son constructeur (`strategy_signature`). Une nouvelle stratégie enregistrée dans `registry.py` apparaît donc dans toutes les pages sans code d'interface, avec un champ libre `clé=valeur` en plus.
- Streamlit ré-exécute le script à chaque interaction. Les résultats à conserver vont dans `st.session_state`, les objets à garder pendant toute la vie du serveur dans `st.cache_resource` (le registre des sessions live), et les données téléchargées dans `st.cache_data`.
- Les sessions live tournent dans des threads (`LiveSessions`). La page les surveille toutes les 5 s en relisant leur journal (`st.fragment(run_every=5)`).
- Graphiques : couleurs par rôle (ordre catégoriel fixe, bleu pour une série unique, bleu / rouge pour gains / pertes), infobulles au survol, heure de New York.

## 6. Points délicats

- **Parité batch et live** : si une stratégie se comporte différemment avec `--bar-by-bar`, c'est qu'une feature n'est pas causale ou que son warmup est trop court. Les tests de `test_features.py` localisent le problème.
- **Fixtures sans fuseau** : `aapl_1m.parquet` a un index sans fuseau. Le runner l'accepte, mais `validate_ohlcv` le convertirait en UTC.
- **`strategy_id` unique** dans un même runner : sinon `StrategyRunner` lève une erreur.
- **Pickle** : une stratégie doit être sauvable pour les sessions. Pas de lambda ni de connexion dans ses attributs. Le test `test_every_strategy_survives_the_session_snapshot` le vérifie.
- **Callbacks IBKR** : ils tournent sur le thread réseau. N'y faire que des copies et des mises en file, jamais de logique de trading.
- **Identifiants de requête IBKR** : ils démarrent à 900 000 000 pour ne pas croiser les identifiants d'ordre, qui partagent le même espace dans les callbacks `error`.
- **Clé Massive** : ne jamais logger une exception `requests` brute, car son message contient l'URL et donc la clé.

---

## 7. Tests

| Fichier | Ce qu'il vérifie |
|---|---|
| `test_characterization.py` (slow) | chaque stratégie, backtestée sur des fixtures figées en batch **et** barre à barre, retrouve exactement `fixtures/baseline_metrics.json` |
| `test_features.py` | causalité, warmup suffisant, parité batch / barre à barre, pipeline, buffer |
| `test_execution.py` | broker simulé (open suivant, gaps, OCA, stop prioritaire), router (lots, fills partiels, rejets, sur-exécution, risque, coupe-circuit), portefeuille, métriques, runner |
| `test_ibkr.py` | client, broker et flux IBKR contre un **faux `EClient`** (`ibkr_fakes.py`) : garde-fous de lecture seule et de compte réel, historique, erreurs, timeouts, reconnexion, commissions, doublons, agrégation des barres, scénario paper complet |
| `test_session.py` | journal, reprise identique à une exécution sans interruption, rattrapage, replay (accord des signaux, mesure du slippage), session shadow complète avec reprise, récupération IBKR après redémarrage, sauvegarde de toutes les stratégies |
| `test_cli.py` | parsing des arguments, registre, commandes `backtest` et `optimize` sur les fixtures |
| `test_ui.py` | interface Streamlit sans navigateur (`AppTest`) : chaque page s'affiche, backtest, replay et graphique de bout en bout, sessions live en tâche de fond sur le faux TWS, refus d'un compte réel non confirmé |
| `test_bugfixes.py` | régressions des bugs corrigés pendant le refactoring (A1 à A12) et fuite de la clé Massive |
| `test_walk_forward.py` | découpage des folds, embargo, sélection des paramètres, instances neuves |
| `test_latency.py` (slow) | latence médiane de `on_bar` pour Cascade < 100 ms |

Les tests n'utilisent **jamais** le réseau ni TWS.

**Changer volontairement un résultat de backtest** (correction de bug, changement de moteur) :

```bash
UPDATE_BASELINE=1 py -3.10 -m pytest tests/test_characterization.py
git diff tests/fixtures/baseline_metrics.json     # relire et justifier chaque écart
```

Un pur refactoring ne doit **jamais** modifier ce fichier.

---

## 8. Sécurité IBKR

- `IBKR_READ_ONLY=true` par défaut. Aucun ordre ne peut partir, ni depuis `IBKRClient.place_order`, ni depuis le socket (`GuardedEClient`).
- Sur un compte réel, les ordres sont refusés sans `IBKR_ALLOW_LIVE=true` ; la CLI exige en plus `--live` et la saisie de l'identifiant du compte.
- Pour tester du code IBKR, utiliser `tests/ibkr_fakes.py`, jamais TWS. Contre un vrai TWS, ne faire que des requêtes de données, avec un client id dédié.
