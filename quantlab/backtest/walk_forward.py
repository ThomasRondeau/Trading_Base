"""
Optimisation walk-forward générique des hyperparamètres d'une stratégie.

Découpage de chaque fold (positions dans le DataFrame, ordre chronologique) :

    [ train ][ embargo ][ validation ][ embargo ][ test ]

Pour chaque fold :
  1. chaque combinaison de la grille est entraînée sur `train` et évaluée sur `validation` ;
  2. la meilleure combinaison (selon `objective`) est ré-entraînée sur `train + validation` ;
  3. elle est évaluée une seule fois sur `test`, qui n'a jamais servi à choisir les paramètres.

Les scores `test` concaténés de tous les folds forment l'estimation out-of-sample.
L'embargo retire des barres entre les blocs pour éviter que des cibles calculées
sur des prix futurs (Target_Plus_N, training_data_shift) ne débordent sur le bloc suivant.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from Engines.BacktestEngine import BacktestEngine
from Strategies.TradingStrategyInterface import TradingStrategyInterface

logger = logging.getLogger(__name__)

Params = dict[str, Any]
StrategyFactory = Callable[..., TradingStrategyInterface]
Metrics = dict[str, Any]


@dataclass(frozen=True)
class WalkForwardFold:
    """Bornes (positions iloc, fin exclue) des blocs d'un fold."""

    index: int
    train: slice
    validation: slice
    test: slice


@dataclass(frozen=True)
class FoldResult:
    fold: WalkForwardFold
    best_params: Params
    validation_score: float
    test_score: float
    test_metrics: Metrics
    grid_scores: list[tuple[Params, float]]


@dataclass
class WalkForwardResult:
    objective: str
    folds: list[FoldResult] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Une ligne par fold : paramètres retenus, score de validation et métriques de test."""
        rows: list[dict[str, Any]] = []
        for res in self.folds:
            row: dict[str, Any] = {"fold": res.fold.index, **res.best_params}
            row["validation_score"] = res.validation_score
            row["test_score"] = res.test_score
            row.update({f"test_{k}": v for k, v in res.test_metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows).set_index("fold") if rows else pd.DataFrame()

    @property
    def test_scores(self) -> list[float]:
        return [res.test_score for res in self.folds]

    @property
    def mean_test_score(self) -> float:
        valid = [s for s in self.test_scores if math.isfinite(s)]
        return sum(valid) / len(valid) if valid else float("nan")

    def param_stability(self) -> pd.Series:
        """Fréquence de chaque combinaison retenue : une combinaison qui change à chaque fold est un signe de bruit."""
        labels = [", ".join(f"{k}={v}" for k, v in res.best_params.items()) for res in self.folds]
        return pd.Series(labels, dtype="object").value_counts()


def expand_grid(param_grid: Mapping[str, Sequence[Any]]) -> list[Params]:
    """Produit cartésien d'une grille {nom: [valeurs]} -> liste de dicts de paramètres."""
    if not param_grid:
        return [{}]
    names = list(param_grid)
    for name in names:
        if len(param_grid[name]) == 0:
            raise ValueError(f"La grille du paramètre '{name}' est vide.")
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*(param_grid[n] for n in names))]


def make_folds(
    n_bars: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    step: int | None = None,
    embargo: int = 0,
    anchored: bool = False,
) -> list[WalkForwardFold]:
    """
    Calcule les folds walk-forward.

    :param anchored: si True, le début du train reste à 0 et le train s'agrandit à chaque fold ;
                     sinon la fenêtre de train glisse (taille constante).
    :param step: décalage entre deux folds (par défaut `test_size`, soit des blocs de test contigus).
    """
    for name, value in (("train_size", train_size), ("validation_size", validation_size), ("test_size", test_size)):
        if value <= 0:
            raise ValueError(f"{name} doit être > 0 (reçu {value}).")
    if embargo < 0:
        raise ValueError(f"embargo doit être >= 0 (reçu {embargo}).")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError(f"step doit être > 0 (reçu {step}).")

    folds: list[WalkForwardFold] = []
    k = 0
    while True:
        offset = k * step
        train_start = 0 if anchored else offset
        train_end = train_size + offset
        val_start = train_end + embargo
        val_end = val_start + validation_size
        test_start = val_end + embargo
        test_end = test_start + test_size
        if test_end > n_bars:
            break
        folds.append(
            WalkForwardFold(
                index=k,
                train=slice(train_start, train_end),
                validation=slice(val_start, val_end),
                test=slice(test_start, test_end),
            )
        )
        k += 1
    return folds


class WalkForwardOptimizer:
    """
    Recherche des hyperparamètres d'une stratégie par walk-forward.

    Exemple :
        optimizer = WalkForwardOptimizer(
            strategy_factory=TrendFollowingStrategy,
            param_grid={"sma_fast": [5, 10], "sma_slow": [20, 50]},
            train_size=4000, validation_size=1000, test_size=1000, embargo=30,
        )
        result = optimizer.run(data)
        print(result.summary())

    `strategy_factory(**params)` doit renvoyer une instance *neuve* à chaque appel :
    les stratégies enregistrent leurs features dans `train()` et ne peuvent pas être ré-entraînées.
    """

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        param_grid: Mapping[str, Sequence[Any]],
        train_size: int,
        validation_size: int,
        test_size: int,
        objective: str = "total_pnl_net_percent",
        step: int | None = None,
        embargo: int = 0,
        anchored: bool = False,
        min_trades: int = 1,
        verbose: bool = False,
    ) -> None:
        self.strategy_factory = strategy_factory
        self.candidates = expand_grid(param_grid)
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.objective = objective
        self.step = step
        self.embargo = embargo
        self.anchored = anchored
        self.min_trades = min_trades
        self.verbose = verbose

    def run(self, data: pd.DataFrame) -> WalkForwardResult:
        folds = make_folds(
            len(data),
            self.train_size,
            self.validation_size,
            self.test_size,
            step=self.step,
            embargo=self.embargo,
            anchored=self.anchored,
        )
        if not folds:
            needed = self.train_size + self.validation_size + self.test_size + 2 * self.embargo
            raise ValueError(f"Pas assez de données pour un seul fold : {len(data)} barres, {needed} nécessaires.")

        logger.info("Walk-forward : %d folds x %d combinaisons", len(folds), len(self.candidates))
        result = WalkForwardResult(objective=self.objective)
        for fold in folds:
            result.folds.append(self._run_fold(data, fold))
        return result

    def _run_fold(self, data: pd.DataFrame, fold: WalkForwardFold) -> FoldResult:
        train = data.iloc[fold.train]
        validation = data.iloc[fold.validation]

        grid_scores: list[tuple[Params, float]] = []
        for params in self.candidates:
            metrics = self._backtest(params, train, validation)
            score = self._score(metrics)
            grid_scores.append((params, score))
            logger.debug("Fold %d %s -> %s=%.4f", fold.index, params, self.objective, score)

        best_params, best_score = max(grid_scores, key=lambda item: item[1])
        if not math.isfinite(best_score):
            logger.warning("Fold %d : aucune combinaison valide (erreurs ou < %d trades).", fold.index, self.min_trades)

        # Ré-entraînement sur train + validation (l'embargo entre les deux inclus), puis test unique.
        refit = data.iloc[fold.train.start : fold.validation.stop]
        test = data.iloc[fold.test]
        test_metrics = self._backtest(best_params, refit, test)
        test_score = self._score(test_metrics)
        logger.info(
            "Fold %d : best=%s | validation %s=%.4f | test %s=%.4f",
            fold.index, best_params, self.objective, best_score, self.objective, test_score,
        )
        return FoldResult(
            fold=fold,
            best_params=best_params,
            validation_score=best_score,
            test_score=test_score,
            test_metrics=test_metrics,
            grid_scores=grid_scores,
        )

    def _backtest(self, params: Params, training_data: pd.DataFrame, backtest_data: pd.DataFrame) -> Metrics:
        """Entraîne une stratégie neuve sur `training_data` et renvoie les métriques du backtest. {} en cas d'erreur."""
        output = contextlib.nullcontext() if self.verbose else contextlib.redirect_stdout(io.StringIO())
        try:
            with output:
                engine = BacktestEngine()
                engine.set_training_data(training_data)
                engine.set_backtest_data(backtest_data)
                engine.add_strategy(self.strategy_factory(**params))
                engine.train_strategies()
                metrics = engine.run()
        except Exception:
            logger.exception("Backtest en échec pour %s", params)
            return {}
        return dict(metrics) if metrics else {}

    def _score(self, metrics: Metrics) -> float:
        if not metrics or metrics.get("total_trades", 0) < self.min_trades:
            return float("-inf")
        value = metrics.get(self.objective)
        if value is None:
            raise KeyError(f"Métrique '{self.objective}' absente des résultats ; disponibles : {sorted(metrics)}")
        value = float(value)
        return value if math.isfinite(value) else float("-inf")
