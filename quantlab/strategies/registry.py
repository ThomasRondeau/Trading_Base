"""Strategies by name, for the CLI and the live sessions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from quantlab.strategies.arima import ArimaStrategy
from quantlab.strategies.base import Strategy
from quantlab.strategies.cascade import CascadeStrategy
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.strategies.ml_hybrid import MLHybridStrategy
from quantlab.strategies.pair_trading import PairTradingStrategy
from quantlab.strategies.trend_following import TrendFollowingStrategy


@dataclass(frozen=True)
class StrategySpec:
    factory: Callable[..., Strategy]
    n_symbols: int = 1
    #: IBKR history needed to fit it before a live session (the ML strategies train on it)
    live_history: str = "2 D"

    def build(self, symbols: Sequence[str], params: Mapping[str, Any]) -> Strategy:
        if len(symbols) != self.n_symbols:
            raise ValueError(f"{self.factory.__name__} needs {self.n_symbols} symbol(s), got {list(symbols)}.")
        if self.n_symbols == 2:
            return self.factory(symbol_a=symbols[0], symbol_b=symbols[1], **params)
        return self.factory(symbol=symbols[0], **params)


STRATEGIES: dict[str, StrategySpec] = {
    "trend": StrategySpec(TrendFollowingStrategy),
    "mean_reversion": StrategySpec(MeanReversionStrategy),
    "arima": StrategySpec(ArimaStrategy),
    "ml_hybrid": StrategySpec(MLHybridStrategy, live_history="10 D"),
    "cascade": StrategySpec(CascadeStrategy, live_history="10 D"),
    "pair": StrategySpec(PairTradingStrategy, n_symbols=2),
}


def build_strategy(name: str, symbols: Sequence[str], params: Mapping[str, Any] | None = None) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}; available: {', '.join(STRATEGIES)}.")
    return STRATEGIES[name].build(symbols, params or {})
