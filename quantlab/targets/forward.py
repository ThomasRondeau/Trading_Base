"""Forward close-price targets and the labels derived from them."""

from __future__ import annotations

from quantlab.targets.base import CloseBasedTarget, ResolvedTargets


class ForwardPrice(CloseBasedTarget):
    """Close price `horizon` bars ahead."""

    horizon: int

    def calculate(self, resolved_dependencies: ResolvedTargets | None = None) -> None:
        self.target = self._trim(self.data["close"].shift(-self.horizon))


class TargetPlus1(ForwardPrice):
    horizon = 1


class TargetPlus5(ForwardPrice):
    horizon = 5


class TargetPlus10(ForwardPrice):
    horizon = 10


class TargetPlus20(ForwardPrice):
    horizon = 20


class TargetAmplitude(CloseBasedTarget):
    """Signed move over 5 bars, in percent: (close[t+5] - close[t]) / close[t] * 100."""

    def calculate(self, resolved_dependencies: ResolvedTargets | None = None) -> None:
        close = self.data["close"]
        self.target = self._trim(((close.shift(-5) - close) / close) * 100)


class TargetDirection(CloseBasedTarget):
    """1 if the close is higher 5 bars ahead, else 0."""

    def calculate(self, resolved_dependencies: ResolvedTargets | None = None) -> None:
        close = self.data["close"]
        self.target = self._trim((close.shift(-5) > close).astype(int))


class TargetOpportunity(CloseBasedTarget):
    """1 if |move over `horizon` bars| > `threshold_pct` (in percent: 0.05 = 0.05 %)."""

    def __init__(self, threshold_pct: float = 0.05, horizon: int = 5) -> None:
        super().__init__()
        self.threshold_pct = threshold_pct
        self.horizon = horizon

    def calculate(self, resolved_dependencies: ResolvedTargets | None = None) -> None:
        close = self.data["close"]
        amplitude_pct = ((close.shift(-self.horizon) - close) / close) * 100
        self.target = self._trim((amplitude_pct.abs() > self.threshold_pct).astype(int))
