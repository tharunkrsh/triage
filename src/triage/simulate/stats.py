"""Wilson score confidence intervals and Pareto-frontier dominance for simulation results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal
from math import sqrt
from typing import NamedTuple, Protocol

from triage.simulate.models import CandidateSummary, WilsonInterval

Z_95 = 1.96


def wilson_interval(successes: int, n: int, z: float = Z_95) -> WilsonInterval:
    """Wilson score interval for `successes` out of `n`; n == 0 is the uninformative [0, 1]."""
    if n < 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n, got successes={successes}, n={n}")
    if n == 0:
        return WilsonInterval(successes=0, n=0, p_hat=None, low=0.0, high=1.0, z=z)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return WilsonInterval(successes=successes, n=n, p_hat=p, low=low, high=high, z=z)


class ParetoPoint(Protocol):
    @property
    def cost(self) -> Decimal: ...

    @property
    def accuracy(self) -> float: ...


def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """True if `a` is no worse than `b` on cost and accuracy, and strictly better on one."""
    no_worse = a.cost <= b.cost and a.accuracy >= b.accuracy
    return no_worse and (a.cost < b.cost or a.accuracy > b.accuracy)


def pareto_frontier[P: ParetoPoint](points: Sequence[P]) -> list[P]:
    """Non-dominated points in input order (exact ties are all kept)."""
    return [p for p in points if not any(dominates(q, p) for q in points if q is not p)]


class _Point(NamedTuple):
    cost: Decimal
    accuracy: float
    index: int


def mark_frontier(summaries: Sequence[CandidateSummary]) -> list[CandidateSummary]:
    """Copies of `summaries` with `on_frontier` set within each (feature, prompt_version).

    Accuracy is the Wilson lower bound (conservative); cost is projected cost per task.
    Rows without a single pass (including n == 0) are never on the frontier.
    """
    groups: dict[tuple[str | None, str | None], list[_Point]] = defaultdict(list)
    for i, s in enumerate(summaries):
        if s.passes > 0:
            point = _Point(s.projected_cost_per_task, s.pass_rate.low, i)
            groups[(s.feature, s.prompt_version)].append(point)
    on_frontier = {p.index for pts in groups.values() for p in pareto_frontier(pts)}
    return [s.model_copy(update={"on_frontier": i in on_frontier}) for i, s in enumerate(summaries)]
