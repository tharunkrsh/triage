"""Milestone 4A: Wilson score intervals and Pareto frontier dominance."""

from dataclasses import dataclass
from decimal import Decimal

import pytest

from triage.simulate.models import CandidateSummary
from triage.simulate.stats import (
    Z_95,
    dominates,
    mark_frontier,
    pareto_frontier,
    wilson_interval,
)

# --------------------------------------------------------------------------- Wilson


def test_z_95_constant():
    assert Z_95 == 1.96


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (8, 10, 0.4902, 0.9433),
        (0, 10, 0.0, 0.2775),
        (10, 10, 0.7225, 1.0),
        (1, 1, 0.2065, 1.0),
        (50, 100, 0.4038, 0.5962),
        (950, 1000, 0.9347, 0.9619),
    ],
)
def test_wilson_known_values(k, n, low, high):
    ci = wilson_interval(k, n)
    assert ci.low == pytest.approx(low, abs=1e-4)
    assert ci.high == pytest.approx(high, abs=1e-4)
    assert (ci.successes, ci.n, ci.z) == (k, n, 1.96)
    assert ci.p_hat == pytest.approx(k / n)


def test_wilson_extremes_are_exact():
    assert wilson_interval(0, 7).low == 0.0
    assert wilson_interval(7, 7).high == 1.0


def test_wilson_zero_n_is_uninformative():
    ci = wilson_interval(0, 0)
    assert ci.p_hat is None
    assert (ci.low, ci.high) == (0.0, 1.0)


@pytest.mark.parametrize(("k", "n"), [(-1, 5), (6, 5), (0, -1)])
def test_wilson_rejects_invalid_counts(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 17, 100, 10_000])
def test_wilson_bounds_contain_estimate(n):
    for k in {0, n // 3, n // 2, n}:
        ci = wilson_interval(k, n)
        assert 0.0 <= ci.low <= ci.p_hat <= ci.high <= 1.0


def test_wilson_narrows_with_more_samples():
    widths = [
        wilson_interval(n * 4 // 5, n).high - wilson_interval(n * 4 // 5, n).low
        for n in (10, 100, 1000)
    ]
    assert widths[0] > widths[1] > widths[2]


def test_wilson_custom_z_widens_interval():
    narrow, wide = wilson_interval(8, 10), wilson_interval(8, 10, z=2.576)
    assert wide.z == 2.576
    assert wide.low < narrow.low and wide.high > narrow.high


# --------------------------------------------------------------------------- Pareto


@dataclass(frozen=True)
class P:
    name: str
    cost: Decimal
    accuracy: float


def _p(name, cost, acc):
    return P(name, Decimal(cost), acc)


def test_dominates_strictly_better_on_both():
    assert dominates(_p("a", "1", 0.9), _p("b", "2", 0.8))
    assert not dominates(_p("b", "2", 0.8), _p("a", "1", 0.9))


def test_dominates_equal_cost_higher_accuracy():
    assert dominates(_p("a", "1", 0.9), _p("b", "1", 0.8))


def test_dominates_equal_accuracy_lower_cost():
    assert dominates(_p("a", "1", 0.8), _p("b", "1.5", 0.8))


def test_identical_points_do_not_dominate():
    a, b = _p("a", "1", 0.8), _p("b", "1", 0.8)
    assert not dominates(a, b) and not dominates(b, a)


def test_tradeoff_points_do_not_dominate():
    cheap, good = _p("cheap", "1", 0.6), _p("good", "5", 0.95)
    assert not dominates(cheap, good) and not dominates(good, cheap)


def test_frontier_empty_and_single():
    assert pareto_frontier([]) == []
    only = _p("only", "3", 0.5)
    assert pareto_frontier([only]) == [only]


def test_frontier_drops_dominated_and_keeps_order():
    pts = [
        _p("good", "5", 0.95),
        _p("dominated", "4", 0.5),
        _p("cheap", "1", 0.6),
        _p("tie-a", "2", 0.8),
        _p("tie-b", "2", 0.8),
    ]
    assert [p.name for p in pareto_frontier(pts)] == ["good", "cheap", "tie-a", "tie-b"]


def _summary(model, cost, k, n, *, feature="f", version="v1", logged=None):
    return CandidateSummary(
        feature=feature,
        prompt_version=version,
        model=model,
        n=n,
        passes=k,
        pass_rate=wilson_interval(k, n),
        sim_cost=Decimal("0"),
        projected_cost_per_task=Decimal(cost),
        cost_per_success=None,
        logged_model=logged,
    )


def test_mark_frontier_uses_ci_lower_bound():
    # Same point estimate (0.8) but `big` has a much tighter CI, so its lower bound is higher:
    # it should not be dominated by the equally-priced, noisier `small`.
    small = _summary("small", "1", 4, 5)
    big = _summary("big", "1", 800, 1000)
    marked = {s.model: s.on_frontier for s in mark_frontier([small, big])}
    assert marked == {"small": False, "big": True}


def test_mark_frontier_groups_are_independent():
    rows = [
        _summary("haiku", "1", 90, 100, feature="a"),
        _summary("sonnet", "5", 80, 100, feature="a"),  # dominated within group a
        _summary("sonnet", "5", 80, 100, feature="b"),  # alone in group b
        _summary("haiku", "1", 90, 100, feature="a", version="v2"),
    ]
    flags = [s.on_frontier for s in mark_frontier(rows)]
    assert flags == [True, False, True, True]


def test_mark_frontier_excludes_empty_rows_and_returns_copies():
    empty = _summary("errored-only", "0", 0, 0)
    real = _summary("haiku", "1", 9, 10)
    out = mark_frontier([empty, real])
    assert [s.on_frontier for s in out] == [False, True]
    assert real.on_frontier is False  # input not mutated


def test_mark_frontier_keeps_the_logged_model():
    base = _summary("opus", "10", 10, 10, logged="opus")
    cand = _summary("haiku", "1", 7, 10)
    out = mark_frontier([base, cand])
    assert all(s.on_frontier for s in out)
    assert out[0].logged_model == "opus"
