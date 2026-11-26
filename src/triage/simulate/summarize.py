"""Aggregate replay outcomes per (feature, prompt_version, model) in an in-memory DuckDB.

Money stays DECIMAL end to end: SQL returns exact sums and counts, and averages are divided as
`Decimal` here, because DuckDB's AVG over a DECIMAL column returns a binary DOUBLE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

import duckdb

from triage.simulate.models import (
    CandidateOutcome,
    CandidateSummary,
    SimulationResult,
    WilsonInterval,
)
from triage.simulate.stats import mark_frontier, wilson_interval

_USD_QUANTUM = Decimal("1e-12")

_CREATE = """
CREATE TABLE outcomes (
    feature VARCHAR, prompt_version VARCHAR, model VARCHAR,
    succeeded BOOLEAN, passed BOOLEAN, det BOOLEAN, judge BOOLEAN,
    sim_cost DECIMAL(38, 12), projected_cost DECIMAL(38, 12)
)
"""

_AGGREGATES = """
    COUNT(passed) AS n,
    COUNT(*) FILTER (WHERE passed) AS passes,
    COUNT(*) FILTER (WHERE NOT succeeded) AS errored,
    SUM(sim_cost) AS sim_cost,
    COUNT(*) FILTER (WHERE succeeded) AS successes,
    SUM(projected_cost) FILTER (WHERE succeeded) AS projected,
    SUM(projected_cost) FILTER (WHERE passed IS NOT NULL) AS graded_projected,
    COUNT(*) FILTER (WHERE det IS NOT NULL AND judge IS NOT NULL) AS doubly_graded,
    COUNT(*) FILTER (WHERE det = judge) AS agreed
"""

_BY_GROUP = f"""
SELECT feature, prompt_version, model, {_AGGREGATES}
FROM outcomes GROUP BY feature, prompt_version, model
"""
_BY_MODEL = f"SELECT model, {_AGGREGATES} FROM outcomes GROUP BY model"
_AGREEMENT = f"SELECT {_AGGREGATES} FROM outcomes"


def _usd(value: Decimal) -> Decimal:
    return value.quantize(_USD_QUANTUM)


def _row(outcome: CandidateOutcome) -> tuple[Any, ...]:
    return (
        outcome.feature,
        outcome.prompt_version,
        outcome.model,
        outcome.status == "succeeded",
        outcome.passed,
        outcome.det.passed if outcome.det is not None else None,
        outcome.judge.passed if outcome.judge is not None else None,
        _usd(outcome.sim_cost),
        _usd(outcome.projected_cost),
    )


def _query(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    cursor = con.execute(sql)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]


def _agreement(row: dict[str, Any]) -> WilsonInterval | None:
    both = row["doubly_graded"]
    return wilson_interval(row["agreed"], both) if both else None


def _summary(row: dict[str, Any], *, logged_model: str | None = None) -> CandidateSummary:
    n, passes, successes = row["n"], row["passes"], row["successes"]
    return CandidateSummary(
        feature=row.get("feature"),
        prompt_version=row.get("prompt_version"),
        model=row["model"],
        n=n,
        passes=passes,
        errored=row["errored"],
        pass_rate=wilson_interval(passes, n),
        sim_cost=row["sim_cost"],
        projected_cost_per_task=_usd(row["projected"] / successes) if successes else Decimal(0),
        cost_per_success=_usd(row["graded_projected"] / passes) if passes else None,
        agreement=_agreement(row),
        logged_model=logged_model,
    )


def _empty_summary(model: str) -> CandidateSummary:
    return CandidateSummary(
        feature=None,
        prompt_version=None,
        model=model,
        n=0,
        passes=0,
        pass_rate=wilson_interval(0, 0),
        sim_cost=Decimal(0),
        projected_cost_per_task=Decimal(0),
        cost_per_success=None,
    )


def summarize_simulation(
    outcomes: Sequence[CandidateOutcome],
    *,
    targets: Sequence[str],
    logged_models: Mapping[tuple[str | None, str | None], str] | None = None,
    judge_model: str | None = None,
    judge_cost: Decimal = Decimal(0),
    source: str | None = None,
) -> SimulationResult:
    """Pass rates count graded tasks only; every target gets an `overall` row, even if empty.

    `logged_models` names the production model of each (feature, prompt_version) slice.
    """
    with duckdb.connect(":memory:") as con:
        con.execute(_CREATE)
        if outcomes:
            con.executemany(
                "INSERT INTO outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [_row(o) for o in outcomes],
            )
        groups = _query(con, _BY_GROUP)
        models = {row["model"]: row for row in _query(con, _BY_MODEL)}
        (totals,) = _query(con, _AGREEMENT)

    order = {model: index for index, model in enumerate(targets)}
    groups.sort(
        key=lambda r: (
            r["feature"] or "",
            r["prompt_version"] or "",
            order.get(r["model"], len(order)),
            r["model"],
        )
    )
    logged = logged_models or {}
    summaries = [
        _summary(row, logged_model=logged.get((row["feature"], row["prompt_version"])))
        for row in groups
    ]
    overall = [_summary(models[m]) if m in models else _empty_summary(m) for m in targets]
    spend = sum((row["sim_cost"] for row in models.values()), Decimal(0))
    return SimulationResult(
        source=source,
        targets=list(targets),
        judge_model=judge_model,
        summaries=mark_frontier(summaries),
        overall=mark_frontier(overall),
        agreement=_agreement(totals),
        total_sim_cost=spend + judge_cost,
    )
