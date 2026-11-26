"""Run the Mode 1 bleed audit: price records, load them into DuckDB, run the detectors.

Per-record pricing stays in pricing.py; grouping, windowing and aggregation happen in the
SQL in `sql.py`. Thresholds come from the `audit:` section of pricing.yaml.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

import duckdb

from triage.audit import sql
from triage.audit.models import (
    AuditResult,
    BloatedMaxTokens,
    CliffFinding,
    Concentration,
    Offender,
    PriorityUsage,
    SpendShare,
    UncachedRepeat,
)
from triage.models import AuditThresholds, UsageRecord
from triage.pricing import PricingTable

PRIORITY_TIER = "priority"
UNKNOWN_KEY = "(unknown)"
# Concentration field -> `usage` column (fixed whitelist interpolated into SQL).
_DIMENSIONS = {
    "by_model": "model",
    "by_feature": "feature",
    "by_api_key": "api_key_id",
    "by_workspace": "workspace_id",
}


def _usd(value: float | None) -> Decimal:
    # DuckDB returns DOUBLE; pin to 1e-12 USD so results are stable Decimals.
    return Decimal(f"{value or 0:.12f}")


def _columns(records: Sequence[UsageRecord], table: PricingTable) -> list[list[Any]]:
    cols: dict[str, list[Any]] = {name: [] for name in sql.USAGE_COLUMNS}
    for idx, r in enumerate(records):
        rates = table.effective_rates(r)
        cost = table.cost(r)
        base = table.cost(r, tiered=False).total if cost.tier_applied is not None else cost.total
        row = {
            "idx": idx,
            "request_id": r.request_id,
            "model": rates.model,
            "feature": r.feature,
            "api_key_id": r.api_key_id,
            "workspace_id": r.workspace_id,
            "prompt_hash": r.system_prompt_hash,
            "ts": r.timestamp.timestamp() if r.timestamp else None,
            "aggregated": r.is_aggregated,
            "priority": r.service_tier == PRIORITY_TIER,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cache_read": r.cache_read_input_tokens,
            "cache_write": r.cache_creation_5m_input_tokens + r.cache_creation_1h_input_tokens,
            "prompt_tokens": r.prompt_tokens,
            "max_tokens": r.max_tokens,
            "cost": float(cost.total),
            "base_cost": float(base),
            "input_rate": float(rates.input),
            "read_mult": float(rates.cache_read_multiplier),
        }
        for name, value in row.items():
            cols[name].append(value)
    return list(cols.values())


def _connect(records: Sequence[UsageRecord], table: PricingTable) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(sql.load_sql("usage", sql.USAGE_COLUMNS), _columns(records, table))
    cliffs = table.cliff_thresholds()
    con.execute(sql.load_sql("cliffs", sql.CLIFF_COLUMNS), [list(cliffs), list(cliffs.values())])
    return con


def _query(con: duckdb.DuckDBPyConnection, query: str, **params: Any) -> list[dict[str, Any]]:
    cur = con.execute(query, params) if params else con.execute(query)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]


def _uncached(con, thresholds: AuditThresholds, table: PricingTable) -> list[UncachedRepeat]:
    rows = _query(
        con,
        sql.UNCACHED_REPEATS,
        min_tokens=thresholds.uncached_min_input_tokens,
        min_repeats=thresholds.uncached_min_repeats,
        ttl=thresholds.cache_5m_ttl_seconds,
        write_mult=float(table.config.cache_multipliers.write_5m),
    )
    return [
        UncachedRepeat(
            system_prompt_hash=r["prompt_hash"],
            model=r["model"],
            requests=r["requests"],
            cacheable_tokens=r["cacheable_tokens"],
            current_cost=_usd(r["current_cost"]),
            cached_cost=_usd(r["cached_cost"]),
            estimated_savings=_usd(r["savings"]),
        )
        for r in rows
    ]


def _cliff(con, table: PricingTable) -> list[CliffFinding]:
    multipliers = table.cliff_rate_multipliers()
    return [
        CliffFinding(
            request_id=r["request_id"],
            model=r["model"],
            prompt_tokens=r["prompt_tokens"],
            threshold=r["threshold"],
            excess_tokens=r["excess_tokens"],
            rate_multiplier=multipliers.get(r["model"], Decimal(1)),
            actual_cost=_usd(r["actual_cost"]),
            base_cost=_usd(r["base_cost"]),
            estimated_savings=_usd(r["savings"]),
        )
        for r in _query(con, sql.CLIFF)
    ]


def _bloated(con, thresholds: AuditThresholds) -> list[BloatedMaxTokens]:
    rows = _query(con, sql.BLOATED_MAX_TOKENS, ratio=float(thresholds.bloated_max_tokens_ratio))
    return [BloatedMaxTokens(**r) for r in rows]


def _concentration(con) -> Concentration:
    shares = {
        field: [
            SpendShare(key=r["key"], spend=_usd(r["spend"]), share=float(r["share"]))
            for r in _query(con, sql.CONCENTRATION.format(dim=column), unknown=UNKNOWN_KEY)
        ]
        for field, column in _DIMENSIONS.items()
    }
    return Concentration(**shares)


def _offenders(
    uncached: list[UncachedRepeat], cliff: list[CliffFinding], top_n: int
) -> list[Offender]:
    candidates = [
        Offender(
            detector="uncached_repeat",
            label=f.system_prompt_hash,
            model=f.model,
            estimated_savings=f.estimated_savings,
        )
        for f in uncached
    ] + [
        Offender(
            detector="haiku_cliff",
            label=f.request_id or UNKNOWN_KEY,
            model=f.model,
            estimated_savings=f.estimated_savings,
        )
        for f in cliff
    ]
    ranked = sorted(
        (o for o in candidates if o.estimated_savings > 0),
        key=lambda o: o.estimated_savings,
        reverse=True,
    )
    return ranked[:top_n]


def run_audit(records: Sequence[UsageRecord], table: PricingTable) -> AuditResult:
    """Price `records` and run every bleed detector. Raises ValueError without thresholds."""
    thresholds = table.config.audit
    if thresholds is None:
        raise ValueError("pricing.yaml has no `audit:` section with bleed-detector thresholds")

    con = _connect(records, table)
    try:
        (totals,) = _query(con, sql.TOTALS)
        uncached = _uncached(con, thresholds, table)
        cliff = _cliff(con, table)
        bloated = _bloated(con, thresholds)
        concentration = _concentration(con)
    finally:
        con.close()

    savings = sum((f.estimated_savings for f in [*uncached, *cliff]), Decimal(0))
    return AuditResult(
        rows=totals["rows"],
        total_spend=_usd(totals["spend"]),
        total_estimated_savings=max(savings, Decimal(0)),
        top_n=thresholds.top_n,
        priority=PriorityUsage(
            rows=totals["priority_rows"],
            input_tokens=totals["priority_input"],
            output_tokens=totals["priority_output"],
            cache_tokens=totals["priority_cache"],
        ),
        uncached_repeats=uncached,
        haiku_cliff=cliff,
        bloated_max_tokens=bloated,
        concentration=concentration,
        top_offenders=_offenders(uncached, cliff, thresholds.top_n),
    )
