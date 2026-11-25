"""DuckDB SQL for Mode 1 bleed detection.

Every query runs over the `usage` table built by `bleed.py` (one row per UsageRecord, with
costs pre-computed by pricing.py) and takes thresholds as named parameters, so no pricing
number or threshold is written here.
"""

# Column name -> DuckDB type for the `usage` table.
USAGE_COLUMNS: dict[str, str] = {
    "idx": "BIGINT",
    "request_id": "VARCHAR",
    "model": "VARCHAR",  # canonical pricing.yaml id
    "feature": "VARCHAR",
    "api_key_id": "VARCHAR",
    "workspace_id": "VARCHAR",
    "prompt_hash": "VARCHAR",
    "ts": "DOUBLE",  # epoch seconds
    "aggregated": "BOOLEAN",
    "priority": "BOOLEAN",
    "input_tokens": "BIGINT",
    "output_tokens": "BIGINT",
    "cache_read": "BIGINT",
    "cache_write": "BIGINT",
    "prompt_tokens": "BIGINT",
    "max_tokens": "BIGINT",
    "cost": "DOUBLE",  # USD, as billed
    "base_cost": "DOUBLE",  # USD at base (pre-cliff) rates
    "input_rate": "DOUBLE",  # USD per uncached input token, all multipliers applied
    "read_mult": "DOUBLE",  # model's cache-read multiplier
}

CLIFF_COLUMNS: dict[str, str] = {"model": "VARCHAR", "threshold": "BIGINT"}


def load_sql(table: str, columns: dict[str, str]) -> str:
    """CREATE TABLE from one list parameter per column, zipped by parallel UNNESTs."""
    select = ", ".join(f"UNNEST(?::{typ}[]) AS {name}" for name, typ in columns.items())
    return f"CREATE TABLE {table} AS SELECT {select}"


TOTALS = """
SELECT
    count(*)                                                       AS rows,
    coalesce(sum(cost), 0)                                         AS spend,
    count(*) FILTER (WHERE priority)                               AS priority_rows,
    coalesce(sum(input_tokens) FILTER (WHERE priority), 0)         AS priority_input,
    coalesce(sum(output_tokens) FILTER (WHERE priority), 0)        AS priority_output,
    coalesce(sum(cache_read + cache_write) FILTER (WHERE priority), 0) AS priority_cache
FROM usage
"""

# Spend share per key of one dimension. `{dim}` comes from a fixed whitelist, never input.
CONCENTRATION = """
WITH total AS (SELECT sum(cost) AS t FROM usage)
SELECT
    coalesce({dim}, $unknown) AS key,
    sum(cost)                 AS spend,
    CASE WHEN (SELECT t FROM total) > 0 THEN sum(cost) / (SELECT t FROM total) ELSE 0 END
                              AS share
FROM usage
GROUP BY 1
ORDER BY spend DESC, key
"""

# Identical system prompts sent uncached. Replays each (hash, model) group in time order:
# the first request, and any request arriving more than the TTL after the previous one
# (hits refresh the TTL), pays a 5m cache write; the rest pay the cache-read rate.
# Rows without timestamps are assumed to fall inside the TTL.
UNCACHED_REPEATS = """
WITH eligible AS (
    SELECT
        *,
        count(*)          OVER (PARTITION BY prompt_hash, model) AS n,
        min(input_tokens) OVER (PARTITION BY prompt_hash, model) AS cacheable,
        row_number() OVER (PARTITION BY prompt_hash, model ORDER BY ts NULLS LAST, idx) AS rn,
        lag(ts)      OVER (PARTITION BY prompt_hash, model ORDER BY ts NULLS LAST, idx) AS prev_ts
    FROM usage
    WHERE NOT aggregated
      AND prompt_hash IS NOT NULL
      AND input_tokens >= $min_tokens
      AND cache_read = 0
      AND cache_write = 0
), priced AS (
    SELECT
        prompt_hash, model, n, cacheable,
        cacheable * input_rate AS current_cost,
        cacheable * input_rate
            * CASE WHEN rn = 1 OR ts - prev_ts > $ttl THEN $write_mult ELSE read_mult END
            AS cached_cost
    FROM eligible
    WHERE n >= $min_repeats
)
SELECT
    prompt_hash,
    model,
    any_value(n)                        AS requests,
    any_value(cacheable)                AS cacheable_tokens,
    sum(current_cost)                   AS current_cost,
    sum(cached_cost)                    AS cached_cost,
    sum(current_cost) - sum(cached_cost) AS savings
FROM priced
GROUP BY prompt_hash, model
HAVING sum(current_cost) - sum(cached_cost) > 0
ORDER BY savings DESC, prompt_hash, model
"""

# Single requests priced at a long-context tier. Aggregated rows are skipped: their
# per-request prompt lengths are unknown.
CLIFF = """
SELECT
    u.request_id,
    u.model,
    u.prompt_tokens,
    c.threshold,
    u.prompt_tokens - c.threshold AS excess_tokens,
    u.cost                        AS actual_cost,
    u.base_cost,
    u.cost - u.base_cost          AS savings
FROM usage u
JOIN cliffs c USING (model)
WHERE NOT u.aggregated
  AND u.prompt_tokens > c.threshold
ORDER BY savings DESC, u.idx
"""

BLOATED_MAX_TOKENS = """
SELECT
    request_id,
    model,
    max_tokens,
    output_tokens,
    greatest(max_tokens - output_tokens, 0)                     AS unused_tokens,
    CASE WHEN output_tokens > 0 THEN max_tokens / output_tokens END AS ratio
FROM usage
WHERE NOT aggregated
  AND max_tokens IS NOT NULL
  AND max_tokens > $ratio * output_tokens
ORDER BY unused_tokens DESC, idx
"""
