"""Milestone 3 bleed detectors (`tokenize_cli.audit.run_audit`, DuckDB-backed).

All thresholds come from the `audit:` section of pricing.yaml (fixture: pricing_m3.yaml).
Money is compared as floats with a tight absolute tolerance so the implementation is free to
compute in DuckDB DECIMAL/DOUBLE.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import BaseModel

from tests.conftest import AUDIT_FIXTURES, M3_PRICING, TEST_PRICING
from tokenize_cli.audit import run_audit
from tokenize_cli.ingest import load
from tokenize_cli.models import UsageRecord
from tokenize_cli.pricing import PricingTable

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
_ids = iter(range(1_000_000))


def usd(value: Decimal | float, expected: float) -> bool:
    return float(value) == pytest.approx(expected, abs=1e-9)


def rec(model: str = "test-flat", **kw) -> UsageRecord:
    kw.setdefault("request_id", f"req_{next(_ids)}")
    return UsageRecord(model=model, **kw)


def repeats(
    n: int, *, model="test-flat", tokens=10_000, gap=timedelta(minutes=1), prompt="sha256:p", **kw
):
    """n requests sharing one system prompt hash, `gap` apart."""
    return [
        rec(
            model,
            input_tokens=tokens,
            output_tokens=10,
            system_prompt_hash=prompt,
            timestamp=T0 + i * gap,
            **kw,
        )
        for i in range(n)
    ]


def table_with(tmp_path, old: str, new: str) -> PricingTable:
    text = M3_PRICING.read_text(encoding="utf-8")
    assert old in text
    p = tmp_path / "pricing.yaml"
    p.write_text(text.replace(old, new), encoding="utf-8")
    return PricingTable.load(p)


# ================================================================ result shape


def test_result_is_pydantic(m3_table):
    assert isinstance(run_audit([rec(input_tokens=1)], m3_table), BaseModel)


def test_requires_audit_thresholds_in_pricing():
    # pricing_test.yaml has no `audit:` section; detectors must not fall back to code defaults.
    with pytest.raises(ValueError, match="audit"):
        run_audit([rec("test-flat", input_tokens=1)], PricingTable.load(TEST_PRICING))


# ================================================================ uncached repeated prompts


def test_uncached_repeat_savings_flat_model(m3_table):
    # test-flat: $2/MTok input, 0.05x read. 3 x 10k-token prompt within the 5m TTL:
    # uncached 3 * 10k * $2 = $0.06; cached 1 write (1.25x) $0.025 + 2 reads (0.05x) $0.002.
    result = run_audit(repeats(3), m3_table)
    (finding,) = result.uncached_repeats
    assert finding.system_prompt_hash == "sha256:p"
    assert finding.model == "test-flat"
    assert finding.requests == 3
    assert finding.cacheable_tokens == 10_000
    assert usd(finding.estimated_savings, 0.033)


def test_uncached_repeat_uses_default_read_multiplier(m3_table):
    # test-cliff: $1/MTok, default 0.1x read. 2 x 5k: $0.01 vs $0.00625 + $0.0005.
    (finding,) = run_audit(repeats(2, model="test-cliff", tokens=5_000), m3_table).uncached_repeats
    assert usd(finding.estimated_savings, 0.00325)


def test_uncached_repeat_uses_0025_read_multiplier(m3_table):
    # test-deep-cache: $8/MTok, 0.025x read. 2 x 10k: $0.16 vs $0.10 + $0.002.
    (finding,) = run_audit(repeats(2, model="test-deep-cache"), m3_table).uncached_repeats
    assert usd(finding.estimated_savings, 0.058)


def test_uncached_cacheable_prefix_is_group_minimum(m3_table):
    records = [
        rec(input_tokens=t, system_prompt_hash="sha256:p", timestamp=T0 + timedelta(minutes=i))
        for i, t in enumerate([10_000, 12_000, 11_000])
    ]
    (finding,) = run_audit(records, m3_table).uncached_repeats
    assert finding.cacheable_tokens == 10_000
    assert usd(finding.estimated_savings, 0.033)  # extra suffix tokens cost the same either way


def test_uncached_ttl_expiry_requires_new_write(m3_table):
    # 0m, 2m, 20m: the third request misses the 300s TTL -> 2 writes + 1 read = $0.051.
    records = repeats(2) + [
        rec(
            input_tokens=10_000, system_prompt_hash="sha256:p", timestamp=T0 + timedelta(minutes=20)
        )
    ]
    (finding,) = run_audit(records, m3_table).uncached_repeats
    assert usd(finding.estimated_savings, 0.009)


def test_uncached_reads_refresh_ttl(m3_table):
    # 0m, 4m, 8m, 12m: each hit refreshes the TTL -> 1 write + 3 reads: $0.08 - $0.028.
    (finding,) = run_audit(repeats(4, gap=timedelta(minutes=4)), m3_table).uncached_repeats
    assert usd(finding.estimated_savings, 0.052)


def test_uncached_ttl_read_from_pricing(tmp_path):
    t = table_with(tmp_path, "cache_5m_ttl_seconds: 300", "cache_5m_ttl_seconds: 60")
    # 4m gaps now exceed the TTL: 4 writes cost more than 4 uncached requests -> no finding.
    assert run_audit(repeats(4, gap=timedelta(minutes=4)), t).uncached_repeats == []


def test_uncached_negative_savings_not_flagged(m3_table):
    # Two requests 20 minutes apart: 2 writes ($0.05) > uncached ($0.04).
    assert run_audit(repeats(2, gap=timedelta(minutes=20)), m3_table).uncached_repeats == []


def test_uncached_single_request_not_flagged(m3_table):
    assert run_audit(repeats(1), m3_table).uncached_repeats == []


def test_uncached_groups_by_model(m3_table):
    records = repeats(1, model="test-flat") + repeats(1, model="test-cliff")
    assert run_audit(records, m3_table).uncached_repeats == []


def test_uncached_skips_short_prompts(m3_table):
    assert run_audit(repeats(5, tokens=1023), m3_table).uncached_repeats == []
    assert len(run_audit(repeats(5, tokens=1024), m3_table).uncached_repeats) == 1


def test_uncached_min_tokens_read_from_pricing(tmp_path):
    t = table_with(tmp_path, "uncached_min_input_tokens: 1024", "uncached_min_input_tokens: 50000")
    assert run_audit(repeats(3), t).uncached_repeats == []


def test_uncached_skips_already_cached_requests(m3_table):
    records = repeats(3, cache_read_input_tokens=9_000)
    assert run_audit(records, m3_table).uncached_repeats == []


def test_uncached_skips_missing_hash(m3_table):
    records = [rec(input_tokens=10_000, timestamp=T0 + timedelta(minutes=i)) for i in range(3)]
    assert run_audit(records, m3_table).uncached_repeats == []


def test_uncached_priority_tier_has_no_marginal_savings(m3_table):
    assert run_audit(repeats(3, service_tier="priority"), m3_table).uncached_repeats == []


def test_uncached_without_timestamps_assumes_within_ttl(m3_table):
    records = [rec(input_tokens=10_000, system_prompt_hash="sha256:p") for _ in range(3)]
    (finding,) = run_audit(records, m3_table).uncached_repeats
    assert usd(finding.estimated_savings, 0.033)


# ================================================================ 100k-token cliff


def test_cliff_at_threshold_not_flagged(m3_table):
    assert run_audit([rec("test-cliff", input_tokens=100_000)], m3_table).haiku_cliff == []


def test_cliff_one_over_threshold_flagged_with_penalty(m3_table):
    r = rec("test-cliff", input_tokens=100_001, output_tokens=2_000, request_id="req_over")
    (finding,) = run_audit([r], m3_table).haiku_cliff
    assert finding.request_id == "req_over"
    assert finding.model == "test-cliff"
    assert finding.prompt_tokens == 100_001
    assert finding.threshold == 100_000
    assert finding.excess_tokens == 1
    assert finding.rate_multiplier == 5
    # Actual (5x tier): $0.500005 + $0.05; at base rates: $0.100001 + $0.01.
    assert usd(finding.actual_cost, 0.550005)
    assert usd(finding.base_cost, 0.110001)
    assert usd(finding.estimated_savings, 0.440004)


def test_cliff_counts_cache_tokens_toward_prompt(m3_table):
    r = rec("test-cliff", input_tokens=60_000, cache_read_input_tokens=45_000)
    (finding,) = run_audit([r], m3_table).haiku_cliff
    assert finding.prompt_tokens == 105_000


def test_cliff_ignores_untiered_models(m3_table):
    assert run_audit([rec("test-flat", input_tokens=500_000)], m3_table).haiku_cliff == []


@pytest.mark.parametrize("request_count", [2, None])
def test_cliff_skips_aggregated_rows(m3_table, request_count):
    r = rec("test-cliff", input_tokens=480_000, output_tokens=10, request_count=request_count)
    assert r.is_aggregated
    result = run_audit([r], m3_table)
    assert result.haiku_cliff == []
    assert float(result.total_spend) > 0  # still counted in spend


def test_cliff_real_haiku_5_5_is_5x(real_table):
    r = rec("claude-haiku-5-5", input_tokens=100_001, output_tokens=1_000)
    (finding,) = run_audit([r], real_table).haiku_cliff
    assert finding.rate_multiplier == 5
    # $0.50/$2.50 tier vs $0.10/$0.50 base.
    assert usd(finding.actual_cost, 0.0525005)
    assert usd(finding.estimated_savings, 0.0420004)


def test_cliff_real_haiku_at_threshold_clean(real_table):
    r = rec("claude-haiku-5-5", input_tokens=100_000, output_tokens=1_000)
    assert run_audit([r], real_table).haiku_cliff == []


# ================================================================ bloated max_tokens


def test_bloated_flagged_above_ratio(m3_table):
    r = rec(input_tokens=10, output_tokens=100, max_tokens=401, request_id="req_b")
    (finding,) = run_audit([r], m3_table).bloated_max_tokens
    assert finding.request_id == "req_b"
    assert finding.max_tokens == 401
    assert finding.output_tokens == 100
    assert finding.unused_tokens == 301
    assert finding.ratio == pytest.approx(4.01)
    assert finding.estimated_savings == 0  # rate-limit reservation, not billed tokens


def test_bloated_exact_ratio_not_flagged(m3_table):
    r = rec(input_tokens=10, output_tokens=100, max_tokens=400)
    assert run_audit([r], m3_table).bloated_max_tokens == []


def test_bloated_zero_output_no_division_error(m3_table):
    r = rec(input_tokens=10, output_tokens=0, max_tokens=1_000)
    (finding,) = run_audit([r], m3_table).bloated_max_tokens
    assert finding.ratio is None
    assert finding.unused_tokens == 1_000


def test_bloated_skips_missing_max_tokens(m3_table):
    assert run_audit([rec(input_tokens=10, output_tokens=1)], m3_table).bloated_max_tokens == []


def test_bloated_ratio_read_from_pricing(tmp_path):
    t = table_with(tmp_path, "bloated_max_tokens_ratio: 4", "bloated_max_tokens_ratio: 10")
    r = rec(input_tokens=10, output_tokens=100, max_tokens=500)
    assert run_audit([r], t).bloated_max_tokens == []


@pytest.mark.parametrize("request_count", [5, None])
def test_bloated_skips_aggregated_rows(m3_table, request_count):
    r = rec(input_tokens=10, output_tokens=10, max_tokens=100_000, request_count=request_count)
    assert run_audit([r], m3_table).bloated_max_tokens == []


# ================================================================ spend concentration


def _shares(rows):
    return {row.key: (float(row.spend), row.share) for row in rows}


def test_concentration_by_all_dimensions(m3_table):
    records = [
        rec(input_tokens=1_000_000, feature="a", api_key_id="k1", workspace_id="w1"),  # $2
        rec(input_tokens=1_000_000, feature="a", api_key_id="k2", workspace_id="w1"),  # $2
        rec("test-cliff", input_tokens=1_000_000, request_count=3, feature="b"),  # agg, $5
        rec("test-deep-cache", input_tokens=125_000, api_key_id="k1"),  # $1
    ]
    result = run_audit(records, m3_table)
    c = result.concentration
    assert usd(result.total_spend, 10.0)

    by_model = _shares(c.by_model)
    assert by_model["test-cliff"] == (pytest.approx(5.0), pytest.approx(0.5))
    assert by_model["test-flat"] == (pytest.approx(4.0), pytest.approx(0.4))
    assert by_model["test-deep-cache"] == (pytest.approx(1.0), pytest.approx(0.1))
    assert [row.key for row in c.by_model] == ["test-cliff", "test-flat", "test-deep-cache"]

    assert _shares(c.by_feature)["a"][1] == pytest.approx(0.4)
    assert _shares(c.by_feature)["b"][1] == pytest.approx(0.5)
    assert _shares(c.by_api_key)["k1"][1] == pytest.approx(0.3)
    assert _shares(c.by_workspace)["w1"][1] == pytest.approx(0.4)

    for dim in (c.by_model, c.by_feature, c.by_api_key, c.by_workspace):
        assert sum(row.share for row in dim) == pytest.approx(1.0)


def test_concentration_labels_missing_keys(m3_table):
    c = run_audit([rec(input_tokens=1_000)], m3_table).concentration
    assert [row.key for row in c.by_feature] == ["(unknown)"]
    assert [row.key for row in c.by_workspace] == ["(unknown)"]


def test_concentration_zero_spend_has_zero_shares(m3_table):
    result = run_audit([rec(input_tokens=1_000, service_tier="priority")], m3_table)
    assert result.total_spend == 0
    assert all(row.share == 0 for row in result.concentration.by_model)


# ================================================================ priority tier + totals


def test_priority_tokens_tracked_separately(m3_table):
    records = [
        rec(input_tokens=500_000, output_tokens=50_000, service_tier="priority"),
        rec(input_tokens=1_000_000, cache_read_input_tokens=10, service_tier="priority"),
        rec(input_tokens=1_000_000),
    ]
    result = run_audit(records, m3_table)
    assert usd(result.total_spend, 2.0)  # priority rows add $0
    p = result.priority
    assert p.input_tokens == 1_500_000
    assert p.output_tokens == 50_000
    assert p.cache_tokens == 10
    assert p.total_tokens == 1_550_010
    assert p.rows == 2


def test_fixture_end_to_end(m3_table):
    records = load(AUDIT_FIXTURES / "bleed.jsonl").records
    result = run_audit(records, m3_table)
    assert usd(result.total_spend, 1.745005)
    assert [f.system_prompt_hash for f in result.uncached_repeats] == ["sha256:repeat-a"]
    assert usd(result.uncached_repeats[0].estimated_savings, 0.52)
    assert [f.request_id for f in result.haiku_cliff] == ["req_cliff_over"]
    assert [f.request_id for f in result.bloated_max_tokens] == ["req_bloated"]
    assert result.priority.total_tokens == 550_000
    assert usd(result.total_estimated_savings, 0.960004)


def test_console_aggregated_rows_in_spend_not_per_request_checks(m3_table):
    records = load(AUDIT_FIXTURES / "console_aggregated.csv").records
    assert all(r.is_aggregated for r in records)
    result = run_audit(records, m3_table)
    # cliff row (avg 120k -> 5x tier) $1.25 + fast row $16 + priority $0.
    assert usd(result.total_spend, 17.25)
    assert result.haiku_cliff == []
    assert result.bloated_max_tokens == []
    assert {row.key for row in result.concentration.by_api_key} == {
        "key_agg",
        "key_fast",
        "key_prio",
    }
    assert result.priority.total_tokens == 1_100_000


# ================================================================ top offenders


def test_top_offenders_sorted_by_savings(m3_table):
    records = load(AUDIT_FIXTURES / "bleed.jsonl").records
    offenders = run_audit(records, m3_table).top_offenders
    assert [(o.detector, o.label) for o in offenders[:2]] == [
        ("uncached_repeat", "sha256:repeat-a"),
        ("haiku_cliff", "req_cliff_over"),
    ]
    savings = [float(o.estimated_savings) for o in offenders]
    assert savings == sorted(savings, reverse=True)
    assert all(s > 0 for s in savings)


def test_top_n_read_from_pricing(tmp_path):
    t = table_with(tmp_path, "top_n: 5", "top_n: 1")
    records = load(AUDIT_FIXTURES / "bleed.jsonl").records
    offenders = run_audit(records, t).top_offenders
    assert [o.label for o in offenders] == ["sha256:repeat-a"]


# ================================================================ empty / clean


def test_empty_input_is_clean(m3_table):
    result = run_audit([], m3_table)
    assert result.total_spend == 0
    assert result.total_estimated_savings == 0
    assert result.uncached_repeats == result.haiku_cliff == result.bloated_max_tokens == []
    assert result.top_offenders == []
    c = result.concentration
    assert c.by_model == c.by_feature == c.by_api_key == c.by_workspace == []
    assert result.priority.total_tokens == 0


def test_empty_file_is_clean(m3_table):
    records = load(AUDIT_FIXTURES / "empty.jsonl").records
    assert records == []
    assert run_audit(records, m3_table).total_spend == 0


def test_clean_log_has_no_findings(m3_table):
    result = run_audit(load(AUDIT_FIXTURES / "clean.jsonl").records, m3_table)
    assert float(result.total_spend) > 0
    assert result.uncached_repeats == result.haiku_cliff == result.bloated_max_tokens == []
    assert result.top_offenders == []
    assert result.total_estimated_savings == 0
