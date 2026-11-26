"""Milestone 3 pricing: fast mode (`speed="fast"`) and Priority Tier (`service_tier="priority"`).

Rates and multipliers come from pricing.yaml (`fast_multiplier` per model,
`service_tier_multipliers` at the top level); nothing here should be hardcoded in src/.
"""

from decimal import Decimal as D

import pytest
from pydantic import ValidationError

from tests.conftest import AUDIT_FIXTURES, M3_PRICING
from triage.ingest import load
from triage.models import UsageRecord
from triage.pricing import load_pricing

M = 1_000_000


def rec(model: str, **kw) -> UsageRecord:
    return UsageRecord(model=model, **kw)


# ---------------------------------------------------------------- fast mode (fixture)


def test_speed_field_defaults_to_standard():
    assert rec("test-flat").speed in (None, "standard")


def test_fast_2x_input_and_output(m3_table):
    c = m3_table.cost(rec("test-deep-cache", input_tokens=M, output_tokens=M, speed="fast"))
    assert c.input_cost == D(16)  # $8 * 2
    assert c.output_cost == D(80)  # $40 * 2
    assert c.total == D(96)
    assert c.fast_applied is True


def test_fast_6x_multiplier(m3_table):
    c = m3_table.cost(rec("test-fast6", input_tokens=M, output_tokens=M, speed="fast"))
    assert c.input_cost == D(30)
    assert c.output_cost == D(150)


def test_standard_speed_unchanged(m3_table):
    c = m3_table.cost(rec("test-deep-cache", input_tokens=M, output_tokens=M, speed="standard"))
    assert c.total == D(48)
    assert c.fast_applied is False


def test_fast_ignores_batch_discount(m3_table):
    plain = m3_table.cost(rec("test-deep-cache", input_tokens=M, output_tokens=M, speed="fast"))
    batch = m3_table.cost(
        rec("test-deep-cache", input_tokens=M, output_tokens=M, speed="fast", is_batch=True)
    )
    assert batch.total == plain.total == D(96)
    assert batch.batch_applied is False


def test_fast_cache_multipliers_stack_on_fast_input_rate(m3_table):
    c = m3_table.cost(
        rec(
            "test-deep-cache",
            cache_creation_5m_input_tokens=M,
            cache_read_input_tokens=M,
            speed="fast",
        )
    )
    assert c.cache_write_5m_cost == D(20)  # $16 * 1.25
    assert c.cache_read_cost == D("0.4")  # $16 * 0.025


def test_fast_on_model_without_fast_pricing_raises(m3_table):
    with pytest.raises(ValueError, match="fast"):
        m3_table.cost(rec("test-flat", input_tokens=1, speed="fast"))


# ---------------------------------------------------------------- priority tier (fixture)


def test_priority_tier_marginal_cost_is_zero(m3_table):
    r = rec(
        "test-flat",
        input_tokens=M,
        output_tokens=M,
        cache_read_input_tokens=M,
        cache_creation_5m_input_tokens=M,
        service_tier="priority",
    )
    c = m3_table.cost(r)
    assert c.total == D(0)
    assert c.input_cost == c.output_cost == c.cache_read_cost == c.cache_write_5m_cost == D(0)
    # Tokens are still on the record for reporting.
    assert r.prompt_tokens == 3 * M
    assert r.output_tokens == M


def test_priority_tier_zero_even_with_fast(m3_table):
    r = rec("test-deep-cache", input_tokens=M, speed="fast", service_tier="priority")
    assert m3_table.cost(r).total == D(0)


@pytest.mark.parametrize("tier", ["standard", None])
def test_non_priority_tiers_pay_list_price(m3_table, tier):
    c = m3_table.cost(rec("test-flat", input_tokens=M, service_tier=tier))
    assert c.total == D(2)


def test_service_tier_multipliers_loaded_from_yaml(m3_table):
    assert m3_table.config.service_tier_multipliers["priority"] == D(0)


# ---------------------------------------------------------------- config validation


def test_negative_fast_multiplier_rejected(tmp_path):
    text = M3_PRICING.read_text(encoding="utf-8").replace(
        "fast_multiplier: 6", "fast_multiplier: 0"
    )
    p = tmp_path / "bad.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_pricing(p)


def test_negative_service_tier_multiplier_rejected(tmp_path):
    text = M3_PRICING.read_text(encoding="utf-8").replace("priority: 0.0", "priority: -1")
    p = tmp_path / "bad.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_pricing(p)


# ---------------------------------------------------------------- real pricing.yaml


@pytest.mark.parametrize(
    "model,rate_in,rate_out",
    [
        ("claude-opus-5-5", D(8), D(40)),
        ("claude-fable-5-2", D(16), D(80)),
        ("claude-mythos-5-2", D(16), D(80)),
        ("claude-opus-5", D(10), D(50)),
        ("claude-opus-4-8", D(10), D(50)),
        ("claude-opus-4-6", D(30), D(150)),
    ],
)
def test_real_fast_mode_rates(real_table, model, rate_in, rate_out):
    c = real_table.cost(rec(model, input_tokens=M, output_tokens=M, speed="fast"))
    assert c.input_cost == rate_in
    assert c.output_cost == rate_out
    batch = real_table.cost(
        rec(model, input_tokens=M, output_tokens=M, speed="fast", is_batch=True)
    )
    assert batch.total == rate_in + rate_out  # no Batch discount on fast mode


@pytest.mark.parametrize(
    "model,multiplier",
    [
        ("claude-opus-5-5", D(2)),
        ("claude-fable-5-2", D(2)),
        ("claude-mythos-5-2", D(2)),
        ("claude-opus-5", D(2)),
        ("claude-opus-4-8", D(2)),
        ("claude-opus-4-6", D(6)),
    ],
)
def test_real_fast_multipliers_in_yaml(real_table, model, multiplier):
    assert real_table.get(model).fast_multiplier == multiplier


def test_real_fast_unsupported_model_raises(real_table):
    with pytest.raises(ValueError):
        real_table.cost(rec("claude-sonnet-5-5", input_tokens=1, speed="fast"))


def test_real_priority_tier_is_free(real_table):
    r = rec("claude-sonnet-5-5", input_tokens=M, output_tokens=M, service_tier="priority")
    assert real_table.cost(r).total == D(0)


def test_real_pricing_has_audit_thresholds(real_table):
    audit = real_table.config.audit
    assert audit is not None
    assert audit.uncached_min_input_tokens == 1024
    assert audit.bloated_max_tokens_ratio == 4
    assert audit.cache_5m_ttl_seconds == 300
    assert audit.uncached_min_repeats >= 2
    assert audit.top_n >= 1


def test_old_pricing_fixture_without_new_sections_still_loads(table):
    # pricing_test.yaml has no fast/service-tier/audit sections: they are optional in the schema.
    assert table.config.audit is None
    assert table.cost(rec("test-flat", input_tokens=M)).total == D(2)


# ---------------------------------------------------------------- ingest of `speed`


def test_console_csv_maps_speed_column():
    records = load(AUDIT_FIXTURES / "console_aggregated.csv").records
    assert [r.speed for r in records] == ["standard", "fast", "standard"]
    assert [r.service_tier for r in records] == ["standard", "standard", "priority"]


def test_jsonl_maps_speed_field():
    records = load(AUDIT_FIXTURES / "bleed.jsonl").records
    fast = [r for r in records if r.request_id == "req_fast"]
    assert fast and fast[0].speed == "fast"


def test_console_fast_row_cost(m3_table):
    records = load(AUDIT_FIXTURES / "console_aggregated.csv").records
    assert m3_table.cost(records[1]).total == D(16)  # 1M * $8 * 2
    assert m3_table.cost(records[2]).total == D(0)  # priority
