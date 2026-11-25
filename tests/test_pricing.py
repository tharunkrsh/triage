from decimal import Decimal as D

import pytest
from pydantic import ValidationError

from tests.conftest import TEST_PRICING
from tokenize_cli.models import UsageRecord
from tokenize_cli.pricing import (
    PRICING_ENV_VAR,
    PricingTable,
    UnknownGeoError,
    UnknownModelError,
    load_pricing,
)

M = 1_000_000


def rec(model: str, **kw) -> UsageRecord:
    return UsageRecord(model=model, **kw)


# ---------------------------------------------------------------- tier cliff (fixture)


def test_cliff_boundary_exactly_threshold_stays_base(table):
    c = table.cost(rec("test-cliff", input_tokens=100_000, output_tokens=1_000))
    assert c.tier_applied is None
    assert c.input_cost == D("0.1")  # 100k * $1/MTok
    assert c.output_cost == D("0.005")


def test_cliff_one_over_threshold_reprices_input_and_output(table):
    c = table.cost(rec("test-cliff", input_tokens=100_001, output_tokens=1_000))
    assert c.tier_applied == 100_000
    assert c.input_cost == D(100_001) * 5 / M
    assert c.output_cost == D("0.025")  # output also 5x


def test_cliff_counts_cache_reads_toward_prompt_length(table):
    r = rec("test-cliff", input_tokens=60_000, cache_read_input_tokens=45_000)
    assert r.prompt_tokens == 105_000
    c = table.cost(r)
    assert c.tier_applied == 100_000
    assert c.input_cost == D("0.3")  # 60k * $5
    assert c.cache_read_cost == D("0.0225")  # 45k * $5 * 0.1


def test_cliff_counts_cache_writes(table):
    r = rec("test-cliff", input_tokens=50_000, cache_creation_1h_input_tokens=50_001)
    assert table.cost(r).tier_applied == 100_000


# ---------------------------------------------------------------- batch


def test_batch_halves_cost(table):
    base = table.cost(rec("test-flat", input_tokens=M, output_tokens=M))
    batch = table.cost(rec("test-flat", input_tokens=M, output_tokens=M, is_batch=True))
    assert base.total == D(12)
    assert batch.total == D(6)
    assert batch.batch_applied


def test_batch_stacks_with_cliff(table):
    c = table.cost(rec("test-cliff", input_tokens=200_000, output_tokens=M, is_batch=True))
    assert c.tier_applied == 100_000
    assert c.input_cost == D("0.5")  # 200k * $5 * 0.5
    assert c.output_cost == D("12.5")  # 1M * $25 * 0.5


def test_batch_stacks_with_cache_multipliers(table):
    c = table.cost(
        rec("test-flat", cache_creation_5m_input_tokens=M, cache_read_input_tokens=M, is_batch=True)
    )
    assert c.cache_write_5m_cost == D("1.25")  # $2 * 1.25 * 0.5
    assert c.cache_read_cost == D("0.05")  # $2 * 0.05 * 0.5


# ---------------------------------------------------------------- cache multipliers


def test_cache_multipliers_default_and_override(table):
    c = table.cost(
        rec(
            "test-cliff",
            cache_creation_5m_input_tokens=10_000,
            cache_creation_1h_input_tokens=10_000,
            cache_read_input_tokens=10_000,
        )
    )
    assert c.cache_write_5m_cost == D("0.0125")
    assert c.cache_write_1h_cost == D("0.02")
    assert c.cache_read_cost == D("0.001")
    flat = table.cost(rec("test-flat", cache_read_input_tokens=M))
    assert flat.cache_read_cost == D("0.1")  # 0.05x override


# ---------------------------------------------------------------- geo


def test_geo_us_applies_only_when_supported(table):
    us = table.cost(rec("test-cliff", input_tokens=100_000, inference_geo="us"))
    assert us.total == D("0.11")  # 100k * $1 * 1.1
    assert us.geo_multiplier == D("1.1")
    flat = table.cost(rec("test-flat", input_tokens=M, inference_geo="us"))
    assert flat.total == D(2)


def test_unknown_geo_raises(table):
    with pytest.raises(UnknownGeoError):
        table.cost(rec("test-flat", input_tokens=1, inference_geo="mars"))


# ---------------------------------------------------------------- resolution


@pytest.mark.parametrize(
    "name,expected",
    [
        ("test-cliff", "test-cliff"),
        ("cliff", "test-cliff"),
        ("test-cliff-20261001", "test-cliff"),
        ("legacy", "test-legacy-20240101"),
        ("test-legacy-20240101", "test-legacy-20240101"),
    ],
)
def test_resolve(table, name, expected):
    assert table.resolve(name) == expected


def test_unknown_model_raises(table):
    with pytest.raises(UnknownModelError, match="pricing.yaml"):
        table.cost(rec("gpt-9"))


def test_cliff_thresholds(table):
    assert table.cliff_thresholds() == {"test-cliff": 100_000}


def test_table_constructible_from_config(table):
    assert PricingTable(table.config).resolve("flat") == "test-flat"


# ---------------------------------------------------------------- loader / validation


def test_env_var_override(monkeypatch, tmp_path):
    target = tmp_path / "p.yaml"
    target.write_text(TEST_PRICING.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv(PRICING_ENV_VAR, str(target))
    assert "test-flat" in load_pricing().models


BASE = (
    "version: 1\n"
    "batch_discount: 0.5\n"
    "cache_multipliers: {write_5m: 1.25, write_1h: 2.0, read: 0.1}\n"
    "inference_geo_multipliers: {global: 1.0}\n"
    "models:\n"
)


@pytest.mark.parametrize(
    "models_yaml",
    [
        "  a: {input: -1, output: 1}",
        "  a: {input: 1, output: 1, tiers: [{above_prompt_tokens: 9, input: 1, output: 1},"
        " {above_prompt_tokens: 5, input: 1, output: 1}]}",
        "  a: {input: 1, output: 1, aliases: [b]}\n  b: {input: 1, output: 1}",
        "  a: {input: 1, output: 1, surprise: true}",
    ],
    ids=["negative-price", "unsorted-tiers", "alias-collision", "unknown-key"],
)
def test_invalid_config_rejected(tmp_path, models_yaml):
    p = tmp_path / "bad.yaml"
    p.write_text(BASE + models_yaml + "\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_pricing(p)


# ---------------------------------------------------------------- real pricing.yaml sanity


def test_real_pricing_loads(real_table):
    assert len(real_table.config.models) >= 20


def test_real_haiku_5_5_cliff_is_5x(real_table):
    haiku = real_table.get("haiku-5-5")
    assert real_table.cliff_thresholds()["claude-haiku-5-5"] == 100_000
    tier = haiku.tiers[0]
    assert tier.input == haiku.input * 5
    assert tier.output == haiku.output * 5
    under = real_table.cost(rec("claude-haiku-5-5", input_tokens=100_000, output_tokens=1_000))
    over = real_table.cost(rec("claude-haiku-5-5", input_tokens=100_001, output_tokens=1_000))
    assert over.output_cost == under.output_cost * 5


def test_real_haiku_5_5_batch_long_tier_rates(real_table):
    c = real_table.cost(rec("claude-haiku-5-5", input_tokens=M, output_tokens=M, is_batch=True))
    assert c.input_cost == D("0.25")
    assert c.output_cost == D("1.25")


def test_real_sonnet_5_5_standard_and_batch(real_table):
    assert real_table.cost(rec("sonnet-5-5", input_tokens=M, output_tokens=M)).total == D(12)
    batch = rec("sonnet-5-5", input_tokens=M, output_tokens=M, is_batch=True)
    assert real_table.cost(batch).total == D(6)


@pytest.mark.parametrize(
    "model,read_cost",
    [
        ("claude-sonnet-5-5", D("0.1")),
        ("claude-fable-5-2", D("0.2")),
        ("claude-haiku-4-5", D("0.1")),
    ],
)
def test_real_cache_read_multipliers(real_table, model, read_cost):
    assert real_table.cost(rec(model, cache_read_input_tokens=M)).cache_read_cost == read_cost


@pytest.mark.parametrize(
    "name,canonical",
    [
        ("claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022"),
        ("claude-3-5-haiku-latest", "claude-3-5-haiku-20241022"),
        ("claude-3-haiku-20240307", "claude-3-haiku-20240307"),
        ("claude-3-opus", "claude-3-opus-20240229"),
        ("claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219"),
    ],
)
def test_real_retired_claude_3_resolve(real_table, name, canonical):
    assert real_table.resolve(name) == canonical
    model = real_table.get(name)
    assert model.retired
    assert model.price_source == "historical"


@pytest.mark.parametrize(
    "name,canonical,total",
    [
        ("claude-opus-4-1", "claude-opus-4-1", D(90)),
        ("claude-opus-4-0", "claude-opus-4", D(90)),
        ("claude-opus-4-20250514", "claude-opus-4", D(90)),
        ("claude-sonnet-4-20250514", "claude-sonnet-4", D(18)),
    ],
)
def test_real_retired_claude_4_rates(real_table, name, canonical, total):
    assert real_table.resolve(name) == canonical
    assert real_table.get(name).retired
    assert real_table.cost(rec(name, input_tokens=M, output_tokens=M)).total == total


def test_real_claude_3_haiku_cost(real_table):
    c = real_table.cost(rec("claude-3-haiku-20240307", input_tokens=M, output_tokens=M))
    assert c.total == D("1.5")
