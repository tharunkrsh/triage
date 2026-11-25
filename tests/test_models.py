import pytest
from pydantic import ValidationError

from tokenize_cli.models import UsageRecord


def test_flat_cache_creation_maps_to_5m():
    r = UsageRecord(model="m", cache_creation_input_tokens=500)
    assert r.cache_creation_5m_input_tokens == 500
    assert r.cache_creation_1h_input_tokens == 0


def test_explicit_split_wins_over_flat():
    r = UsageRecord(
        model="m",
        cache_creation_input_tokens=500,
        cache_creation_5m_input_tokens=200,
        cache_creation_1h_input_tokens=300,
    )
    assert (r.cache_creation_5m_input_tokens, r.cache_creation_1h_input_tokens) == (200, 300)


def test_prompt_tokens_sums_all_input_kinds():
    r = UsageRecord(
        model="m",
        input_tokens=1,
        cache_read_input_tokens=10,
        cache_creation_5m_input_tokens=100,
        cache_creation_1h_input_tokens=1000,
        output_tokens=99999,
    )
    assert r.prompt_tokens == 1111


def test_negative_tokens_rejected():
    with pytest.raises(ValidationError):
        UsageRecord(model="m", input_tokens=-1)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        UsageRecord(model="m", bogus=1)


@pytest.mark.parametrize("geo", [None, "", "not_available"])
def test_missing_geo_normalised_to_global(geo):
    assert UsageRecord(model="m", inference_geo=geo).inference_geo == "global"


def test_aggregation_flags():
    assert not UsageRecord(model="m").is_aggregated
    assert UsageRecord(model="m", request_count=5).is_aggregated
    assert UsageRecord(model="m", request_count=None).is_aggregated


def test_tier_prompt_tokens_averages_aggregates():
    assert UsageRecord(model="m", input_tokens=300, request_count=3).tier_prompt_tokens == 100
    assert UsageRecord(model="m", input_tokens=300, request_count=None).tier_prompt_tokens == 0
    assert UsageRecord(model="m", input_tokens=300).tier_prompt_tokens == 300


def test_request_count_must_be_positive():
    with pytest.raises(ValidationError):
        UsageRecord(model="m", request_count=0)
