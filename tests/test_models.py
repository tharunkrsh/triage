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
