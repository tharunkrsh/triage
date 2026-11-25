import json

import pytest

from tests.conftest import FIXTURES
from tokenize_cli.ingest import IngestError, load
from tokenize_cli.ingest.jsonl import map_line

SAMPLE = FIXTURES / "ingest" / "sample.jsonl"


def test_sample_reads_all_lines_and_skips_blank():
    result = load(SAMPLE)
    assert result.source_format == "jsonl"
    assert len(result.records) == 3
    assert result.issues == []
    assert not result.aggregated


def test_flat_line_fields():
    r = load(SAMPLE).records[0]
    assert (r.request_id, r.input_tokens, r.output_tokens, r.max_tokens) == (
        "req_001",
        2400,
        180,
        4096,
    )
    assert r.feature == "summarise" and r.api_key_id == "apikey_test_a"


def test_nested_usage_split_cache_and_system_hash():
    r = load(SAMPLE).records[1]
    assert r.request_id == "msg_002"
    assert r.input_tokens == 120000
    assert r.service_tier == "standard"
    assert r.system_prompt_hash is not None and r.system_prompt_hash.startswith("sha256:")


def test_nested_flat_cache_creation_maps_to_5m():
    r = load(SAMPLE).records[2]
    assert r.cache_creation_5m_input_tokens == 3000
    assert r.cache_read_input_tokens == 9000
    assert r.is_batch


def test_same_system_prompt_hashes_equal():
    a, _ = map_line({"model": "m", "system": [{"type": "text", "text": "x"}]})
    b, _ = map_line({"model": "m", "system": [{"type": "text", "text": "x"}]})
    c, _ = map_line({"model": "m", "system": "y"})
    assert a["system_prompt_hash"] == b["system_prompt_hash"] != c["system_prompt_hash"]


def test_top_level_field_overrides_usage():
    row, _ = map_line({"model": "m", "input_tokens": 7, "usage": {"input_tokens": 9}})
    assert row["input_tokens"] == 7


def test_batch_service_tier_sets_is_batch():
    row, _ = map_line({"model": "m", "usage": {"service_tier": "batch"}})
    assert row["is_batch"] is True


def test_null_usage_tokens_dropped():
    usage = {"input_tokens": 5, "cache_creation_input_tokens": None}
    row, _ = map_line({"model": "m", "usage": usage})
    assert "cache_creation_input_tokens" not in row


def _write(tmp_path, lines):
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_bad_lines_collected_with_line_numbers(tmp_path):
    p = _write(
        tmp_path,
        [
            json.dumps({"model": "m", "input_tokens": 1}),
            "{not json",
            "[1, 2]",
            json.dumps({"model": "m", "input_tokens": -5}),
        ],
    )
    result = load(p)
    assert len(result.records) == 1
    assert [i.location for i in result.issues] == ["line 2", "line 3", "line 4"]
    assert "input_tokens" in result.issues[2].message


def test_strict_raises_on_first_bad_line(tmp_path):
    p = _write(tmp_path, [json.dumps({"model": "m"}), "{bad"])
    with pytest.raises(IngestError, match="line 2"):
        load(p, strict=True)


def test_unknown_keys_reported_once(tmp_path):
    lines = [json.dumps({"model": "m", "colour": n}) for n in (1, 2)]
    result = load(_write(tmp_path, lines))
    assert len(result.records) == 2
    assert [i.message for i in result.issues] == ["ignored unrecognised fields: colour"]


def test_simulate_keys_ignored_silently(tmp_path):
    line = {"model": "m", "messages": [], "response": {}, "expected_schema": {}, "tools": []}
    assert load(_write(tmp_path, [json.dumps(line)])).issues == []
