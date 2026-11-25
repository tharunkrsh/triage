import json
from datetime import UTC, datetime

import pytest

from tests.conftest import FIXTURES
from tokenize_cli.ingest import IngestError, load
from tokenize_cli.ingest.anthropic_csv import map_row

DIR = FIXTURES / "ingest"


# ---------------------------------------------------------------- analytics JSON


def test_analytics_json_rows():
    result = load(DIR / "console_usage_analytics.json")
    assert result.source_format == "console-json"
    assert result.aggregated
    sonnet, haiku = result.records
    assert sonnet.model == "claude-sonnet-5-5"
    assert sonnet.input_tokens == 1284500
    assert sonnet.cache_creation_5m_input_tokens == 215300
    assert sonnet.cache_read_input_tokens == 3200000
    assert sonnet.output_tokens == 891000
    assert sonnet.request_count == 4210
    assert sonnet.timestamp == datetime(2026, 10, 1, tzinfo=UTC)
    assert sonnet.period_end == datetime(2026, 10, 2, tzinfo=UTC)
    assert (sonnet.product, sonnet.actor_id, sonnet.api_key_id) == (
        "claude_code",
        "user_test_1",
        "apikey_test_cc",
    )
    assert sonnet.long_context is False
    assert sonnet.inference_geo == "global"  # null geo normalised
    assert haiku.is_batch and haiku.long_context is True and haiku.inference_geo == "us"


def test_analytics_null_model_row_skipped_with_issue():
    result = load(DIR / "console_usage_analytics.json")
    assert len(result.issues) == 1
    assert result.issues[0].location == "data[2]"
    assert "no model" in result.issues[0].message


def test_cost_report_rejected(tmp_path):
    p = tmp_path / "cost.json"
    p.write_text(json.dumps({"data": [{"group": {}, "cost": {"amount": "1.0"}}]}))
    with pytest.raises(IngestError, match="cost report"):
        load(p)


def test_non_report_json_rejected(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"rows": []}))
    with pytest.raises(IngestError, match="`data` list"):
        load(p)


def test_paged_list_of_reports(tmp_path):
    page = json.loads((DIR / "console_usage_analytics.json").read_text())
    p = tmp_path / "pages.json"
    p.write_text(json.dumps([page, page]))
    result = load(p)
    assert len(result.records) == 4
    assert result.issues[0].location == "page 0 data[2]"


# ---------------------------------------------------------------- legacy JSON


def test_legacy_report_buckets():
    result = load(DIR / "console_usage_legacy.json")
    (r,) = result.records
    assert r.model == "claude-opus-5-5"
    assert (r.input_tokens, r.cache_creation_1h_input_tokens, r.cache_read_input_tokens) == (
        1500,
        100,
        200,
    )
    assert r.request_count is None and r.is_aggregated
    assert r.actor_id == "user_test_2"
    assert r.inference_geo == "global"  # "not_available" normalised
    assert r.timestamp == datetime(2026, 10, 1, tzinfo=UTC)
    assert result.issues == []  # speed, server_tool_use etc. are known-unused


# ---------------------------------------------------------------- CSV


def test_csv_title_case_headers():
    result = load(DIR / "console_usage.csv")
    assert result.source_format == "console-csv"
    sonnet, haiku = result.records
    assert sonnet.input_tokens == 1_000_000 and sonnet.request_count == 500
    assert haiku.is_batch and haiku.cache_read_input_tokens == 500_000
    assert haiku.inference_geo == "global"


def test_csv_bad_row_reported_by_row_number():
    result = load(DIR / "console_usage.csv")
    (issue,) = result.issues
    assert issue.location == "row 4"
    assert "input_tokens" in issue.message


def test_csv_strict_raises():
    with pytest.raises(IngestError, match="row 4"):
        load(DIR / "console_usage.csv", strict=True)


def test_csv_missing_columns_rejected(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("foo,bar\n1,2\n")
    with pytest.raises(IngestError, match="missing usage-report columns"):
        load(p)


def test_csv_unknown_column_warned(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("model,output_tokens,mystery\nclaude-haiku-5-5,10,1\n")
    result = load(p)
    assert result.records[0].request_count is None
    assert result.issues[0].message == "ignored unrecognised fields: mystery"


# ---------------------------------------------------------------- mapping


@pytest.mark.parametrize(
    "column", ["cache_creation.ephemeral_5m_input_tokens", "Ephemeral 5m Input Tokens"]
)
def test_cache_column_spellings(column):
    row, unknown = map_row({"model": "m", column: "7"})
    assert row["cache_creation_5m_input_tokens"] == "7" and not unknown


def test_first_actor_identity_wins():
    row, _ = map_row({"model": "m", "account_id": "user_1", "service_account_id": "svac_1"})
    assert row["actor_id"] == "user_1"


def test_unrecognised_context_window_ignored():
    row, _ = map_row({"model": "m", "context_window": "weird"})
    assert "long_context" not in row
