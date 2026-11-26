import json
import shutil
from datetime import UTC, datetime

import pytest

from tests.conftest import FIXTURES
from triage.ingest import IngestError, detect_format, load
from triage.ingest.claude_code import project_name

DIR = FIXTURES / "ingest" / "claude_code"
APP = DIR / "C--work-demo-app"
MAIN = APP / "00000000-0000-4000-8000-000000000001.jsonl"
SUB = next(MAIN.with_suffix("").rglob("agent-*.jsonl"))


def _line(mid="msg_1", rid="req_1", model="claude-sonnet-5-5", out=5, **extra):
    obj = {
        "type": "assistant",
        "sessionId": "s",
        "cwd": "/work/app",
        "timestamp": "2026-10-01T09:00:00Z",
        "requestId": rid,
        "message": {
            "id": mid,
            "model": model,
            "usage": {"input_tokens": 10, "output_tokens": out},
        },
    }
    obj.update(extra)
    if rid is None:
        del obj["requestId"]
    if mid is None:
        del obj["message"]["id"]
    return obj


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines)
    path.write_text(text + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("path", [MAIN, SUB])
def test_session_and_subagent_logs_are_detected(path):
    assert detect_format(path) == "claude-code"


def test_detection_reads_past_a_long_first_line_and_a_data_key(tmp_path):
    first = {"type": "user", "sessionId": "s", "data": "x" * 10_000}
    path = _write(tmp_path / "s.jsonl", [first, _line()])
    assert detect_format(path) == "claude-code"


def test_other_formats_are_not_claimed():
    assert detect_format(FIXTURES / "ingest" / "sample.jsonl") == "jsonl"
    assert detect_format(FIXTURES / "simulate" / "replay.jsonl") == "jsonl"
    assert detect_format(FIXTURES / "ingest" / "console_usage_legacy.json") == "console-json"


def test_one_record_per_response():
    result = load(MAIN)
    assert result.source_format == "claude-code"
    assert [r.request_id for r in result.records] == ["req_synthetic_001", "req_synthetic_002"]
    first = result.records[0]
    assert (first.input_tokens, first.output_tokens, first.cache_read_input_tokens) == (
        10,
        200,
        5000,
    )
    assert first.cache_creation_1h_input_tokens == 1000
    assert first.timestamp == datetime(2026, 10, 1, 9, 0, 1, tzinfo=UTC)
    assert result.records[1].cache_creation_5m_input_tokens == 400


def test_synthetic_zero_token_and_other_events_are_silent():
    result = load(MAIN)
    assert result.issues == []
    assert {r.model for r in result.records} == {"claude-opus-5-5", "claude-sonnet-5-5"}


def test_records_carry_project_and_product_but_no_max_tokens_or_prompt_hash():
    records = load(MAIN).records
    assert {(r.feature, r.product) for r in records} == {("demo-app", "claude-code")}
    assert all(r.max_tokens is None and r.system_prompt_hash is None for r in records)
    assert "synthetic prompt" not in "".join(r.model_dump_json() for r in records)


def test_subagent_takes_the_parent_project():
    (record,) = load(SUB).records
    assert (record.feature, record.product) == ("demo-app", "claude-code:subagent")


def test_subagent_without_a_parent_log_uses_its_own_cwd(tmp_path):
    copy = tmp_path / "s" / "subagents" / "agent-a1.jsonl"
    copy.parent.mkdir(parents=True)
    shutil.copy(SUB, copy)
    assert load(copy).records[0].feature == "scratch"


def test_directory_includes_subagents_and_deduplicates_across_files():
    result = load(DIR)
    assert result.file_count == 3  # the .meta.json sidecar is not read
    assert sorted(r.request_id for r in result.records) == [
        f"req_synthetic_{n:03d}" for n in (1, 2, 3, 4)
    ]
    assert {r.feature for r in result.records} == {"demo-app", "other-tool"}
    repeated = next(r for r in result.records if r.request_id == "req_synthetic_002")
    assert repeated.timestamp == datetime(2026, 10, 1, 9, 0, 10, tzinfo=UTC)


def test_highest_output_wins_with_the_earliest_timestamp(tmp_path):
    lines = [
        _line(out=1, timestamp="2026-10-01T09:00:01Z"),
        _line(out=90, timestamp="2026-10-01T09:00:09Z"),
        _line(out=30, timestamp="2026-10-01T09:00:05Z"),
    ]
    (record,) = load(_write(tmp_path / "s.jsonl", lines)).records
    assert record.output_tokens == 90
    assert record.timestamp == datetime(2026, 10, 1, 9, 0, 1, tzinfo=UTC)


def test_same_message_with_different_request_ids_is_two_responses(tmp_path):
    path = _write(tmp_path / "s.jsonl", [_line(rid="req_a"), _line(rid="req_b")])
    assert len(load(path).records) == 2


@pytest.mark.parametrize("order", [1, -1])
def test_a_missing_request_id_falls_back_to_the_message_id(tmp_path, order):
    lines = [_line(rid="req_1"), _line(rid=None, out=8)][::order]
    (record,) = load(_write(tmp_path / "s.jsonl", lines)).records
    assert record.output_tokens == 8
    alone = _write(tmp_path / "t.jsonl", [_line(rid=None), _line(rid=None)])
    assert [r.request_id for r in load(alone).records] == ["msg_1"]


def test_lines_without_any_id_are_counted_with_a_warning(tmp_path):
    path = _write(tmp_path / "s.jsonl", [_line(mid=None, rid=None)] * 2)
    result = load(path, strict=True)  # a warning, not an invalid row
    assert len(result.records) == 2
    assert [i.location for i in result.issues] == ["line 1", "line 2"]
    assert "cannot deduplicate" in result.issues[0].message


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ('{"type": "assistant", "mess', "invalid JSON"),
        ({"type": "assistant", "sessionId": "s"}, "no message"),
        ({"type": "assistant", "message": {"model": "claude-sonnet-5-5"}}, "no token usage"),
        (_line(out=-1), "output_tokens"),
        (
            _line(
                message={
                    "model": "claude-sonnet-5-5",
                    "usage": {"input_tokens": "many", "output_tokens": 1},
                }
            ),
            "input_tokens",
        ),
        (
            _line(
                message={
                    "model": "claude-sonnet-5-5",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_creation": {"ephemeral_5m_input_tokens": "x"},
                    },
                }
            ),
            "cache_creation",
        ),
        (
            _line(
                message={
                    "model": "claude-sonnet-5-5",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "inference_geo": "mars"},
                }
            ),
            "mars",
        ),
    ],
)
def test_invalid_assistant_lines_warn_or_fail_under_strict(tmp_path, bad, message):
    path = _write(tmp_path / "s.jsonl", [_line(mid="ok", rid="ok"), bad])
    result = load(path)
    assert len(result.records) == 1
    (issue,) = result.issues
    assert issue.location == "line 2" and message in issue.message
    with pytest.raises(IngestError, match="line 2"):
        load(path, strict=True)


def test_unknown_models_are_skipped_with_one_warning_even_under_strict(tmp_path):
    lines = [_line(mid="a", rid="a", model="claude-next")] * 2
    lines += [_line(mid="b", rid="b", model="claude-next"), _line()]
    result = load(_write(tmp_path / "s.jsonl", lines), strict=True)
    assert [r.model for r in result.records] == ["claude-sonnet-5-5"]
    (issue,) = result.issues
    assert "skipped 2 responses from unknown model 'claude-next'" in issue.message


def test_each_unkeyed_line_from_an_unknown_model_is_counted(tmp_path):
    lines = [_line(mid=None, rid=None, model="claude-next")] * 3
    (issue,) = load(_write(tmp_path / "s.jsonl", lines)).issues
    assert "skipped 3 responses" in issue.message


def test_undecodable_bytes_cost_one_line_not_the_file(tmp_path):
    path = _write(tmp_path / "s.jsonl", [_line()])
    path.write_bytes(path.read_bytes() + b'{"type": "assist\xe2')
    result = load(path)
    assert len(result.records) == 1 and "invalid JSON" in result.issues[0].message


def test_format_override(tmp_path):
    path = _write(tmp_path / "log.txt", [_line()])
    assert len(load(path, fmt="claude-code").records) == 1


@pytest.mark.parametrize(
    ("cwd", "name"),
    [
        ("C:\\work\\demo-app", "demo-app"),
        ("/srv/demo-app/", "demo-app"),
        ("C:\\work\\demo-app\\.claude\\worktrees\\fix", "demo-app"),
        ("/srv/demo-app/.claude/worktrees/fix/src", "demo-app"),
        ("C:\\Users\\someone", "(home)"),
        ("/home/someone", "(home)"),
        ("/Users/someone/tool", "tool"),
        ("C:\\", None),
        ("/", None),
        (None, None),
    ],
)
def test_project_name(cwd, name):
    assert project_name(cwd) == name
