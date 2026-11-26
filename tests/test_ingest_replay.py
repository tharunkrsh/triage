"""Milestone 4A: canonical JSONL replay loader for `tokenize simulate`."""

import pytest

from tests.conftest import FIXTURES
from tokenize_cli.ingest import IngestError
from tokenize_cli.ingest.jsonl import ReplayIngestResult, map_line, map_replay, read_replays
from tokenize_cli.simulate.models import ReplayTask, ToolCall

REPLAY = FIXTURES / "simulate" / "replay.jsonl"


@pytest.fixture(scope="module")
def loaded() -> ReplayIngestResult:
    return read_replays(REPLAY)


def _by_id(result, task_id) -> ReplayTask:
    return next(t for t in result.tasks if t.task_id == task_id)


def test_fixture_counts(loaded):
    assert [t.task_id for t in loaded.tasks] == ["t-text", "t-json", "msg_tool", "t-long"]
    assert {i.location for i in loaded.issues} == {
        "line 6",
        "line 7",
        "line 8",
        "line 9",
        "line 10",
    }


def test_issue_messages_explain_rejection(loaded):
    by_loc = {i.location: i.message for i in loaded.issues}
    assert "messages" in by_loc["line 6"]
    assert "multi-turn" in by_loc["line 7"] or "tool_result" in by_loc["line 7"]
    assert "schema" in by_loc["line 8"].lower()
    assert "json" in by_loc["line 9"].lower()
    assert "baseline_response" in by_loc["line 10"]


def test_text_task_uses_model_fallback_and_flat_usage(loaded):
    t = _by_id(loaded, "t-text")
    assert t.baseline_model == "claude-opus-5-5"
    assert (t.feature, t.prompt_version) == ("summarise", "v1")
    assert t.system == "You write one-sentence summaries."
    assert t.baseline_response.text == "Revenue rose 12% this quarter."
    assert t.max_tokens == 1024 and t.temperature == 0.2
    u = t.baseline_usage
    assert u is not None and (u.model, u.input_tokens, u.output_tokens) == (
        "claude-opus-5-5",
        1200,
        80,
    )
    assert u.feature == "summarise"


def test_json_task_schema_and_usage_object(loaded):
    t = _by_id(loaded, "t-json")
    assert t.baseline_model == "claude-sonnet-4-6"
    assert t.expected_schema["required"] == ["name", "age"]
    assert t.baseline_usage.input_tokens == 900 and t.baseline_usage.output_tokens == 20
    assert t.messages[0]["content"][0]["type"] == "text"


def test_tool_task_id_fallback_and_full_message_baseline(loaded):
    t = _by_id(loaded, "msg_tool")
    assert t.tools[0]["name"] == "get_weather"
    assert t.baseline_response.tool_calls == [
        ToolCall(name="get_weather", input={"city": "Paris", "unit": "c"})
    ]
    assert t.baseline_response.stop_reason == "tool_use"
    assert t.prompt_version is None


def test_long_task_tokens_kept_for_tier_pricing(loaded):
    assert _by_id(loaded, "t-long").baseline_usage.prompt_tokens == 150000


def test_strict_raises_on_first_bad_line():
    with pytest.raises(IngestError, match="line 6"):
        read_replays(REPLAY, strict=True)


def test_map_replay_minimal_and_fallbacks():
    t = map_replay(
        {"model": "m", "messages": [{"role": "user", "content": "q"}], "response": "a"},
        loc="line 3",
    )
    assert t.task_id == "line-3"
    assert t.baseline_model == "m" and t.baseline_response.text == "a"
    assert t.baseline_usage is None  # no token counts logged


def test_map_replay_explicit_fields_win():
    t = map_replay(
        {
            "task_id": "x",
            "request_id": "r",
            "model": "ignored",
            "baseline_model": "claude-haiku-5-5",
            "messages": [{"role": "user", "content": "q"}],
            "baseline_response": "a",
            "response": "ignored",
        },
        loc="line 1",
    )
    assert (t.task_id, t.baseline_model, t.baseline_response.text) == ("x", "claude-haiku-5-5", "a")


def test_map_replay_request_id_fallback():
    t = map_replay(
        {
            "request_id": "req_9",
            "model": "m",
            "messages": [{"role": "user", "content": "q"}],
            "baseline_response": "a",
        },
        loc="line 1",
    )
    assert t.task_id == "req_9"


@pytest.mark.parametrize(
    "obj",
    [
        {"messages": [{"role": "user", "content": "q"}], "baseline_response": "a"},  # no model
        {"model": "m", "messages": "q", "baseline_response": "a"},  # messages not a list
        {"model": "m", "messages": [{"role": "user", "content": "q"}], "baseline_response": 5},
        {
            "model": "m",
            "messages": [{"role": "user", "content": "q"}],
            "baseline_response": "a",
            "expected_schema": {"type": "nope"},
        },
    ],
)
def test_map_replay_rejects_bad_rows(obj):
    with pytest.raises(ValueError):
        map_replay(obj, loc="line 1")


def test_audit_ignores_replay_only_keys():
    _, unknown = map_line(
        {
            "model": "m",
            "task_id": "t",
            "baseline_model": "x",
            "baseline_response": "y",
            "tool_choice": {"type": "auto"},
            "temperature": 0.1,
        }
    )
    assert unknown == set()
