"""Milestone 4A: deterministic JSON / JSON Schema / single-step tool-call checks."""

import pytest

from triage.simulate.eval_det import (
    InvalidSchemaError,
    check_json_parse,
    check_schema,
    check_tool_calls,
    evaluate,
    extract_json,
    validate_schema,
)
from triage.simulate.models import ReplayTask, ResponseContent, ToolCall

PERSON = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "address": {"type": "object", "properties": {"zip": {"type": "string"}}},
    },
}

# --------------------------------------------------------------------------- JSON parsing


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ("  \n[1, 2]\n ", [1, 2]),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ("```\n[true]\n```", [True]),
        ('```JSON\n{"a": null}\n```\n', {"a": None}),
        ("42", 42),
    ],
)
def test_extract_json_valid(text, expected):
    value, error = extract_json(text)
    assert error is None
    assert value == expected


@pytest.mark.parametrize(
    "text",
    ["", "not json", '{"a": 1', 'Sure! Here it is: {"a": 1}', '{"a": 1} hope that helps'],
)
def test_extract_json_invalid(text):
    value, error = extract_json(text)
    assert value is None
    assert isinstance(error, str) and error


def test_extract_json_distinguishes_null_from_failure():
    assert extract_json("null") == (None, None)


def test_check_json_parse():
    ok, bad = check_json_parse('{"a": 1}'), check_json_parse("{oops")
    assert (ok.name, ok.passed) == ("json_parse", True)
    assert (bad.name, bad.passed) == ("json_parse", False)
    assert bad.detail


# --------------------------------------------------------------------------- JSON Schema


def test_schema_pass():
    result = check_schema({"name": "Ada", "age": 36}, PERSON)
    assert (result.name, result.passed) == ("json_schema", True)


def test_schema_missing_required():
    result = check_schema({"name": "Ada"}, PERSON)
    assert not result.passed
    assert "age" in result.detail


def test_schema_wrong_type():
    result = check_schema({"name": "Ada", "age": "36"}, PERSON)
    assert not result.passed
    assert "age" in result.detail


def test_schema_nested_error_reports_path():
    result = check_schema({"name": "Ada", "age": 36, "address": {"zip": 12345}}, PERSON)
    assert not result.passed
    assert "address" in result.detail and "zip" in result.detail


def test_validate_schema_accepts_valid_and_rejects_invalid():
    validate_schema(PERSON)
    with pytest.raises(InvalidSchemaError):
        validate_schema({"type": 12})
    assert issubclass(InvalidSchemaError, ValueError)


# --------------------------------------------------------------------------- tool calls


def _calls(*pairs):
    return ResponseContent(tool_calls=[ToolCall(name=n, input=i) for n, i in pairs])


BASE = _calls(("get_weather", {"city": "Paris", "unit": "c"}))


def test_tool_call_exact_match():
    result = check_tool_calls(_calls(("get_weather", {"city": "Paris", "unit": "c"})), BASE)
    assert (result.name, result.passed) == ("tool_call", True)


def test_tool_call_key_order_irrelevant():
    assert check_tool_calls(_calls(("get_weather", {"unit": "c", "city": "Paris"})), BASE).passed


def test_tool_call_int_float_equal():
    base = _calls(("set_temp", {"value": 1}))
    assert check_tool_calls(_calls(("set_temp", {"value": 1.0})), base).passed


def test_tool_call_nested_args():
    base = _calls(("book", {"guests": [{"name": "a"}, {"name": "b"}], "opts": {"x": True}}))
    same = _calls(("book", {"opts": {"x": True}, "guests": [{"name": "a"}, {"name": "b"}]}))
    reordered_list = _calls(
        ("book", {"opts": {"x": True}, "guests": [{"name": "b"}, {"name": "a"}]})
    )
    assert check_tool_calls(same, base).passed
    assert not check_tool_calls(reordered_list, base).passed  # list order inside args matters


def test_tool_call_wrong_name():
    result = check_tool_calls(_calls(("get_forecast", {"city": "Paris", "unit": "c"})), BASE)
    assert not result.passed
    assert "get_forecast" in result.detail or "get_weather" in result.detail


def test_tool_call_wrong_args():
    result = check_tool_calls(_calls(("get_weather", {"city": "Lyon", "unit": "c"})), BASE)
    assert not result.passed
    assert result.detail


def test_tool_call_missing_call():
    result = check_tool_calls(ResponseContent(text="It is sunny."), BASE)
    assert not result.passed


def test_tool_call_extra_call_fails():
    cand = _calls(("get_weather", {"city": "Paris", "unit": "c"}), ("get_time", {}))
    assert not check_tool_calls(cand, BASE).passed


def test_parallel_tool_calls_order_insensitive():
    base = _calls(("a", {"x": 1}), ("b", {"y": 2}))
    assert check_tool_calls(_calls(("b", {"y": 2}), ("a", {"x": 1})), base).passed


def test_duplicate_tool_calls_counted():
    base = _calls(("a", {"x": 1}), ("a", {"x": 1}))
    assert not check_tool_calls(_calls(("a", {"x": 1})), base).passed


# --------------------------------------------------------------------------- evaluate()


def _task(baseline, **kw):
    return ReplayTask(
        task_id="t",
        baseline_model="claude-opus-5-5",
        messages=[{"role": "user", "content": "hi"}],
        baseline_response=baseline,
        **kw,
    )


def test_evaluate_tool_task_only_checks_tool_calls():
    result = evaluate(
        _task(BASE, expected_schema=PERSON), _calls(("get_weather", {"city": "Paris", "unit": "c"}))
    )
    assert [c.name for c in result.checks] == ["tool_call"]
    assert result.applicable and result.passed is True


def test_evaluate_schema_task_pass():
    task = _task(ResponseContent(text='{"name": "Ada", "age": 36}'), expected_schema=PERSON)
    result = evaluate(task, ResponseContent(text='```json\n{"name": "Bob", "age": 7}\n```'))
    assert [c.name for c in result.checks] == ["json_parse", "json_schema"]
    assert result.passed is True


def test_evaluate_schema_task_unparseable_fails_both():
    task = _task(ResponseContent(text='{"name": "Ada", "age": 36}'), expected_schema=PERSON)
    result = evaluate(task, ResponseContent(text="Ada is 36."))
    assert [(c.name, c.passed) for c in result.checks] == [
        ("json_parse", False),
        ("json_schema", False),
    ]
    assert result.passed is False


def test_evaluate_schema_violation():
    task = _task(ResponseContent(text="{}"), expected_schema=PERSON)
    result = evaluate(task, ResponseContent(text='{"name": "Bob"}'))
    assert result.passed is False


def test_evaluate_json_baseline_without_schema_checks_parse_only():
    task = _task(ResponseContent(text='{"ok": true}'))
    result = evaluate(task, ResponseContent(text='{"ok": false}'))
    assert [c.name for c in result.checks] == ["json_parse"]
    assert result.passed is True  # semantics are the judge's job


def test_evaluate_plain_text_task_has_no_checks():
    result = evaluate(_task(ResponseContent(text="A summary.")), ResponseContent(text="Other."))
    assert result.checks == []
    assert not result.applicable
    assert result.passed is None
