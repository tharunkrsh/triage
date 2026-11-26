"""Deterministic checks: JSON parsing, JSON Schema validation, single-step tool-call match.

Tool calls are compared, never executed.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

from triage.simulate.models import (
    CheckResult,
    DeterministicResult,
    JsonValue,
    ReplayTask,
    ResponseContent,
    ToolCall,
)

_FENCE = re.compile(r"^```[\w-]*[ \t]*\n(?P<body>.*?)\n?```$", re.DOTALL)
_MAX_SCHEMA_ERRORS = 3


class InvalidSchemaError(ValueError):
    """Raised when an expected_schema is not itself a valid JSON Schema."""


# --------------------------------------------------------------------------- JSON


def extract_json(text: str) -> tuple[JsonValue | None, str | None]:
    """Parse `text` as JSON, allowing one surrounding ``` fence. Returns (value, error)."""
    body = text.strip()
    if fence := _FENCE.match(body):
        body = fence.group("body").strip()
    try:
        return json.loads(body), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"


def check_json_parse(text: str) -> CheckResult:
    _, error = extract_json(text)
    return CheckResult(name="json_parse", passed=error is None, detail=error)


# --------------------------------------------------------------------------- JSON Schema


def validate_schema(schema: Mapping[str, Any]) -> None:
    try:
        validator_for(schema).check_schema(schema)
    except SchemaError as exc:
        raise InvalidSchemaError(f"invalid expected_schema: {exc.message}") from exc


def _path(error: Any) -> str:
    return "/".join(str(p) for p in error.absolute_path) or "$"


def check_schema(value: JsonValue, schema: Mapping[str, Any]) -> CheckResult:
    validator = validator_for(schema)(schema)
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))
    if not errors:
        return CheckResult(name="json_schema", passed=True)
    detail = "; ".join(f"{_path(e)}: {e.message}" for e in errors[:_MAX_SCHEMA_ERRORS])
    return CheckResult(name="json_schema", passed=False, detail=detail)


# --------------------------------------------------------------------------- tool calls


def _normalise(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def _signature(call: ToolCall) -> str:
    return f"{call.name}({json.dumps(_normalise(call.input), sort_keys=True)})"


def check_tool_calls(candidate: ResponseContent, baseline: ResponseContent) -> CheckResult:
    """Multiset match of (name, arguments); order of parallel calls and keys is ignored."""
    want = Counter(_signature(c) for c in baseline.tool_calls)
    got = Counter(_signature(c) for c in candidate.tool_calls)
    if want == got:
        return CheckResult(name="tool_call", passed=True)
    problems = [f"missing call {sig}" for sig in (want - got).elements()]
    problems += [f"unexpected call {sig}" for sig in (got - want).elements()]
    return CheckResult(name="tool_call", passed=False, detail="; ".join(problems))


# --------------------------------------------------------------------------- per task


def evaluate(task: ReplayTask, candidate: ResponseContent) -> DeterministicResult:
    """Run the checks that apply to `task` (empty when only the judge can grade it)."""
    baseline = task.baseline_response
    if baseline.tool_calls:
        return DeterministicResult(checks=[check_tool_calls(candidate, baseline)])
    if task.expected_schema is not None:
        value, error = extract_json(candidate.text)
        parse = CheckResult(name="json_parse", passed=error is None, detail=error)
        if error is not None:
            schema = CheckResult(
                name="json_schema", passed=False, detail="skipped: response is not valid JSON"
            )
        else:
            schema = check_schema(value, task.expected_schema)
        return DeterministicResult(checks=[parse, schema])
    if extract_json(baseline.text)[1] is None:
        return DeterministicResult(checks=[check_json_parse(candidate.text)])
    return DeterministicResult()
