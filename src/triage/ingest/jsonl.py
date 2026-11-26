"""Canonical JSONL reader: one request per line.

Each line is either flat (UsageRecord field names at the top level) or carries an API-style
`usage` object as returned by the Messages API. Simulator-only keys (prompt messages,
baseline response, schemas) are ignored by the audit reader and consumed by `read_replays`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from triage.ingest.base import (
    RECORD_FIELDS,
    IngestError,
    IngestIssue,
    IngestResult,
    RowCollector,
    _summarise,
)
from triage.models import UsageRecord
from triage.simulate.eval_det import validate_schema
from triage.simulate.models import ReplayTask, ResponseContent

NAME = "jsonl"

# Keys the simulator's canonical JSONL carries that have no meaning for the audit.
_SIMULATE_KEYS = frozenset(
    {
        "messages",
        "response",
        "baseline",
        "baseline_model",
        "baseline_response",
        "expected_schema",
        "tools",
        "tool_choice",
        "temperature",
        "task_id",
        "prompt_version",
        "metadata",
    }
)
_USAGE_TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens")
_TOKEN_FIELDS = (
    *_USAGE_TOKEN_KEYS,
    "cache_creation_input_tokens",
    "cache_creation_5m_input_tokens",
    "cache_creation_1h_input_tokens",
)


def _hash_system(system: Any) -> str:
    text = system if isinstance(system, str) else json.dumps(system, sort_keys=True)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _from_usage(usage: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: usage[k] for k in _USAGE_TOKEN_KEYS if usage.get(k) is not None}
    split = usage.get("cache_creation")
    if isinstance(split, dict):
        out["cache_creation_5m_input_tokens"] = split.get("ephemeral_5m_input_tokens") or 0
        out["cache_creation_1h_input_tokens"] = split.get("ephemeral_1h_input_tokens") or 0
    elif usage.get("cache_creation_input_tokens") is not None:
        out["cache_creation_input_tokens"] = usage["cache_creation_input_tokens"]
    for key in ("service_tier", "inference_geo", "speed"):
        if usage.get(key) is not None:
            out[key] = usage[key]
    return out


def map_line(obj: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Map one JSON object to UsageRecord input; also return unrecognised keys."""
    obj = dict(obj)
    row: dict[str, Any] = {}
    usage = obj.pop("usage", None)
    if isinstance(usage, dict):
        row.update(_from_usage(usage))
    system = obj.pop("system", None)
    if system is not None and "system_prompt_hash" not in obj:
        row["system_prompt_hash"] = _hash_system(system)
    if "id" in obj and "request_id" not in obj:
        row["request_id"] = obj.pop("id")

    unknown = set()
    for key, value in obj.items():
        if key in RECORD_FIELDS:
            row[key] = value  # explicit top-level fields win over `usage`
        elif key not in _SIMULATE_KEYS:
            unknown.add(key)
    if row.get("service_tier") == "batch" and "is_batch" not in row:
        row["is_batch"] = True
    return row, unknown


def _iter_objects(
    path: Path, issue: Callable[[str, str], None]
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (location, object) per non-blank line, reporting unparseable lines via `issue`."""
    with path.open(encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            loc = f"line {lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                issue(loc, f"invalid JSON: {exc.msg}")
                continue
            if not isinstance(obj, dict):
                issue(loc, "expected a JSON object")
                continue
            yield loc, obj


class JsonlAdapter:
    name = NAME

    def detect(self, path: Path, head: str) -> bool:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            return True
        first = next((ln for ln in head.splitlines() if ln.strip()), "")
        try:
            return isinstance(json.loads(first), dict) and '"data"' not in first
        except json.JSONDecodeError:
            return False

    def read(self, path: Path, *, strict: bool = False) -> IngestResult:
        rows = RowCollector(NAME, strict=strict)
        for loc, obj in _iter_objects(path, rows.issue):
            row, unknown = map_line(obj)
            rows.ignore(unknown)
            rows.add(loc, row)
        return rows.result()


# --------------------------------------------------------------------------- replay loader


class ReplayIngestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tasks: list[ReplayTask] = Field(default_factory=list)
    issues: list[IngestIssue] = Field(default_factory=list)


def _map_replay(obj: dict[str, Any], loc: str) -> tuple[ReplayTask, set[str]]:
    baseline_model = obj.get("baseline_model") or obj.get("model")
    if not isinstance(baseline_model, str) or not baseline_model:
        raise ValueError("missing baseline_model (or model)")
    raw = obj["baseline_response"] if "baseline_response" in obj else obj.get("response")
    if raw is None:
        raise ValueError("missing baseline_response (or response)")
    schema = obj.get("expected_schema")
    if schema is not None:
        if not isinstance(schema, dict):
            raise ValueError("expected_schema must be a JSON object")
        validate_schema(schema)

    row, unknown = map_line({k: v for k, v in obj.items() if k not in ("model", "baseline_model")})
    usage = None
    if any(k in row for k in _TOKEN_FIELDS):
        usage = UsageRecord.model_validate({**row, "model": baseline_model})

    task_id = obj.get("task_id") or obj.get("request_id") or obj.get("id") or loc.replace(" ", "-")
    task = ReplayTask.model_validate(
        {
            "task_id": str(task_id),
            "feature": obj.get("feature"),
            "prompt_version": obj.get("prompt_version"),
            "system": obj.get("system"),
            "messages": obj.get("messages"),
            "tools": obj.get("tools"),
            "tool_choice": obj.get("tool_choice"),
            "expected_schema": schema,
            "baseline_model": baseline_model,
            "baseline_response": ResponseContent.from_api(raw),
            "baseline_usage": usage,
            "max_tokens": obj.get("max_tokens"),
            "temperature": obj.get("temperature"),
        }
    )
    return task, unknown


def map_replay(obj: dict[str, Any], *, loc: str) -> ReplayTask:
    """Map one canonical JSONL object to a ReplayTask; raises ValueError if it can't replay."""
    return _map_replay(obj, loc)[0]


def read_replays(path: Path | str, *, strict: bool = False) -> ReplayIngestResult:
    """Load replayable single-turn tasks; bad lines become issues (or raise when strict)."""
    tasks: list[ReplayTask] = []
    issues: list[IngestIssue] = []
    ignored: set[str] = set()

    def issue(loc: str | None, message: str) -> None:
        if strict:
            raise IngestError(f"{loc}: {message}" if loc else message)
        issues.append(IngestIssue(location=loc, message=message))

    for loc, obj in _iter_objects(Path(path), issue):
        try:
            task, unknown = _map_replay(obj, loc)
        except ValidationError as exc:
            issue(loc, _summarise(exc))
            continue
        except (ValueError, KeyError, TypeError) as exc:
            issue(loc, str(exc))
            continue
        tasks.append(task)
        ignored |= unknown
    if ignored:
        issue(None, f"ignored unrecognised fields: {', '.join(sorted(ignored))}")
    return ReplayIngestResult(tasks=tasks, issues=issues)
