"""Claude Code session logs: `~/.claude/projects/<project>/<session>.jsonl`.

Subagent logs live in `<project>/<session>/subagents/agent-<id>.jsonl`. Only `assistant`
lines carry usage. One API response is written as several lines (one per content block), each
repeating the same usage, so lines are deduplicated by (message.id, requestId) across every
input file. Prompt and response text is never read into a record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from triage.ingest.base import (
    LINE_SUFFIXES,
    IngestAdapter,
    IngestIssue,
    IngestResult,
    RowCollector,
    _summarise,
    read_each,
)
from triage.ingest.jsonl import from_usage, iter_objects
from triage.ingest.paths import InputSet
from triage.models import UsageRecord
from triage.pricing import PricingTable, UnknownModelError

NAME = "claude-code"
PRODUCT = "claude-code"
SUBAGENT_PRODUCT = "claude-code:subagent"
SYNTHETIC_MODEL = "<synthetic>"
_SUBAGENT_DIR = "subagents"
_DETECT_LINES = 20
_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_5m_input_tokens",
    "cache_creation_1h_input_tokens",
)
_CANONICAL_KEYS = frozenset({"model", "usage", "input_tokens", "messages"})

_Key = tuple[str, str | None]


def _is_session_event(obj: Any) -> bool:
    if not isinstance(obj, dict) or not isinstance(obj.get("type"), str):
        return False
    if _CANONICAL_KEYS & obj.keys():
        return False
    return bool({"sessionId", "parentUuid", "leafUuid", "snapshot"} & obj.keys())


def _first_cwd(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8-sig", errors="replace") as fh:
            for line in fh:
                if '"cwd"' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and isinstance(obj.get("cwd"), str) and obj["cwd"]:
                    return obj["cwd"]
    except OSError:
        return None
    return None


def project_name(cwd: str | None) -> str | None:
    """The folder name of `cwd`, with Claude Code worktrees mapped to their repository.

    A home directory is reported as "(home)" so that a user name never becomes a feature.
    """
    parts = [p for p in re.split(r"[\\/]+", cwd or "") if p]
    for index in range(1, len(parts) - 1):
        if parts[index] == ".claude" and parts[index + 1] == "worktrees":
            parts = parts[:index]
            break
    if not parts or parts[-1].endswith(":"):
        return None
    if len(parts) >= 2 and parts[-2].lower() in {"users", "home"}:
        return "(home)"
    return parts[-1]


def _session_project(path: Path) -> str | None:
    """A subagent log belongs to its parent session's project when that log is present."""
    for folder in path.parents:
        if folder.name == _SUBAGENT_DIR:
            session = folder.parent
            parent_log = session.with_name(f"{session.name}.jsonl")
            if (cwd := _first_cwd(parent_log)) is not None:
                return project_name(cwd)
            break
    return project_name(_first_cwd(path))


def _merge(kept: UsageRecord, new: UsageRecord) -> UsageRecord:
    """One record per response: the usage with most output, stamped with the earliest time."""
    best = new if new.output_tokens > kept.output_tokens else kept
    stamps = [r.timestamp for r in (kept, new) if r.timestamp is not None]
    return best.model_copy(update={"timestamp": min(stamps) if stamps else None})


class _Reader:
    """Deduplicates responses across every file of one set; issues are reported per file."""

    def __init__(self, *, strict: bool) -> None:
        self.strict = strict
        self.pricing = PricingTable.load()
        self.responses: dict[_Key, UsageRecord] = {}
        self.unkeyed: list[UsageRecord] = []
        self.unknown_models: dict[str, set[object]] = {}

    def read_file(self, path: Path) -> IngestResult:
        """Fold one log into the set; the result carries only this file's issues."""
        rows = RowCollector(NAME, source=path.name, strict=self.strict)
        project = _session_project(path)
        subagent = _SUBAGENT_DIR in path.parts
        for location, obj in iter_objects(path, rows.issue, errors="replace"):
            if obj.get("type") != "assistant":
                continue
            message = obj.get("message")
            if not isinstance(message, dict):
                rows.issue(location, "assistant line has no message")
                continue
            if message.get("model") == SYNTHETIC_MODEL:
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict) or not {"input_tokens", "output_tokens"} <= set(usage):
                rows.issue(location, "assistant line has no token usage")
                continue
            message_id, request_id = message.get("id"), obj.get("requestId")
            sidechain = subagent or obj.get("isSidechain") is True
            try:
                record = UsageRecord.model_validate(
                    {
                        **from_usage(usage),
                        "model": message.get("model"),
                        "is_batch": usage.get("service_tier") == "batch",
                        "request_id": request_id or message_id,
                        "timestamp": obj.get("timestamp"),
                        "feature": project,
                        "product": SUBAGENT_PRODUCT if sidechain else PRODUCT,
                    }
                )
            except ValidationError as exc:
                rows.issue(location, _summarise(exc))
                continue
            except (TypeError, AttributeError):
                rows.issue(location, "malformed token usage")
                continue
            if not any(getattr(record, name) for name in _TOKEN_FIELDS):
                continue
            ident = message_id or request_id
            key = (str(ident), str(request_id) if request_id else None) if ident else None
            try:
                self.pricing.effective_rates(record)
            except UnknownModelError:
                self.unknown_models.setdefault(record.model, set()).add(key or object())
                continue
            except ValueError as exc:  # unknown inference_geo or unpriced fast mode
                rows.issue(location, str(exc))
                continue
            if key is None:
                rows.warn(location, "no message.id or requestId; cannot deduplicate")
                self.unkeyed.append(record)
            elif key in self.responses:
                self.responses[key] = _merge(self.responses[key], record)
            else:
                self.responses[key] = record
        return rows.result()

    def _fold_missing_request_ids(self) -> None:
        """A line lacking requestId belongs to the response that shares its message.id."""
        identified = {mid: (mid, rid) for mid, rid in self.responses if rid is not None}
        for mid, rid in list(self.responses):
            if rid is None and mid in identified:
                record = self.responses.pop((mid, None))
                target = identified[mid]
                self.responses[target] = _merge(self.responses[target], record)

    def result(self, read: IngestResult) -> IngestResult:
        """`read` (the per-file issues and counts) with the deduplicated records."""
        self._fold_missing_request_ids()
        skipped = [
            IngestIssue(
                message=f"skipped {len(seen):,} responses from unknown model {model!r}: "
                "add it to pricing.yaml"
            )
            for model, seen in sorted(self.unknown_models.items(), reverse=True)
        ]
        return read.model_copy(
            update={
                "records": [*self.responses.values(), *self.unkeyed],
                "issues": [*skipped, *read.issues],
            }
        )


class ClaudeCodeAdapter(IngestAdapter):
    name = NAME

    def detect(self, path: Path, head: str) -> bool:
        if path.suffix.lower() not in LINE_SUFFIXES:
            return False
        # `head` can end inside a long first line, so read whole lines from the file.
        with path.open(encoding="utf-8-sig", errors="replace") as fh:
            for _ in range(_DETECT_LINES):
                line = fh.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _is_session_event(obj):
                    return True
                if isinstance(obj, dict) and _CANONICAL_KEYS & obj.keys():
                    return False
        return False

    def read(self, path: Path, *, strict: bool = False) -> IngestResult:
        return self.read_set(InputSet(label=path.name, files=[path]), strict=strict)

    def read_set(self, inputs: InputSet, *, strict: bool = False) -> IngestResult:
        """Read the logs as one set, so a response split across files is counted once."""
        reader = _Reader(strict=strict)
        return reader.result(read_each(NAME, inputs, reader.read_file, strict=strict))
