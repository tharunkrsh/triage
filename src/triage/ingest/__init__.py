"""Usage ingestion: pick an adapter per input file and normalise it into UsageRecords."""

from __future__ import annotations

import csv
from pathlib import Path

from triage.ingest.anthropic_csv import ConsoleCsvAdapter, ConsoleJsonAdapter
from triage.ingest.base import IngestAdapter, IngestError, IngestIssue, IngestResult
from triage.ingest.claude_code import ClaudeCodeAdapter
from triage.ingest.jsonl import JsonlAdapter
from triage.ingest.paths import InputSet, resolve_inputs

_CLAUDE_CODE = ClaudeCodeAdapter()
# Detection order matters: session logs can look like Console JSON (a "data" key) or plain
# JSONL, and Console JSON is checked before JSONL sniffing.
ADAPTERS: dict[str, IngestAdapter] = {
    a.name: a for a in (ConsoleCsvAdapter(), _CLAUDE_CODE, ConsoleJsonAdapter(), JsonlAdapter())
}
_CONSOLE_FORMATS = frozenset({ConsoleCsvAdapter.name, ConsoleJsonAdapter.name})
_HEAD_BYTES = 4096

__all__ = [
    "ADAPTERS",
    "IngestError",
    "IngestIssue",
    "IngestResult",
    "InputSet",
    "detect_format",
    "load",
    "resolve_inputs",
]


def detect_format(path: Path) -> str:
    with path.open(encoding="utf-8-sig", errors="replace") as fh:
        head = fh.read(_HEAD_BYTES)
    for name, adapter in ADAPTERS.items():
        if adapter.detect(path, head):
            return name
    raise IngestError(f"can't tell the format of {path.name}; pass --format")


def load(path: Path | str, *, fmt: str | None = None, strict: bool = False) -> IngestResult:
    """Read a file, directory or glob. `fmt` is an ADAPTERS key, or None to auto-detect."""
    if fmt is not None and fmt not in ADAPTERS:
        raise IngestError(f"unknown format {fmt!r}; choose from {', '.join(ADAPTERS)}")
    inputs = resolve_inputs(path)
    if inputs.single:
        file = inputs.files[0]
        return ADAPTERS[fmt or detect_format(file)].read(file, strict=strict)
    return _load_many(inputs, fmt=fmt, strict=strict)


def _load_many(inputs: InputSet, *, fmt: str | None, strict: bool) -> IngestResult:
    """Combine several files; one that can't be read is a warning unless `strict`."""
    results: list[IngestResult] = []
    issues: list[IngestIssue] = []
    sessions: list[tuple[Path, str | None]] = []  # read together to deduplicate across files
    skipped = 0

    def skip(label: str, reason: str) -> None:
        nonlocal skipped
        if strict:
            raise IngestError(f"{label}: {reason}")
        skipped += 1
        issues.append(IngestIssue(location=label, message=f"skipped file: {reason}"))

    for file in inputs.files:
        label = inputs.display(file)
        try:
            name = fmt or detect_format(file)
            if name == _CLAUDE_CODE.name:
                sessions.append((file, label))
                continue
            result = ADAPTERS[name].read(file, strict=strict)
        except IngestError as exc:
            skip(label, str(exc))
            continue
        except (OSError, UnicodeError, csv.Error) as exc:
            skip(label, f"unreadable ({type(exc).__name__})")
            continue
        results.append(result)
        for issue in result.issues:
            where = f"{label}: {issue.location}" if issue.location else label
            issues.append(issue.model_copy(update={"location": where}))
    if sessions:
        combined = _CLAUDE_CODE.read_files(sessions, strict=strict)
        results.append(combined)
        issues.extend(combined.issues)

    formats = {r.source_format for r in results}
    if not formats:
        first = f" ({issues[0].location}: {issues[0].message})" if issues else ""
        raise IngestError(f"none of the {len(inputs.files)} files could be read{first}")
    if formats & _CONSOLE_FORMATS and formats - _CONSOLE_FORMATS:
        raise IngestError(
            "can't combine Console usage exports with per-request logs: the same spend would "
            "be counted twice. Narrow the path, for example with a glob."
        )
    return IngestResult(
        source_format=", ".join(sorted(formats)),
        file_count=len(inputs.files) - skipped,
        records=[record for r in results for record in r.records],
        issues=issues,
    )
