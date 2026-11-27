"""Usage ingestion: pick an adapter per input file and normalise it into UsageRecords."""

from __future__ import annotations

from pathlib import Path

from triage.ingest.anthropic_csv import ConsoleCsvAdapter, ConsoleJsonAdapter
from triage.ingest.base import (
    IngestAdapter,
    IngestError,
    IngestIssue,
    IngestResult,
    skip_reason,
    skipped_file,
)
from triage.ingest.claude_code import ClaudeCodeAdapter
from triage.ingest.jsonl import JsonlAdapter
from triage.ingest.paths import InputSet, resolve_inputs

# Detection order matters: session logs can look like plain JSONL, so they are checked first.
ADAPTERS: dict[str, IngestAdapter] = {
    a.name: a
    for a in (ConsoleCsvAdapter(), ClaudeCodeAdapter(), ConsoleJsonAdapter(), JsonlAdapter())
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
        return ADAPTERS[fmt or detect_format(inputs.files[0])].read_set(inputs, strict=strict)

    groups: dict[str, list[Path]] = {}
    undetected: list[IngestIssue] = []
    for file in inputs.files:
        try:
            groups.setdefault(fmt or detect_format(file), []).append(file)
        except (IngestError, OSError) as exc:
            reason = skip_reason(exc)
            undetected.append(skipped_file(inputs.display(file), reason, strict=strict))
    results = [
        ADAPTERS[name].read_set(inputs.subset(files), strict=strict)
        for name, files in groups.items()
    ]
    return _combine(inputs, results, undetected)


def _combine(
    inputs: InputSet, results: list[IngestResult], undetected: list[IngestIssue]
) -> IngestResult:
    issues = [*undetected, *(issue for r in results for issue in r.issues)]
    formats = {r.source_format for r in results if r.file_count}
    if not formats:
        first = f" ({issues[0].location}: {issues[0].message})" if issues else ""
        raise IngestError(f"none of the {len(inputs.files)} files could be read{first}")
    if formats & _CONSOLE_FORMATS and formats - _CONSOLE_FORMATS:
        raise IngestError(
            "can't combine Console usage exports with per-request logs: the same spend would "
            "be counted twice. Narrow the path, for example with a glob."
        )
    return IngestResult(
        source=inputs.label,
        source_format=", ".join(sorted(formats)),
        file_count=sum(r.file_count for r in results),
        records=[record for r in results for record in r.records],
        issues=issues,
    )
