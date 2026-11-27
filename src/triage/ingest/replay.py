"""Load replay tasks from a file, directory or glob of canonical JSONL."""

from __future__ import annotations

from pathlib import Path

from triage.ingest.base import LINE_SUFFIXES, IngestError, IngestIssue, skip_reason, skipped_file
from triage.ingest.claude_code import ClaudeCodeAdapter
from triage.ingest.jsonl import ReplayIngestResult, read_replays
from triage.ingest.paths import InputSet, resolve_inputs
from triage.simulate.models import ReplayTask

_TRANSCRIPTS = ClaudeCodeAdapter()
_TRANSCRIPT_REASON = "Claude Code transcripts can't be replayed: they are multi-turn sessions"


def _read_one(inputs: InputSet, file: Path) -> ReplayIngestResult:
    """Read one of several files, labelling its issues and generated ids with its path."""
    label = inputs.display(file)
    if file.suffix.lower() not in LINE_SUFFIXES:
        skipped = skipped_file(label, "not JSONL", strict=False)
        return ReplayIngestResult(source=label, issues=[skipped])
    result = read_replays(file, id_prefix=f"{label}:")
    return result.model_copy(update={"issues": [i.under(label) for i in result.issues]})


def load_replays(spec: Path | str) -> ReplayIngestResult:
    """Collect replayable tasks. Transcripts and non-JSONL files are skipped with a warning.

    With several files, generated task ids are prefixed with the file's relative path.
    Explicit ids are kept, so duplicates are rejected later by the batch id check.
    """
    inputs = resolve_inputs(spec)
    tasks: list[ReplayTask] = []
    issues: list[IngestIssue] = []
    transcripts = 0
    for file in inputs.files:
        try:
            if _TRANSCRIPTS.detect(file, ""):
                transcripts += 1
                continue
            result = read_replays(file) if inputs.single else _read_one(inputs, file)
        except (OSError, UnicodeError) as exc:
            if inputs.single:
                raise IngestError(f"can't read {file.name}: {skip_reason(exc)}") from exc
            issues.append(skipped_file(inputs.display(file), skip_reason(exc), strict=False))
            continue
        tasks.extend(result.tasks)
        issues.extend(result.issues)
    if transcripts:
        noun = "file" if transcripts == 1 else "files"
        message = f"skipped {transcripts:,} {noun}. {_TRANSCRIPT_REASON}"
        issues.insert(0, IngestIssue(message=message))
    return ReplayIngestResult(source=inputs.label, tasks=tasks, issues=issues)
