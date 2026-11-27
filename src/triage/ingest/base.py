"""Adapter base class, result models and shared helpers for usage ingestion."""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from triage.models import UsageRecord

if TYPE_CHECKING:
    from triage.ingest.paths import InputSet

RECORD_FIELDS = frozenset(UsageRecord.model_fields) | {"cache_creation_input_tokens"}
LINE_SUFFIXES = frozenset({".jsonl", ".ndjson"})  # one JSON object per line


class IngestError(ValueError):
    """Raised for unreadable input, or for the first bad row when strict=True."""


class IngestIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    location: str | None = None  # e.g. "line 12", "row 4", "data[2].results[0]"
    message: str

    def under(self, label: str) -> IngestIssue:
        """This issue, located inside the file `label`."""
        where = f"{label}: {self.location}" if self.location else label
        return self.model_copy(update={"location": where})


class IngestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str  # the file, folder or pattern name that was loaded; never a full path
    source_format: str
    file_count: int = 1
    records: list[UsageRecord] = Field(default_factory=list)
    issues: list[IngestIssue] = Field(default_factory=list)

    @property
    def aggregated(self) -> bool:
        return any(r.is_aggregated for r in self.records)

    @property
    def time_range(self) -> tuple[datetime, datetime] | None:
        stamps = [t for r in self.records for t in (r.timestamp, r.period_end) if t is not None]
        return (min(stamps), max(stamps)) if stamps else None


class IngestAdapter:
    """Reads one usage format. Subclasses implement `detect` and `read`."""

    name: str

    def detect(self, path: Path, head: str) -> bool:
        """Whether this adapter can read `path`; `head` is the file's first few KB."""
        raise NotImplementedError

    def read(self, path: Path, *, strict: bool = False) -> IngestResult:
        raise NotImplementedError

    def read_set(self, inputs: InputSet, *, strict: bool = False) -> IngestResult:
        """Read every file in `inputs` as one result. Override to combine across files."""

        def read_one(path: Path) -> IngestResult:
            return self.read(path, strict=strict)

        return read_each(self.name, inputs, read_one, strict=strict)


_READ_ERRORS = (OSError, UnicodeError, csv.Error)


def skip_reason(exc: Exception) -> str:
    return str(exc) if isinstance(exc, IngestError) else f"unreadable ({type(exc).__name__})"


def skipped_file(label: str, reason: str, *, strict: bool) -> IngestIssue:
    """The warning for one of several files that can't be read; raises under `strict`."""
    if strict:
        raise IngestError(f"{label}: {reason}")
    return IngestIssue(location=label, message=f"skipped file: {reason}")


def read_each(
    name: str,
    inputs: InputSet,
    read_one: Callable[[Path], IngestResult],
    *,
    strict: bool,
) -> IngestResult:
    """Read each file with `read_one` under the shared policy for file sets.

    A file named on its own fails loudly. One of several that can't be read is skipped with a
    warning (an error under `strict`), and every issue is labelled with the file it came from.
    `file_count` counts only the files that were read.
    """
    if inputs.single:
        (path,) = inputs.files
        try:
            result = read_one(path)
        except _READ_ERRORS as exc:
            raise IngestError(f"can't read {path.name}: {skip_reason(exc)}") from exc
        return result.model_copy(update={"source": inputs.label})
    records: list[UsageRecord] = []
    issues: list[IngestIssue] = []
    read = 0
    for path in inputs.files:
        label = inputs.display(path)
        try:
            result = read_one(path)
        except (IngestError, *_READ_ERRORS) as exc:
            issues.append(skipped_file(label, skip_reason(exc), strict=strict))
            continue
        read += 1
        records.extend(result.records)
        issues.extend(issue.under(label) for issue in result.issues)
    return IngestResult(
        source=inputs.label, source_format=name, file_count=read, records=records, issues=issues
    )


class RowCollector:
    """Validates mapped rows into UsageRecords, collecting or raising issues."""

    def __init__(self, source_format: str, *, source: str, strict: bool) -> None:
        self.source_format = source_format
        self.source = source
        self.strict = strict
        self.records: list[UsageRecord] = []
        self.issues: list[IngestIssue] = []
        self._ignored: set[str] = set()

    def issue(self, location: str | None, message: str) -> None:
        if self.strict:
            raise IngestError(f"{location}: {message}" if location else message)
        self.issues.append(IngestIssue(location=location, message=message))

    def warn(self, location: str | None, message: str) -> None:
        """An issue that never fails the read, even under `strict`."""
        self.issues.append(IngestIssue(location=location, message=message))

    def ignore(self, keys: Iterable[str]) -> None:
        self._ignored.update(keys)

    def add(self, location: str, row: Mapping[str, Any]) -> None:
        try:
            self.records.append(UsageRecord.model_validate(row))
        except ValidationError as exc:
            self.issue(location, _summarise(exc))

    def result(self) -> IngestResult:
        issues = list(self.issues)
        if self._ignored:
            names = ", ".join(sorted(self._ignored))
            issues.append(IngestIssue(message=f"ignored unrecognised fields: {names}"))
        return IngestResult(
            source=self.source,
            source_format=self.source_format,
            records=self.records,
            issues=issues,
        )


def _summarise(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "record"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def flatten(obj: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into dotted keys: {"a": {"b": 1}} -> {"a.b": 1}."""
    out: dict[str, Any] = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out
