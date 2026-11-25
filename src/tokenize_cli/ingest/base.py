"""Adapter protocol, result models and shared helpers for usage ingestion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tokenize_cli.models import UsageRecord

RECORD_FIELDS = frozenset(UsageRecord.model_fields) | {"cache_creation_input_tokens"}


class IngestError(ValueError):
    """Raised for unreadable input, or for the first bad row when strict=True."""


class IngestIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    location: str | None = None  # e.g. "line 12", "row 4", "data[2].results[0]"
    message: str


class IngestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_format: str
    records: list[UsageRecord] = Field(default_factory=list)
    issues: list[IngestIssue] = Field(default_factory=list)

    @property
    def aggregated(self) -> bool:
        return any(r.is_aggregated for r in self.records)

    @property
    def time_range(self) -> tuple[datetime, datetime] | None:
        stamps = [t for r in self.records for t in (r.timestamp, r.period_end) if t is not None]
        return (min(stamps), max(stamps)) if stamps else None


class IngestAdapter(Protocol):
    name: str

    def detect(self, path: Path, head: str) -> bool:
        """Whether this adapter can read `path`; `head` is the file's first few KB."""
        ...

    def read(self, path: Path, *, strict: bool = False) -> IngestResult: ...


class RowCollector:
    """Validates mapped rows into UsageRecords, collecting or raising issues."""

    def __init__(self, source_format: str, *, strict: bool) -> None:
        self.source_format = source_format
        self.strict = strict
        self.records: list[UsageRecord] = []
        self.issues: list[IngestIssue] = []
        self._ignored: set[str] = set()

    def issue(self, location: str | None, message: str) -> None:
        if self.strict:
            raise IngestError(f"{location}: {message}" if location else message)
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
        return IngestResult(source_format=self.source_format, records=self.records, issues=issues)


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
