"""Usage ingestion: pick an adapter for a file and normalise it into UsageRecords."""

from __future__ import annotations

from pathlib import Path

from tokenize_cli.ingest.anthropic_csv import ConsoleCsvAdapter, ConsoleJsonAdapter
from tokenize_cli.ingest.base import IngestAdapter, IngestError, IngestIssue, IngestResult
from tokenize_cli.ingest.jsonl import JsonlAdapter

# Detection order matters: Console JSON is checked before JSONL sniffing.
ADAPTERS: dict[str, IngestAdapter] = {
    a.name: a for a in (ConsoleCsvAdapter(), ConsoleJsonAdapter(), JsonlAdapter())
}
_HEAD_BYTES = 4096

__all__ = ["ADAPTERS", "IngestError", "IngestIssue", "IngestResult", "detect_format", "load"]


def detect_format(path: Path) -> str:
    with path.open(encoding="utf-8-sig", errors="replace") as fh:
        head = fh.read(_HEAD_BYTES)
    for name, adapter in ADAPTERS.items():
        if adapter.detect(path, head):
            return name
    raise IngestError(f"can't tell the format of {path.name}; pass --format")


def load(path: Path | str, *, fmt: str | None = None, strict: bool = False) -> IngestResult:
    """Read a usage file. `fmt` is an ADAPTERS key, or None to auto-detect."""
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"no such file: {path}")
    name = fmt or detect_format(path)
    if name not in ADAPTERS:
        raise IngestError(f"unknown format {name!r}; choose from {', '.join(ADAPTERS)}")
    return ADAPTERS[name].read(path, strict=strict)
