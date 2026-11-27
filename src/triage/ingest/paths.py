"""Turn one path argument (file, directory or glob) into an ordered list of input files.

`~` is expanded before anything else, so quoted paths work in shells that pass them through
literally (Windows PowerShell). Globs are expanded here, including `**`.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from triage.ingest.base import LINE_SUFFIXES, IngestError

_GLOB_CHARS = frozenset("*?[")


class InputSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str  # a file, folder or pattern name for report headers; never a full path
    files: list[Path]
    base: Path | None = None  # None when one file was named directly

    @property
    def single(self) -> bool:
        return self.base is None

    def subset(self, files: list[Path]) -> InputSet:
        """The same input, narrowed to `files` (for example, the files of one format)."""
        return self.model_copy(update={"files": files})

    def display(self, path: Path) -> str:
        """`path` relative to the directory or fixed glob prefix it was found under."""
        if self.base is None:
            return path.name
        return Path(os.path.relpath(path, self.base)).as_posix()


def _unique(paths: list[Path]) -> list[Path]:
    """Sort for a stable order and drop different spellings of the same file."""
    seen: dict[Path, Path] = {}
    for path in sorted(paths, key=lambda p: p.as_posix()):
        seen.setdefault(path.resolve(), path)
    return list(seen.values())


def _glob_base(pattern: Path) -> Path:
    """The directories before the first wildcard."""
    fixed: list[str] = []
    for part in pattern.parts[:-1]:
        if _GLOB_CHARS & set(part):
            break
        fixed.append(part)
    return Path(*fixed) if fixed else Path()


def resolve_inputs(spec: Path | str) -> InputSet:
    """Resolve `spec`; raises IngestError when it names no files.

    An existing file or directory wins over glob syntax, so names containing `[` still work.
    Directories are searched recursively for JSONL only; a glob takes whatever it matches.
    """
    path = Path(os.path.expanduser(str(spec)))
    if path.is_file():
        return InputSet(label=path.name, files=[path])
    if path.is_dir():
        found = [p for p in path.rglob("*") if p.suffix.lower() in LINE_SUFFIXES and p.is_file()]
        if not found:
            raise IngestError(f"no .jsonl or .ndjson files under {path}")
        label = path.name or path.resolve().name or str(path)
        return InputSet(label=label, files=_unique(found), base=path)
    if _GLOB_CHARS & set(str(path)):
        # include_hidden: `**` must be able to cross dot folders such as `.claude`.
        matches = glob.glob(str(path), recursive=True, include_hidden=True)
        found = [Path(m) for m in matches if os.path.isfile(m)]
        if not found:
            raise IngestError(f"no files match {path}")
        return InputSet(label=path.name, files=_unique(found), base=_glob_base(path))
    raise IngestError(f"no such file: {path}")
