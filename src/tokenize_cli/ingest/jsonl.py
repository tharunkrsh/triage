"""Canonical JSONL reader: one request per line.

Each line is either flat (UsageRecord field names at the top level) or carries an API-style
`usage` object as returned by the Messages API. Simulator-only keys (prompt messages,
baseline response, schemas) are allowed and ignored here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tokenize_cli.ingest.base import RECORD_FIELDS, IngestResult, RowCollector

NAME = "jsonl"

# Keys the simulator's canonical JSONL carries that have no meaning for the audit.
_SIMULATE_KEYS = frozenset(
    {"messages", "response", "baseline", "expected_schema", "tools", "prompt_version", "metadata"}
)
_USAGE_TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens")


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
    for key in ("service_tier", "inference_geo"):
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
        with path.open(encoding="utf-8-sig") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                loc = f"line {lineno}"
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    rows.issue(loc, f"invalid JSON: {exc.msg}")
                    continue
                if not isinstance(obj, dict):
                    rows.issue(loc, "expected a JSON object")
                    continue
                row, unknown = map_line(obj)
                rows.ignore(unknown)
                rows.add(loc, row)
        return rows.result()
