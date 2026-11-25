"""Anthropic Console / Admin API usage-report adapters (JSON and CSV).

Column names follow the Usage and Cost Analytics API (`/v1/organizations/analytics/usage`)
and the legacy usage report (`/v1/organizations/usage_report/messages`). CSV headers are
matched case-insensitively against the same names, with nested fields flattened using `.`
or `_` (e.g. `cache_creation.ephemeral_5m_input_tokens` or `ephemeral_5m_input_tokens`).

Every row is an aggregate over a time bucket. Adjust COLUMN_MAP if a real export differs.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tokenize_cli.ingest.base import IngestError, IngestResult, RowCollector, flatten

CSV_NAME = "console-csv"
JSON_NAME = "console-json"

# Source column (normalised) -> UsageRecord field.
COLUMN_MAP: dict[str, str] = {
    "starting_at": "timestamp",
    "ending_at": "period_end",
    "model": "model",
    "uncached_input_tokens": "input_tokens",
    "cache_creation.ephemeral_5m_input_tokens": "cache_creation_5m_input_tokens",
    "cache_creation_ephemeral_5m_input_tokens": "cache_creation_5m_input_tokens",
    "ephemeral_5m_input_tokens": "cache_creation_5m_input_tokens",
    "cache_creation.ephemeral_1h_input_tokens": "cache_creation_1h_input_tokens",
    "cache_creation_ephemeral_1h_input_tokens": "cache_creation_1h_input_tokens",
    "ephemeral_1h_input_tokens": "cache_creation_1h_input_tokens",
    "cache_read_input_tokens": "cache_read_input_tokens",
    "output_tokens": "output_tokens",
    "requests": "request_count",
    "api_key_id": "api_key_id",
    "workspace_id": "workspace_id",
    "service_tier": "service_tier",
    "inference_geo": "inference_geo",
    "product": "product",
    "context_window": "long_context",
    # Actor identity: analytics `actor.*` object, legacy account/service-account ids, or CSV.
    "actor": "actor_id",
    "actor_id": "actor_id",
    "actor.user_id": "actor_id",
    "actor.service_account_id": "actor_id",
    "actor.api_key_id": "actor_id",
    "actor.id": "actor_id",
    "account_id": "actor_id",
    "service_account_id": "actor_id",
}

# Known report columns tokenize doesn't use; dropped without a warning.
_KNOWN_UNUSED = re.compile(
    r"^(total_tokens|speed|actor_type|actor\..*|server_tool_use.*|web_search_requests|"
    r"slack_channel_id|claude_tag_.*|rbac_group_id|user_profile_id|agent_id|environment_id|"
    r"deployment_id|session_id)$"
)
_LONG_CONTEXT = {"0-200k": False, "200k-1m": True}


def _norm(column: str) -> str:
    return (
        re.sub(r"[\s\-]+", "_", column.strip().lower())
        .removeprefix("group.")
        .removeprefix("usage.")
    )


def map_row(flat: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Map one flattened report row to UsageRecord input; also return unrecognised columns."""
    row: dict[str, Any] = {}
    unknown: set[str] = set()
    for column, value in flat.items():
        key = _norm(column)
        field = COLUMN_MAP.get(key)
        if field is None:
            if not _KNOWN_UNUSED.match(key):
                unknown.add(column)
            continue
        if value is None or value == "":
            continue
        if field == "long_context":
            value = _LONG_CONTEXT.get(str(value).lower())
            if value is None:
                continue
        if field == "actor_id" and "actor_id" in row:
            continue  # first identity column wins
        row[field] = value
    # Legacy report has no request counts: mark the aggregate as count-unknown.
    row.setdefault("request_count", None)
    if row.get("service_tier") == "batch":
        row["is_batch"] = True
    return row, unknown


def _add(rows: RowCollector, loc: str, flat: dict[str, Any]) -> None:
    row, unknown = map_row(flat)
    rows.ignore(unknown)
    if row.get("model") is None:
        # e.g. code-execution usage, or a report not grouped by model.
        rows.issue(loc, "row has no model (group the export by model); skipped")
        return
    rows.add(loc, row)


def _iter_json_rows(doc: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    pages = doc if isinstance(doc, list) else [doc]
    for p, page in enumerate(pages):
        if not isinstance(page, dict) or not isinstance(page.get("data"), list):
            raise IngestError("expected a usage report object with a `data` list")
        prefix = f"page {p} " if len(pages) > 1 else ""
        for i, item in enumerate(page["data"]):
            if not isinstance(item, dict):
                continue
            if "cost" in item:
                raise IngestError("this is a cost report; export the usage report instead")
            bounds = {k: item.get(k) for k in ("starting_at", "ending_at")}
            if isinstance(item.get("results"), list):  # legacy: buckets of results
                for j, result in enumerate(item["results"]):
                    yield f"{prefix}data[{i}].results[{j}]", {**bounds, **flatten(result)}
            else:  # analytics: one row with `group` and `usage`
                rest = {k: v for k, v in item.items() if k not in ("group", "usage")}
                merged = {**rest, **(item.get("group") or {}), **(item.get("usage") or {})}
                yield f"{prefix}data[{i}]", flatten(merged)


class ConsoleJsonAdapter:
    name = JSON_NAME

    def detect(self, path: Path, head: str) -> bool:
        return path.suffix.lower() == ".json" or (
            head.lstrip().startswith(("{", "[")) and '"data"' in head
        )

    def read(self, path: Path, *, strict: bool = False) -> IngestResult:
        try:
            doc = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise IngestError(f"invalid JSON: {exc}") from exc
        rows = RowCollector(JSON_NAME, strict=strict)
        for loc, flat in _iter_json_rows(doc):
            _add(rows, loc, flat)
        return rows.result()


class ConsoleCsvAdapter:
    name = CSV_NAME

    def detect(self, path: Path, head: str) -> bool:
        return path.suffix.lower() == ".csv"

    def read(self, path: Path, *, strict: bool = False) -> IngestResult:
        rows = RowCollector(CSV_NAME, strict=strict)
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            headers = {_norm(h) for h in reader.fieldnames or []}
            if "model" not in headers or not headers & {"uncached_input_tokens", "output_tokens"}:
                raise IngestError(
                    "CSV is missing usage-report columns (need `model` plus token columns)"
                )
            for rowno, raw in enumerate(reader, start=2):  # header is row 1
                _add(rows, f"row {rowno}", {k: v for k, v in raw.items() if k is not None})
        return rows.result()
