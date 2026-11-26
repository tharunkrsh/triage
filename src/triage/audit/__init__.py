"""Mode 1 metadata bleed audit (DuckDB-backed)."""

from triage.audit.bleed import run_audit
from triage.audit.models import AuditResult

__all__ = ["AuditResult", "run_audit"]
