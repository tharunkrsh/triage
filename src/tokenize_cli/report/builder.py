"""Render an AuditResult as Markdown or self-contained HTML via Jinja2 templates."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tokenize_cli.audit.models import AuditResult

TEMPLATES = Path(__file__).parent / "templates"
_CENT = Decimal("0.01")

DETECTOR_LABELS = {
    "uncached_repeat": "Uncached repeated prompt",
    "haiku_cliff": "100k-token cliff",
    "bloated_max_tokens": "Bloated max_tokens",
}


def usd(value: Decimal | float | int) -> str:
    amount = Decimal(str(value))
    if 0 < amount < _CENT / 2:
        return "<$0.01"
    return f"${amount.quantize(_CENT, rounding=ROUND_HALF_UP):,}"


def tokens(value: int) -> str:
    return f"{value:,}"


def pct(share: float) -> str:
    return f"{share:.1%}"


def ratio(value: float | None) -> str:
    return "n/a (no output)" if value is None else f"{value:,.1f}x"


def md_cell(value: object) -> str:
    """Make a value safe inside a Markdown table cell."""
    text = "" if value is None else str(value)
    return text.replace("|", "&#124;").replace("\r", " ").replace("\n", " ")


def _environment(*, html: bool) -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=html,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters.update(usd=usd, tokens=tokens, pct=pct, ratio=ratio, md=md_cell)
    env.globals["detector_label"] = DETECTOR_LABELS.get
    return env


_MD = _environment(html=False)
_HTML = _environment(html=True)


def _context(result: AuditResult, source: str | None) -> dict[str, object]:
    c = result.concentration
    return {
        "r": result,
        "source": source,
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "concentration": [
            ("Model", c.by_model),
            ("Feature", c.by_feature),
            ("API key", c.by_api_key),
            ("Workspace", c.by_workspace),
        ],
    }


def render_markdown(result: AuditResult, *, source: str | None = None) -> str:
    return _MD.get_template("report.md.j2").render(_context(result, source))


def render_html(result: AuditResult, *, source: str | None = None) -> str:
    return _HTML.get_template("report.html.j2").render(_context(result, source))
