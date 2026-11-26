"""Render audit and simulation results as Markdown or self-contained HTML (Jinja2)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import NamedTuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tokenize_cli.audit.models import AuditResult
from tokenize_cli.simulate.models import (
    CandidateOutcome,
    CandidateSummary,
    SimulationResult,
    WilsonInterval,
)

TEMPLATES = Path(__file__).parent / "templates"
_CENT = Decimal("0.01")
_MAX_FAILURES = 20
_MAX_EXCERPT = 500

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


def ci(interval: WilsonInterval | None) -> str:
    """A proportion with its Wilson bounds and sample size; "n/a" without evidence."""
    if interval is None or interval.p_hat is None:
        return "n/a (n=0)"
    return f"{pct(interval.p_hat)} [{pct(interval.low)}–{pct(interval.high)}], n={interval.n}"


def usd_fine(value: Decimal | None) -> str:
    """Per-task amounts are often far below a cent."""
    return "n/a" if value is None else f"${value:,.6f}"


def md_text(value: object) -> str:
    """`md_cell` for model-written text: also neutralises HTML, links and images."""
    text = "" if value is None else str(value)
    for char, entity in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ("[", "&#91;")):
        text = text.replace(char, entity)
    return md_cell(text)


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
    env.filters.update(ci=ci, usd_fine=usd_fine, md_text=md_text)
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


# --------------------------------------------------------------------------- simulation


class _Point(NamedTuple):
    summary: CandidateSummary
    x: float
    y: float
    y_low: float
    y_high: float


class _Chart(NamedTuple):
    title: str
    max_cost: Decimal
    points: list[_Point]


class _Failure(NamedTuple):
    outcome: CandidateOutcome
    reason: str
    excerpt: str


# Plot area inside the template's 640x320 viewBox.
_LEFT, _TOP, _WIDTH, _HEIGHT = 60, 20, 540, 250


def group_label(feature: str | None, prompt_version: str | None) -> str:
    return f"{feature or '(no feature)'} / {prompt_version or '(no version)'}"


def _chart(title: str, rows: list[CandidateSummary]) -> _Chart:
    """Cost (x) against pass rate (y) for the rows of one group that have evidence."""
    max_cost = max(row.projected_cost_per_task for row in rows)
    span = float(max_cost) or 1.0  # an all-free group plots on the y axis

    def y(rate: float) -> float:
        return round(_TOP + (1 - rate) * _HEIGHT, 2)

    points = [
        _Point(
            summary=row,
            x=round(_LEFT + float(row.projected_cost_per_task) / span * _WIDTH, 2),
            y=y(row.pass_rate.p_hat),
            y_low=y(row.pass_rate.low),
            y_high=y(row.pass_rate.high),
        )
        for row in rows
    ]
    return _Chart(title, max_cost, points)


def _charts(result: SimulationResult) -> list[_Chart]:
    """One chart per (feature, prompt_version) group with graded tasks."""
    groups: dict[str, list[CandidateSummary]] = {}
    for row in result.summaries:
        if row.n:
            groups.setdefault(group_label(row.feature, row.prompt_version), []).append(row)
    return [_chart(title, rows) for title, rows in groups.items()]


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_EXCERPT else text[:_MAX_EXCERPT] + "…"


def _failure(outcome: CandidateOutcome) -> _Failure:
    if outcome.status != "succeeded":
        return _Failure(outcome, f"replay {outcome.status}", "")
    reasons = []
    if outcome.det is not None:
        reasons += [f"{c.name}: {c.detail or 'failed'}" for c in outcome.det.checks if not c.passed]
    if outcome.judge is not None and not outcome.judge.passed:
        reasons.append(f"judge ({outcome.judge.score}/5): {outcome.judge.rationale}")
    excerpt = ""
    if (response := outcome.response) is not None:
        calls = " ".join(f"{c.name}({c.input})" for c in response.tool_calls)
        excerpt = f"{response.text} {calls}".strip()
    return _Failure(outcome, _clip("; ".join(reasons)), _clip(excerpt))


def _simulation_context(
    result: SimulationResult, outcomes: Sequence[CandidateOutcome]
) -> dict[str, object]:
    failed = [o for o in outcomes if o.status != "succeeded" or o.passed is False]
    return {
        "r": result,
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "group_label": group_label,
        "charts": _charts(result),
        "failures": [_failure(o) for o in failed[:_MAX_FAILURES]],
        "failure_count": len(failed),
    }


def render_simulation_markdown(
    result: SimulationResult, outcomes: Sequence[CandidateOutcome] = ()
) -> str:
    return _MD.get_template("simulate.md.j2").render(_simulation_context(result, outcomes))


def render_simulation_html(
    result: SimulationResult, outcomes: Sequence[CandidateOutcome] = ()
) -> str:
    return _HTML.get_template("simulate.html.j2").render(_simulation_context(result, outcomes))
