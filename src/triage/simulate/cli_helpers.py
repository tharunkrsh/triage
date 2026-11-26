"""Task selection, terminal tables and report output for `triage simulate`.

Nothing here constructs a client or submits a batch; `cli.py` owns those steps.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.measure import Measurement
from rich.table import Table

from triage.pricing import PricingTable, UnknownModelError
from triage.report.builder import (
    ci,
    render_simulation_html,
    render_simulation_markdown,
    usd,
    usd_fine,
)
from triage.simulate.batch import check_unique_task_ids
from triage.simulate.models import (
    MIXED_MODELS,
    CandidateOutcome,
    CandidateSummary,
    CostEstimate,
    ReplayTask,
    SimulationResult,
    WilsonInterval,
)

REPORT_STEM = "triage-simulate"


def select_tasks(
    tasks: Sequence[ReplayTask], source: Path | str, *, sample: int | None, seed: int
) -> list[ReplayTask]:
    """Validate the loaded tasks and apply `--sample`, keeping file order."""
    if not tasks:
        raise ValueError(f"no replayable tasks in {source}")
    check_unique_task_ids(tasks)
    if sample is None or sample >= len(tasks):
        return list(tasks)
    # Seeded, so a dry run prices exactly the tasks the paid run will submit.
    chosen = sorted(random.Random(seed).sample(range(len(tasks)), sample))
    return [tasks[i] for i in chosen]


def logged_models(
    tasks: Sequence[ReplayTask], pricing: PricingTable
) -> dict[tuple[str | None, str | None], str]:
    """The production model of each (feature, prompt_version) slice, or "mixed"."""
    seen: dict[tuple[str | None, str | None], set[str]] = {}
    for task in tasks:
        try:
            model = pricing.resolve(task.baseline_model)  # aliases count as one model
        except UnknownModelError:
            model = task.baseline_model  # unpriced, but still what production ran
        seen.setdefault((task.feature, task.prompt_version), set()).add(model)
    return {key: min(names) if len(names) == 1 else MIXED_MODELS for key, names in seen.items()}


def announce(kind: str) -> Callable[[str], None]:
    """Print a batch id at once: paid results stay retrievable if polling is interrupted."""
    return lambda batch_id: typer.echo(f"Submitted {kind} batch {batch_id}; waiting for results.")


def _table(title: str, headers: Sequence[str], *, compact: bool) -> Table:
    """Two left-aligned label columns, then numbers. Only the first (model) column may wrap."""
    table = Table(title=title, pad_edge=not compact)
    for index, header in enumerate(headers):
        table.add_column(
            header,
            justify="left" if index < 2 else "right",
            no_wrap=index > 0,
            overflow="fold",  # the default, "ellipsis", would cut a squeezed value short
        )
    return table


def _fit(build: Callable[[bool], Table], console: Console | None) -> Table:
    """Fit the console: full layout, then without outer padding, then the compact layout."""
    table = build(False)
    if console is None:
        return table
    unbounded = console.options.update_width(10_000)
    for pad_edge in (True, False):
        table.pad_edge = pad_edge
        if Measurement.get(console, unbounded, table).maximum <= console.width:
            return table
    return build(True)


def estimate_table(cost: CostEstimate, console: Console | None = None) -> Table:
    def build(compact: bool) -> Table:
        numbers = ("Reqs", "Est.", "Max") if compact else ("Requests", "Expected", "Upper bound")
        headers = ("Model", "Role", *numbers)
        table = _table("Estimated Batch API spend", headers, compact=compact)
        for line in cost.lines:
            table.add_row(
                line.model + (" *" if line.estimated_tokens else ""),
                line.role,
                f"{line.requests:,}",
                usd(line.expected_usd),
                usd(line.upper_bound_usd),
            )
        requests = sum(line.requests for line in cost.lines)
        totals = (usd(cost.expected_usd), usd(cost.upper_bound_usd))
        table.add_row("Total", "", f"{requests:,}", *totals, style="bold")
        if any(line.estimated_tokens for line in cost.lines):
            table.caption = "* token counts partly approximated"
        return table

    return _fit(build, console)


def _interval(interval: WilsonInterval, *, compact: bool) -> str:
    text = ci(interval)
    if compact and " [" in text:  # put the rate and sample size above the bounds
        rate, rest = text.split(" [")
        bounds, size = rest.split("], ")
        text = f"{rate}, {size}\n[{bounds}]"
    return escape(text)


def simulation_table(result: SimulationResult, console: Console | None = None) -> Table:
    def build(compact: bool) -> Table:
        rate = "Pass [95% CI]" if compact else "Pass rate [95% CI]"
        headers = ("Model", rate, "Errored", "Cost per task")
        table = _table("Simulation results", headers, compact=compact)
        for row in result.overall:
            table.add_row(
                row.model,
                _interval(row.pass_rate, compact=compact),
                f"{row.errored:,}",
                usd_fine(row.projected_cost_per_task),
            )
        return table

    return _fit(build, console)


def write_reports(
    result: SimulationResult, outcomes: Sequence[CandidateOutcome], out: Path
) -> tuple[Path, Path]:
    """Write the Markdown and HTML reports into the existing directory `out`."""
    md_path, html_path = out / f"{REPORT_STEM}.md", out / f"{REPORT_STEM}.html"
    md_path.write_text(render_simulation_markdown(result, outcomes), encoding="utf-8")
    html_path.write_text(render_simulation_html(result, outcomes), encoding="utf-8")
    return md_path, html_path


def below_pass_rate(result: SimulationResult, minimum: float) -> list[CandidateSummary]:
    """Targets whose 95% Wilson lower bound falls short of `minimum`."""
    return [row for row in result.overall if minimum > row.pass_rate.low]
