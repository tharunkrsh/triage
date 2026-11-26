"""Task selection, terminal tables and report output for `tokenize simulate`.

Nothing here constructs a client or submits a batch; `cli.py` owns those steps.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path

import typer
from rich.markup import escape
from rich.table import Table

from tokenize_cli.pricing import PricingTable, UnknownModelError
from tokenize_cli.report.builder import (
    ci,
    render_simulation_html,
    render_simulation_markdown,
    usd,
    usd_fine,
)
from tokenize_cli.simulate.batch import check_unique_task_ids
from tokenize_cli.simulate.models import (
    CandidateOutcome,
    CandidateSummary,
    CostEstimate,
    ReplayTask,
    SimulationResult,
)

REPORT_STEM = "tokenize-simulate"


def select_tasks(
    tasks: Sequence[ReplayTask], source: Path, *, sample: int | None, seed: int
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


def baseline_keys(
    tasks: Sequence[ReplayTask], pricing: PricingTable
) -> set[tuple[str | None, str | None, str]]:
    """(feature, prompt_version, model) keys of the incumbent models."""
    keys = set()
    for task in tasks:
        try:
            model = pricing.resolve(task.baseline_model)
        except UnknownModelError:
            continue  # an unpriced incumbent just isn't flagged
        keys.add((task.feature, task.prompt_version, model))
    return keys


def announce(kind: str) -> Callable[[str], None]:
    """Print a batch id at once: paid results stay retrievable if polling is interrupted."""
    return lambda batch_id: typer.echo(f"Submitted {kind} batch {batch_id}; waiting for results.")


def estimate_table(cost: CostEstimate) -> Table:
    table = Table(title="Estimated Batch API spend")
    table.add_column("Model", overflow="fold")
    table.add_column("Role")
    for name in ("Requests", "Expected", "Upper bound"):
        table.add_column(name, justify="right")
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


def simulation_table(result: SimulationResult) -> Table:
    table = Table(title="Simulation results")
    table.add_column("Model", overflow="fold")
    table.add_column("Pass rate [95% CI]")
    table.add_column("Errored", justify="right")
    table.add_column("Cost per task", justify="right")
    for row in result.overall:
        table.add_row(
            row.model,
            escape(ci(row.pass_rate)),
            f"{row.errored:,}",
            usd_fine(row.projected_cost_per_task),
        )
    return table


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
