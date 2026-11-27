"""Typer entrypoint.

`ingest` previews a usage file, `audit` writes a bleed report and `simulate` replays logged
requests against other models through the Batch API.
"""

import sys
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from triage import __version__
from triage.audit import AuditResult, run_audit
from triage.ingest import ADAPTERS, IngestError, IngestResult, load
from triage.ingest.base import IngestIssue
from triage.ingest.replay import load_replays
from triage.pricing import PricingTable, UnknownModelError
from triage.report.builder import render_html, render_markdown, tokens, usd

# Modules rather than functions, so spend estimation and the paid steps can be swapped in tests.
from triage.simulate import batch, cli_helpers, estimate, eval_judge, summarize

app = typer.Typer(
    help="Audit Claude API spend and simulate model downgrades offline.",
    no_args_is_help=True,
)
console = Console()
_MAX_ISSUES_SHOWN = 10
REPORT_STEM = "triage-audit"
DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"triage {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """triage: Claude API spend auditor."""


def _summary_table(result: IngestResult) -> Table:
    records = result.records
    table = Table(title=f"Ingest preview: {escape(result.source)}", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Format", result.source_format)
    if result.file_count > 1:
        table.add_row("Files", f"{result.file_count:,}")
    table.add_row("Rows", f"{len(records):,}")
    table.add_row("Kind", "aggregated report rows" if result.aggregated else "per-request")
    if (span := result.time_range) is not None:
        table.add_row("Time range", f"{span[0].isoformat()} -> {span[1].isoformat()}")
    table.add_row("Models", ", ".join(sorted({r.model for r in records})) or "-")
    table.add_row("Issues", str(len(result.issues)))
    return table


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error:[/red] {escape(message)}")
    return typer.Exit(1)


def _print_issues(issues: list[IngestIssue]) -> None:
    for issue in issues[:_MAX_ISSUES_SHOWN]:
        where = f"{issue.location}: " if issue.location else ""
        console.print(f"[yellow]warning:[/yellow] {escape(where + issue.message)}")
    if len(issues) > _MAX_ISSUES_SHOWN:
        console.print(f"... and {len(issues) - _MAX_ISSUES_SHOWN} more issues")


_FormatOption = Annotated[
    str | None,
    typer.Option("--format", help=f"Override auto-detection: {', '.join(ADAPTERS)}."),
]
_StrictOption = Annotated[bool, typer.Option("--strict", help="Fail on the first invalid row.")]
# A str, so that glob patterns reach the ingest path resolver untouched.
_UsagePath = Annotated[
    str,
    typer.Argument(
        metavar="PATH",
        help="Usage log or Console export: a file, a folder (searched for JSONL) or a glob.",
    ),
]


def _out_dir(output_dir: Path | None) -> Path:
    return output_dir.expanduser() if output_dir else Path.cwd()


@app.command()
def ingest(
    path: _UsagePath,
    fmt: _FormatOption = None,
    strict: _StrictOption = False,
) -> None:
    """Parse usage logs and preview what triage sees in them."""
    try:
        result = load(path, fmt=fmt, strict=strict)
    except ValueError as exc:  # IngestError included
        raise _fail(str(exc)) from exc

    console.print(_summary_table(result))
    _print_issues(result.issues)


def _audit_table(source: str, audit_result: AuditResult) -> Table:
    table = Table(title=f"Audit: {escape(source)}", show_header=False)
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("Total spend", usd(audit_result.total_spend))
    table.add_row("Estimated savings", usd(audit_result.total_estimated_savings))
    table.add_row("Priority Tier tokens ($0)", tokens(audit_result.priority.total_tokens))
    table.add_row("Uncached repeated prompts", str(len(audit_result.uncached_repeats)))
    table.add_row("100k-token cliff requests", str(len(audit_result.haiku_cliff)))
    table.add_row("Bloated max_tokens requests", str(len(audit_result.bloated_max_tokens)))
    return table


@app.command()
def audit(
    path: _UsagePath,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Where to write the reports (default: cwd)."),
    ] = None,
    fmt: _FormatOption = None,
    strict: _StrictOption = False,
) -> None:
    """Detect spend bleed offline and write Markdown + HTML reports. Makes no API calls."""
    try:
        ingested = load(path, fmt=fmt, strict=strict)
        source = ingested.source
        result = run_audit(ingested.records, PricingTable.load())
    except (IngestError, UnknownModelError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    _print_issues(ingested.issues)
    out = _out_dir(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path, html_path = out / f"{REPORT_STEM}.md", out / f"{REPORT_STEM}.html"
    md_path.write_text(render_markdown(result, source=source), encoding="utf-8")
    html_path.write_text(render_html(result, source=source), encoding="utf-8")

    console.print(_audit_table(source, result))
    # Plain echo so long paths are never wrapped mid-name.
    typer.echo(f"Wrote {md_path}")
    typer.echo(f"Wrote {html_path}")


# --------------------------------------------------------------------------- simulate


@app.command()
def simulate(
    path: Annotated[
        str,
        typer.Argument(
            metavar="PATH",
            help="Canonical JSONL with prompts and responses: a file, folder or glob.",
        ),
    ],
    target: Annotated[
        list[str],
        typer.Option("--target", "-t", "-m", help="Candidate model; repeat to compare several."),
    ],
    judge_model: Annotated[
        str, typer.Option("--judge-model", help="Model that grades candidate responses.")
    ] = DEFAULT_JUDGE_MODEL,
    no_judge: Annotated[
        bool, typer.Option("--no-judge", help="Deterministic checks only.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Estimate spend and stop. Needs no API key.")
    ] = False,
    sample: Annotated[
        int | None, typer.Option("--sample", min=1, help="Replay a random subset of N tasks.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for --sample.")] = 0,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the interactive spend confirmation.")
    ] = False,
    min_pass_rate: Annotated[
        float | None,
        typer.Option(
            "--min-pass-rate",
            min=0.0,
            max=1.0,
            help="Exit 1 if any target's 95% lower bound is below this proportion.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Where to write the reports (default: cwd)."),
    ] = None,
) -> None:
    """Replay logged requests against other models via the Batch API; compare cost and quality."""
    try:
        pricing = PricingTable.load()
        targets = batch.resolve_targets(target, pricing)
        judge = None if no_judge else pricing.resolve(judge_model)
        replays = load_replays(path)
        source = replays.source
        _print_issues(replays.issues)
        tasks = cli_helpers.select_tasks(replays.tasks, source, sample=sample, seed=seed)
        if len(tasks) < len(replays.tasks):
            console.print(f"Sampled {len(tasks):,} of {len(replays.tasks):,} tasks (seed {seed}).")
        cost = estimate.estimate_cost(tasks, targets, pricing, judge_model=judge)
    except (OSError, UnknownModelError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    console.print(cli_helpers.estimate_table(cost, console))
    if dry_run:
        console.print("Dry run: nothing was submitted.")
        return
    # Only a terminal can answer the prompt; scripted runs proceed after the printed estimate.
    if not yes and sys.stdin.isatty():
        typer.confirm("Submit these requests to the paid Batch API?", abort=True)

    out = _out_dir(output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)  # fail on a bad path before spending
        client = batch.create_client()
        outcomes = batch.run_replay(
            client, tasks, targets, pricing, on_submit=cli_helpers.announce("replay")
        )
    except Exception as exc:  # SDK, auth and network errors all end the run the same way
        raise _fail(f"replay failed: {exc}") from exc

    judge_cost, judge_error = Decimal(0), None
    if judge is not None:
        try:
            judged = eval_judge.run_judge(
                client, tasks, outcomes, judge, pricing, on_submit=cli_helpers.announce("judge")
            )
            outcomes, judge_cost = judged.outcomes, judged.cost
        except Exception as exc:  # still report the replay results that are already paid for
            judge_error = f"judge failed, reports have no verdicts: {exc}"

    result = summarize.summarize_simulation(
        outcomes,
        targets=targets,
        logged_models=cli_helpers.logged_models(tasks, pricing),
        judge_model=judge,
        judge_cost=judge_cost,
        source=source,
    )
    md_path, html_path = cli_helpers.write_reports(result, outcomes, out)

    console.print(cli_helpers.simulation_table(result, console))
    console.print(f"Batch API spend (from reported usage): {usd(result.total_sim_cost)}")
    typer.echo(f"Wrote {md_path}")
    typer.echo(f"Wrote {html_path}")

    if judge_error is not None:
        raise _fail(judge_error)
    if min_pass_rate is None:
        return
    below = cli_helpers.below_pass_rate(result, min_pass_rate)
    for row in below:
        console.print(
            f"[red]below --min-pass-rate:[/red] {escape(row.model)} lower bound "
            f"{row.pass_rate.low:.3f} < {min_pass_rate:g}"
        )
    if below:
        raise typer.Exit(1)


def run() -> None:
    """Console-script entry point."""
    # Click would otherwise expand globs and `~` itself on Windows, turning one quoted
    # pattern into many arguments before the ingest path resolver could see it.
    app(windows_expand_args=False)
