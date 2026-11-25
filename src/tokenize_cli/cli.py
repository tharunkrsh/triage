"""Typer entrypoint: `ingest` previews a usage file, `audit` writes a bleed report."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tokenize_cli import __version__
from tokenize_cli.audit import AuditResult, run_audit
from tokenize_cli.ingest import ADAPTERS, IngestError, IngestResult, load
from tokenize_cli.pricing import PricingTable, UnknownModelError
from tokenize_cli.report.builder import render_html, render_markdown, tokens, usd

app = typer.Typer(
    help="Audit Claude API spend and simulate model downgrades offline.",
    no_args_is_help=True,
)
console = Console()
_MAX_ISSUES_SHOWN = 10
REPORT_STEM = "tokenize-audit"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tokenize {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """tokenize: Claude API spend auditor."""


def _summary_table(path: Path, result: IngestResult) -> Table:
    records = result.records
    table = Table(title=f"Ingest preview: {path.name}", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Format", result.source_format)
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


def _print_issues(result: IngestResult) -> None:
    for issue in result.issues[:_MAX_ISSUES_SHOWN]:
        where = f"{issue.location}: " if issue.location else ""
        console.print(f"[yellow]warning:[/yellow] {escape(where + issue.message)}")
    if len(result.issues) > _MAX_ISSUES_SHOWN:
        console.print(f"... and {len(result.issues) - _MAX_ISSUES_SHOWN} more issues")


_FormatOption = Annotated[
    str | None,
    typer.Option("--format", help=f"Override auto-detection: {', '.join(ADAPTERS)}."),
]
_StrictOption = Annotated[bool, typer.Option("--strict", help="Fail on the first invalid row.")]


@app.command()
def ingest(
    file: Annotated[Path, typer.Argument(help="Canonical JSONL log or Console usage export.")],
    fmt: _FormatOption = None,
    strict: _StrictOption = False,
) -> None:
    """Parse a usage file and preview what tokenize sees in it."""
    try:
        result = load(file, fmt=fmt, strict=strict)
    except IngestError as exc:
        raise _fail(str(exc)) from exc

    console.print(_summary_table(file, result))
    _print_issues(result)


def _audit_table(path: Path, audit_result: AuditResult) -> Table:
    table = Table(title=f"Audit: {path.name}", show_header=False)
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
    file: Annotated[Path, typer.Argument(help="Canonical JSONL log or Console usage export.")],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Where to write the reports (default: cwd)."),
    ] = None,
    fmt: _FormatOption = None,
    strict: _StrictOption = False,
) -> None:
    """Detect spend bleed offline and write Markdown + HTML reports. Makes no API calls."""
    try:
        ingested = load(file, fmt=fmt, strict=strict)
        result = run_audit(ingested.records, PricingTable.load())
    except (IngestError, UnknownModelError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    _print_issues(ingested)
    out = output_dir or Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    md_path, html_path = out / f"{REPORT_STEM}.md", out / f"{REPORT_STEM}.html"
    md_path.write_text(render_markdown(result, source=file.name), encoding="utf-8")
    html_path.write_text(render_html(result, source=file.name), encoding="utf-8")

    console.print(_audit_table(file, result))
    # Plain echo so long paths are never wrapped mid-name.
    typer.echo(f"Wrote {md_path}")
    typer.echo(f"Wrote {html_path}")
