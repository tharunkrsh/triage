"""Typer entrypoint. `ingest` previews a usage file; `audit`/`simulate` come later."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tokenize_cli import __version__
from tokenize_cli.ingest import ADAPTERS, IngestError, IngestResult, load

app = typer.Typer(
    help="Audit Claude API spend and simulate model downgrades offline.",
    no_args_is_help=True,
)
console = Console()
_MAX_ISSUES_SHOWN = 10


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


@app.command()
def ingest(
    file: Annotated[Path, typer.Argument(help="Canonical JSONL log or Console usage export.")],
    fmt: Annotated[
        str | None,
        typer.Option("--format", help=f"Override auto-detection: {', '.join(ADAPTERS)}."),
    ] = None,
    strict: Annotated[
        bool, typer.Option("--strict", help="Fail on the first invalid row.")
    ] = False,
) -> None:
    """Parse a usage file and preview what tokenize sees in it."""
    try:
        result = load(file, fmt=fmt, strict=strict)
    except IngestError as exc:
        console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    console.print(_summary_table(file, result))
    for issue in result.issues[:_MAX_ISSUES_SHOWN]:
        where = f"{issue.location}: " if issue.location else ""
        console.print(f"[yellow]warning:[/yellow] {escape(where + issue.message)}")
    if len(result.issues) > _MAX_ISSUES_SHOWN:
        console.print(f"... and {len(result.issues) - _MAX_ISSUES_SHOWN} more issues")
