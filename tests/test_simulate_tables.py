"""Layout of the `triage simulate` terminal tables at common widths."""

from decimal import Decimal
from io import StringIO

import pytest
from rich.console import Console

from tests.simulate_fakes import make_outcome
from triage.report.builder import ci
from triage.simulate import cli_helpers
from triage.simulate.models import CostEstimate, CostLine, SimulationResult
from triage.simulate.summarize import summarize_simulation

SHORT = ("claude-haiku-5-5", "claude-sonnet-5-5")
LONG = "claude-3-7-sonnet-20250219"  # among the longest model ids in pricing.yaml
FULL = ["Model", "Role", "Requests", "Expected", "Upper bound"]
COMPACT = ["Model", "Role", "Reqs", "Est.", "Max"]


def _estimate(models=SHORT, *, requests: int = 4, usd: str = "0.52") -> CostEstimate:
    roles = [(model, "candidate") for model in models] + [(models[-1], "judge")]
    return CostEstimate(
        lines=[
            CostLine(
                model=model,
                role=role,
                requests=requests,
                input_tokens=10,
                output_tokens=10,
                expected_usd=Decimal(usd),
                upper_bound_usd=Decimal(usd),
                estimated_tokens=True,
            )
            for model, role in roles
        ]
    )


def _results(models=SHORT, *, tasks: int = 4) -> SimulationResult:
    outcomes = [
        make_outcome(f"t{i}", model, det=i < tasks - 1, projected="0.0004")
        for model in models
        for i in range(tasks)
    ]
    return summarize_simulation(outcomes, targets=list(models))


def _render(build, data, width: int) -> str:
    console = Console(
        width=width, record=True, file=StringIO(), color_system=None, legacy_windows=False
    )
    console.print(build(data, console))
    text = console.export_text()
    assert "…" not in text and "..." not in text
    assert max(len(line) for line in text.splitlines()) <= width
    return text


def _cells(text: str) -> list[list[str]]:
    """Stripped cells of the header line, then of every body line."""
    return [
        [cell.strip() for cell in line.replace("┃", "│").split("│")[1:-1]]
        for line in text.splitlines()
        if line.startswith(("┃", "│"))
    ]


@pytest.mark.parametrize(("width", "headers"), [(64, COMPACT), (80, FULL), (120, FULL)])
def test_estimate_table_keeps_headers_roles_and_models_whole(width, headers):
    header, *rows = _cells(_render(cli_helpers.estimate_table, _estimate(), width))

    assert header == headers
    assert [row[:2] for row in rows] == [
        [f"{SHORT[0]} *", "candidate"],
        [f"{SHORT[1]} *", "candidate"],
        [f"{SHORT[1]} *", "judge"],
        ["Total", ""],
    ]


@pytest.mark.parametrize(("width", "headers"), [(64, COMPACT), (80, FULL), (120, FULL)])
def test_estimate_table_with_long_values_only_wraps_the_model_when_narrow(width, headers):
    data = _estimate((LONG,), requests=12_000, usd="1234.56")

    header, *rows = _cells(_render(cli_helpers.estimate_table, data, width))

    assert header == headers
    assert [row[1] for row in rows if row[1]] == ["candidate", "judge"]
    assert [row[2] for row in rows if row[2]] == ["12,000", "12,000", "24,000"]
    if width >= 80:
        assert [row[0] for row in rows] == [f"{LONG} *", f"{LONG} *", "Total"]


@pytest.mark.parametrize(
    ("width", "rate_header"),
    [(64, "Pass [95% CI]"), (80, "Pass rate [95% CI]"), (120, "Pass rate [95% CI]")],
)
def test_simulation_table_keeps_headers_and_models_whole(width, rate_header):
    result = _results()

    text = _render(cli_helpers.simulation_table, result, width)
    header, *rows = _cells(text)

    assert header == ["Model", rate_header, "Errored", "Cost per task"]
    assert [row[0] for row in rows if row[0]] == list(SHORT)
    if width >= 80:
        assert ci(result.overall[0].pass_rate) in text
    else:
        assert [row[1] for row in rows[:2]] == ["75.0%, n=4", "[30.1%–95.4%]"]


@pytest.mark.parametrize("width", [80, 120])
def test_simulation_table_keeps_a_long_model_id_whole(width):
    result = _results((LONG,), tasks=1000)

    rows = _cells(_render(cli_helpers.simulation_table, result, width))

    assert LONG in [row[0] for row in rows]


def test_tables_without_a_console_use_the_full_layout():
    table = cli_helpers.estimate_table(_estimate())

    assert [column.header for column in table.columns] == FULL
