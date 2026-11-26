"""`triage.simulate.cli_helpers`, plus the spend confirmation that sits between its tables.

The prompt itself lives in `triage.cli`: it runs after `estimate_table` is printed and before
anything is submitted, so it is exercised here through the `simulate` command.
"""

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tests.simulate_fakes import (
    FakeAnthropic,
    forbid_client,
    install_client,
    make_outcome,
    make_task,
)
from triage import cli
from triage.pricing import PricingTable
from triage.report.builder import ci, usd, usd_fine
from triage.simulate import cli_helpers
from triage.simulate.models import CostEstimate, CostLine, SimulationResult
from triage.simulate.summarize import summarize_simulation

REPLAY = Path(__file__).parent / "fixtures" / "simulate" / "replay.jsonl"
SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"
SOURCE = Path("replay.jsonl")
PROMPT = "Submit these requests to the paid Batch API?"
REPORTS = ("triage-simulate.md", "triage-simulate.html")

runner = CliRunner()


def _render(table) -> str:
    console = Console(width=200, color_system=None)
    with console.capture() as capture:
        console.print(table)
    return capture.get()


def _tasks(count: int):
    return [make_task(f"t{i}") for i in range(count)]


def _line(model: str = SONNET, role: str = "candidate", *, estimated: bool = False) -> CostLine:
    return CostLine(
        model=model,
        role=role,
        requests=1200,
        input_tokens=10,
        output_tokens=10,
        expected_usd=Decimal("1.25"),
        upper_bound_usd=Decimal("3.50"),
        estimated_tokens=estimated,
    )


def _result(*outcomes, targets=(SONNET,)) -> SimulationResult:
    return summarize_simulation(list(outcomes), targets=list(targets), source=SOURCE.name)


# --------------------------------------------------------------------------- select_tasks


def test_select_tasks_rejects_an_empty_file():
    with pytest.raises(ValueError, match="no replayable tasks in replay.jsonl"):
        cli_helpers.select_tasks([], SOURCE, sample=None, seed=0)


def test_select_tasks_rejects_duplicate_ids():
    tasks = [make_task("same"), make_task("same")]

    with pytest.raises(ValueError, match="duplicate task_id"):
        cli_helpers.select_tasks(tasks, SOURCE, sample=None, seed=0)


@pytest.mark.parametrize("sample", [None, 5, 50])
def test_select_tasks_keeps_everything_unless_sample_is_smaller(sample):
    tasks = _tasks(5)

    assert cli_helpers.select_tasks(tasks, SOURCE, sample=sample, seed=0) == tasks


def test_select_tasks_sample_is_seeded_and_keeps_file_order():
    tasks = _tasks(20)

    first = cli_helpers.select_tasks(tasks, SOURCE, sample=6, seed=7)
    again = cli_helpers.select_tasks(tasks, SOURCE, sample=6, seed=7)

    assert first == again
    assert len(first) == 6
    positions = [tasks.index(task) for task in first]
    assert positions == sorted(set(positions))


def test_select_tasks_seed_changes_the_subset():
    tasks = _tasks(20)

    subsets = {
        tuple(t.task_id for t in cli_helpers.select_tasks(tasks, SOURCE, sample=6, seed=seed))
        for seed in range(5)
    }

    assert len(subsets) > 1


# --------------------------------------------------------------------------- baseline + announce


def test_logged_models_names_one_model_per_slice_and_resolves_aliases():
    tasks = [
        make_task("a", baseline_model="opus-5-5"),
        make_task("b", baseline_model="claude-opus-5-5"),
        make_task("c", baseline_model="claude-haiku-5-5", prompt_version="v2"),
        make_task("d", baseline_model="some-unpriced-model", feature=None),
    ]

    logged = cli_helpers.logged_models(tasks, PricingTable.load())

    assert logged == {
        ("summarise", "v1"): "claude-opus-5-5",
        ("summarise", "v2"): "claude-haiku-5-5",
        (None, "v1"): "some-unpriced-model",
    }


@pytest.mark.parametrize("other", ["claude-haiku-5-5", "some-unpriced-model"])
def test_logged_models_reports_a_slice_with_several_models_as_mixed(other):
    tasks = [make_task("a"), make_task("b", baseline_model=other)]

    logged = cli_helpers.logged_models(tasks, PricingTable.load())

    assert logged == {("summarise", "v1"): "mixed"}


def test_announce_prints_the_batch_id(capsys):
    cli_helpers.announce("judge")("msgbatch_123")

    assert capsys.readouterr().out == "Submitted judge batch msgbatch_123; waiting for results.\n"


# --------------------------------------------------------------------------- tables


def test_estimate_table_lists_each_line_and_totals():
    cost = CostEstimate(lines=[_line(), _line(HAIKU, "judge")])

    text = _render(cli_helpers.estimate_table(cost))

    for expected in (SONNET, HAIKU, "candidate", "judge", "1,200", "2,400", "Total"):
        assert expected in text
    assert usd(cost.expected_usd) in text
    assert usd(cost.upper_bound_usd) in text
    assert "approximated" not in text


def test_estimate_table_marks_approximated_token_counts():
    cost = CostEstimate(lines=[_line(estimated=True), _line(HAIKU)])

    text = _render(cli_helpers.estimate_table(cost))

    assert f"{SONNET} *" in text
    assert f"{HAIKU} *" not in text
    assert "token counts partly approximated" in text


def test_simulation_table_shows_interval_errors_and_cost():
    result = _result(
        make_outcome("t1", det=True, projected="0.000123"),
        make_outcome("t2", status="errored"),
    )

    text = _render(cli_helpers.simulation_table(result))

    (row,) = result.overall
    assert row.errored == 1
    assert SONNET in text
    assert ci(row.pass_rate) in text  # brackets survive Rich markup
    assert usd_fine(row.projected_cost_per_task) in text


def test_simulation_table_without_graded_tasks_says_not_applicable():
    text = _render(cli_helpers.simulation_table(_result()))

    assert "n/a (n=0)" in text


# --------------------------------------------------------------------------- reports + gate


def test_write_reports_writes_markdown_and_html(tmp_path):
    outcomes = [make_outcome("t1", det=True)]

    paths = cli_helpers.write_reports(_result(*outcomes), outcomes, tmp_path)

    assert paths == tuple(tmp_path / name for name in REPORTS)
    markdown, html = (path.read_text(encoding="utf-8") for path in paths)
    assert markdown.startswith("# triage simulation report")
    assert "<h1>triage simulation report</h1>" in html


def test_below_pass_rate_compares_the_lower_bound_strictly():
    result = _result(
        make_outcome("t1", SONNET, det=True),
        make_outcome("t1", HAIKU, det=False),
        targets=(SONNET, HAIKU),
    )
    sonnet, haiku = result.overall

    assert cli_helpers.below_pass_rate(result, haiku.pass_rate.low) == []
    assert cli_helpers.below_pass_rate(result, sonnet.pass_rate.low) == [haiku]
    assert cli_helpers.below_pass_rate(result, 1.0) == [sonnet, haiku]


# --------------------------------------------------------------------------- confirmation prompt


@pytest.fixture
def workdir(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TRIAGE_PRICING", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _terminal(monkeypatch, *, interactive: bool) -> None:
    # CliRunner swaps sys.stdin during invoke, so replace the `sys` that cli.py looks at.
    stdin = SimpleNamespace(isatty=lambda: interactive)
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=stdin))


def _simulate(*args, answer: str | None = None):
    command = ["simulate", str(REPLAY), "-m", SONNET, "--no-judge", *args]
    return runner.invoke(cli.app, command, input=answer)


@pytest.mark.parametrize("answer", ["n\n", "\n", ""], ids=["no", "default", "eof"])
def test_declining_the_prompt_submits_nothing(monkeypatch, workdir, answer):
    attempts = forbid_client(monkeypatch)
    _terminal(monkeypatch, interactive=True)

    result = _simulate(answer=answer)

    assert result.exit_code == 1, result.output
    assert PROMPT in result.output
    assert attempts == []
    assert not any((workdir / name).exists() for name in REPORTS)


def test_accepting_the_prompt_runs_after_showing_the_estimate(monkeypatch, workdir):
    client = FakeAnthropic()
    install_client(monkeypatch, client)
    _terminal(monkeypatch, interactive=True)

    result = _simulate(answer="y\n")

    assert result.exit_code == 0, result.output
    assert result.output.index("Estimated Batch API spend") < result.output.index(PROMPT)
    assert len(client.batches.create_calls) == 1
    assert all((workdir / name).is_file() for name in REPORTS)


@pytest.mark.parametrize(
    ("interactive", "extra"),
    [(True, ["--yes"]), (True, ["-y"]), (False, [])],
    ids=["yes", "y", "not-a-terminal"],
)
def test_prompt_is_skipped(monkeypatch, workdir, interactive, extra):
    client = FakeAnthropic()
    install_client(monkeypatch, client)
    _terminal(monkeypatch, interactive=interactive)

    result = _simulate(*extra)

    assert result.exit_code == 0, result.output
    assert PROMPT not in result.output
    assert len(client.batches.create_calls) == 1


def test_dry_run_never_prompts(monkeypatch, workdir):
    attempts = forbid_client(monkeypatch)
    _terminal(monkeypatch, interactive=True)

    result = _simulate("--dry-run")

    assert result.exit_code == 0, result.output
    assert PROMPT not in result.output
    assert attempts == []
