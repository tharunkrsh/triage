"""`triage simulate` end to end against the fake Batch API.

Seam contract: cli.py calls `estimate.estimate_cost`, `summarize.summarize_simulation` and
`batch.create_client` through their modules, so they can be observed or replaced here.
"""

from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.simulate_fakes import (
    SAMPLING_PARAMS,
    FakeAnthropic,
    errored,
    fixture_perfect_responder,
    forbid_client,
    install_client,
    is_judge_request,
)
from triage.cli import app
from triage.pricing import PricingTable
from triage.report.builder import usd
from triage.simulate import estimate, summarize
from triage.simulate.stats import wilson_interval

REPLAY = Path(__file__).parent / "fixtures" / "simulate" / "replay.jsonl"
SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"
VALID_TASKS = 4  # replay.jsonl also holds invalid rows
REPORTS = ("triage-simulate.md", "triage-simulate.html")

runner = CliRunner()


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TRIAGE_PRICING", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def estimates(monkeypatch):
    """Record each estimate_cost() call as (kwargs, returned estimate)."""
    calls = []
    original = estimate.estimate_cost

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((kwargs, result))
        return result

    monkeypatch.setattr(estimate, "estimate_cost", spy)
    return calls


def _client(monkeypatch, responder=fixture_perfect_responder) -> FakeAnthropic:
    client = FakeAnthropic(responder)
    install_client(monkeypatch, client)
    return client


def _simulate(*args, replay: Path = REPLAY):
    return runner.invoke(app, ["simulate", str(replay), *args])


def _reports_exist(directory: Path) -> bool:
    return all((directory / name).is_file() for name in REPORTS)


# --------------------------------------------------------------------------- dry run


def test_dry_run_needs_no_key_and_never_constructs_a_client(monkeypatch, tmp_path, estimates):
    attempts = forbid_client(monkeypatch)

    result = _simulate("-m", SONNET, "-m", HAIKU, "--dry-run")

    assert result.exit_code == 0, result.output
    assert attempts == []
    ((_, cost),) = estimates
    assert [line.requests for line in cost.lines] == [VALID_TASKS, VALID_TASKS, VALID_TASKS * 2]
    assert usd(cost.expected_usd) in result.output
    assert usd(cost.upper_bound_usd) in result.output
    assert "estimate" in result.output.lower()
    assert not any((tmp_path / name).exists() for name in REPORTS)


def test_dry_run_reports_skipped_rows(monkeypatch):
    forbid_client(monkeypatch)

    result = _simulate("-m", SONNET, "--dry-run")

    assert "warning" in result.output.lower()


def test_dry_run_does_not_apply_the_pass_rate_gate(monkeypatch):
    forbid_client(monkeypatch)

    result = _simulate("-m", SONNET, "--dry-run", "--min-pass-rate", "0.99")

    assert result.exit_code == 0, result.output


# --------------------------------------------------------------------------- judge model


def test_judge_model_is_resolved_before_estimating(monkeypatch, estimates):
    forbid_client(monkeypatch)

    result = _simulate("-m", SONNET, "--judge-model", "opus-5-5", "--dry-run")

    assert result.exit_code == 0, result.output
    assert estimates[0][0]["judge_model"] == "claude-opus-5-5"


def test_default_judge_model_is_a_priced_model(monkeypatch, estimates):
    forbid_client(monkeypatch)

    result = _simulate("-m", SONNET, "--dry-run")

    assert result.exit_code == 0, result.output
    assert estimates[0][0]["judge_model"] in PricingTable.load().config.models


@pytest.mark.parametrize("extra", [["--dry-run"], []], ids=["dry-run", "paid"])
def test_unknown_judge_model_fails_upfront(monkeypatch, tmp_path, extra):
    attempts = forbid_client(monkeypatch)

    result = _simulate("-m", SONNET, "--judge-model", "claude-nope", *extra)

    assert result.exit_code == 1, result.output
    assert "claude-nope" in result.output
    assert attempts == []
    assert not any((tmp_path / name).exists() for name in REPORTS)


@pytest.mark.parametrize("extra", [[], ["--judge-model", "claude-nope"]])
def test_no_judge_passes_none_and_ignores_the_judge_model(monkeypatch, estimates, extra):
    forbid_client(monkeypatch)

    result = _simulate("-m", SONNET, "--no-judge", "--dry-run", *extra)

    assert result.exit_code == 0, result.output
    kwargs, cost = estimates[0]
    assert kwargs["judge_model"] is None
    assert {line.role for line in cost.lines} == {"candidate"}


def test_no_judge_submits_only_the_candidate_batch(monkeypatch):
    client = _client(monkeypatch)

    result = _simulate("-m", SONNET, "--no-judge")

    assert result.exit_code == 0, result.output
    assert len(client.batches.create_calls) == 1
    assert not any(is_judge_request(r["params"]) for r in client.batches.requests)


# --------------------------------------------------------------------------- other input errors


@pytest.mark.parametrize("extra", [["--dry-run"], []], ids=["dry-run", "paid"])
def test_unknown_target_model_fails_upfront(monkeypatch, extra):
    attempts = forbid_client(monkeypatch)

    result = _simulate("-m", "claude-nope", *extra)

    assert result.exit_code == 1, result.output
    assert "claude-nope" in result.output
    assert attempts == []


@pytest.mark.parametrize("content", [None, ""], ids=["missing", "empty"])
def test_unusable_replay_file_fails_without_a_client(monkeypatch, tmp_path, content):
    attempts = forbid_client(monkeypatch)
    path = tmp_path / "input.jsonl"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    result = _simulate("-m", SONNET, replay=path)

    assert result.exit_code == 1, result.output
    assert attempts == []


def test_a_target_model_is_required():
    assert _simulate("--dry-run").exit_code == 2


@pytest.mark.parametrize("value", ["-0.1", "1.5", "75"])
def test_min_pass_rate_must_be_a_proportion(monkeypatch, value):
    attempts = forbid_client(monkeypatch)

    result = _simulate("-m", SONNET, "--min-pass-rate", value)

    assert result.exit_code == 2
    assert attempts == []


# --------------------------------------------------------------------------- full run


def test_run_shows_the_estimate_before_spending(monkeypatch, estimates):
    client = _client(monkeypatch)

    def refuse(**kwargs):
        raise RuntimeError("stop at submission")

    monkeypatch.setattr(client.batches, "create", refuse)

    result = _simulate("-m", SONNET)

    assert result.exit_code != 0
    assert usd(estimates[0][1].expected_usd) in result.output
    assert usd(estimates[0][1].upper_bound_usd) in result.output


def test_run_replays_judges_and_writes_reports(monkeypatch, tmp_path):
    client = _client(monkeypatch)
    out = tmp_path / "nested" / "out"

    result = _simulate("-m", "sonnet-5-5", "-m", HAIKU, "-o", str(out))

    assert result.exit_code == 0, result.output
    assert len(client.batches.create_calls) == 2  # candidates, then judge
    judged = [r for r in client.batches.requests if is_judge_request(r["params"])]
    assert len(judged) == VALID_TASKS * 2
    assert len(client.batches.requests) == VALID_TASKS * 4
    for request in client.batches.requests:
        assert SAMPLING_PARAMS.isdisjoint(request["params"])
    assert _reports_exist(out)
    markdown = (out / REPORTS[0]).read_text(encoding="utf-8")
    html = (out / REPORTS[1]).read_text(encoding="utf-8")
    for report in (markdown, html):
        assert SONNET in report
        assert HAIKU in report
        assert REPLAY.name in report
    assert "<svg" in html


def test_reports_default_to_the_working_directory(monkeypatch, tmp_path):
    _client(monkeypatch)

    result = _simulate("-m", SONNET, "--no-judge")

    assert result.exit_code == 0, result.output
    assert _reports_exist(tmp_path)


# --------------------------------------------------------------------------- pass-rate gate

# With --no-judge only the two deterministically checked fixture tasks are graded.
_LOW_2_OF_2 = wilson_interval(2, 2).low


@pytest.mark.parametrize(
    ("minimum", "code"),
    [
        (None, 0),
        (0.0, 0),
        (_LOW_2_OF_2, 0),  # equal does not exceed
        (_LOW_2_OF_2 + 0.001, 1),
        (1.0, 1),  # a perfect point estimate is not enough: the lower bound gates
    ],
)
def test_min_pass_rate_gates_on_the_wilson_lower_bound(monkeypatch, tmp_path, minimum, code):
    _client(monkeypatch)
    args = [] if minimum is None else ["--min-pass-rate", repr(minimum)]

    result = _simulate("-m", SONNET, "--no-judge", *args)

    assert result.exit_code == code, result.output
    assert _reports_exist(tmp_path)


def test_gate_fails_if_any_target_is_below_and_still_writes_reports(monkeypatch, tmp_path):
    def respond(custom_id, params):
        if params["model"] == HAIKU:
            return errored()
        return fixture_perfect_responder(custom_id, params)

    _client(monkeypatch, respond)

    result = _simulate("-m", SONNET, "-m", HAIKU, "--no-judge", "--min-pass-rate", "0.1")

    assert result.exit_code == 1, result.output
    assert HAIKU in result.output
    assert _reports_exist(tmp_path)


def test_judged_run_gates_on_all_graded_tasks(monkeypatch):
    _client(monkeypatch)
    minimum = (_LOW_2_OF_2 + wilson_interval(4, 4).low) / 2  # needs all four graded

    result = _simulate("-m", SONNET, "--min-pass-rate", str(minimum))

    assert result.exit_code == 0, result.output


def test_total_spend_in_the_report_includes_the_judge(monkeypatch, tmp_path):
    _client(monkeypatch)
    captured = []
    original = summarize.summarize_simulation

    def spy(*args, **kwargs):
        captured.append(original(*args, **kwargs))
        return captured[-1]

    monkeypatch.setattr(summarize, "summarize_simulation", spy)

    result = _simulate("-m", SONNET)

    assert result.exit_code == 0, result.output
    (summary,) = captured
    assert summary.judge_model is not None
    assert summary.total_sim_cost > sum((s.sim_cost for s in summary.overall), Decimal(0))
