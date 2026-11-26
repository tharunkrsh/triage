"""Milestone 4A: simulate schemas — response normalisation, pass rules, Decimal-only USD."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from tokenize_cli.simulate.models import (
    CandidateOutcome,
    CandidateSummary,
    CheckResult,
    CostEstimate,
    CostLine,
    DeterministicResult,
    JudgeVerdict,
    ReplayTask,
    ResponseContent,
    ToolCall,
)
from tokenize_cli.simulate.stats import wilson_interval

# --------------------------------------------------------------------------- ResponseContent


def test_from_api_plain_string():
    r = ResponseContent.from_api("hello")
    assert (r.text, r.tool_calls, r.stop_reason) == ("hello", [], None)


def test_from_api_content_blocks():
    r = ResponseContent.from_api(
        [
            {"type": "thinking", "thinking": "hmm", "signature": "s"},
            {"type": "text", "text": '{"a": '},
            {"type": "text", "text": "1}"},
            {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"x": 1}},
        ]
    )
    assert r.text == '{"a": 1}'
    assert r.tool_calls == [ToolCall(name="f", input={"x": 1})]


def test_from_api_full_message():
    r = ResponseContent.from_api(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
        }
    )
    assert (r.text, r.stop_reason) == ("hi", "end_turn")


def test_from_api_rejects_garbage():
    with pytest.raises(ValueError):
        ResponseContent.from_api({"no_content": True})
    with pytest.raises(ValueError):
        ResponseContent.from_api(42)


# --------------------------------------------------------------------------- ReplayTask


def _task(**kw):
    base = {
        "task_id": "t",
        "baseline_model": "claude-opus-5-5",
        "messages": [{"role": "user", "content": "hi"}],
        "baseline_response": ResponseContent(text="ok"),
    }
    return ReplayTask(**{**base, **kw})


def test_replay_task_defaults():
    t = _task()
    assert t.feature is None and t.prompt_version is None
    assert t.tools is None and t.expected_schema is None and t.baseline_usage is None
    assert t.temperature is None and t.max_tokens is None


def test_replay_task_keeps_temperature_metadata():
    assert _task(temperature=0.3).temperature == 0.3


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "x", "name": "f", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "x", "content": "r"}],
            },
        ],
    ],
)
def test_replay_task_rejects_non_single_turn(messages):
    with pytest.raises(ValidationError):
        _task(messages=messages)


def test_replay_task_allows_few_shot_history():
    msgs = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "3+3?"},
    ]
    assert len(_task(messages=msgs).messages) == 3


# --------------------------------------------------------------------------- pass rules


def _det(*passed):
    return DeterministicResult(checks=[CheckResult(name="json_parse", passed=p) for p in passed])


def test_deterministic_result_aggregation():
    assert _det().passed is None and not _det().applicable
    assert _det(True, True).passed is True
    assert _det(True, False).passed is False


def _outcome(**kw):
    return CandidateOutcome(task_id="t", model="claude-haiku-5-5", status="succeeded", **kw)


YES = JudgeVerdict(passed=True, score=5, rationale="same")
NO = JudgeVerdict(passed=False, score=1, rationale="different")


@pytest.mark.parametrize(
    ("status", "det", "judge", "expected"),
    [
        ("errored", None, None, None),
        ("expired", _det(True), YES, None),
        ("canceled", None, None, None),
        ("succeeded", None, None, None),  # nothing graded it
        ("succeeded", _det(), None, None),
        ("succeeded", _det(True), None, True),
        ("succeeded", _det(False), None, False),
        ("succeeded", None, YES, True),
        ("succeeded", None, NO, False),
        ("succeeded", _det(True), YES, True),
        ("succeeded", _det(True), NO, False),
        ("succeeded", _det(False), YES, False),
        ("succeeded", _det(), YES, True),
    ],
)
def test_outcome_passed_truth_table(status, det, judge, expected):
    assert _outcome(status=status, det=det, judge=judge).passed is expected


def test_judge_score_bounds():
    with pytest.raises(ValidationError):
        JudgeVerdict(passed=True, score=0, rationale="")
    with pytest.raises(ValidationError):
        JudgeVerdict(passed=True, score=6, rationale="")


# --------------------------------------------------------------------------- Decimal-only USD


def test_usd_accepts_decimal_str_and_int():
    o = _outcome(sim_cost=Decimal("0.10"), projected_cost="0.2")
    assert o.sim_cost == Decimal("0.10") and o.projected_cost == Decimal("0.2")
    assert isinstance(_outcome(sim_cost=1).sim_cost, Decimal)
    assert _outcome().sim_cost == Decimal(0)


@pytest.mark.parametrize("field", ["sim_cost", "projected_cost"])
def test_usd_rejects_float(field):
    with pytest.raises(ValidationError, match="float"):
        _outcome(**{field: 0.1})


def test_usd_rejects_negative():
    with pytest.raises(ValidationError):
        _outcome(sim_cost=Decimal("-0.01"))


def test_summary_usd_fields_reject_float():
    kw = {
        "feature": None,
        "prompt_version": None,
        "model": "m",
        "n": 1,
        "passes": 1,
        "pass_rate": wilson_interval(1, 1),
        "sim_cost": Decimal("1"),
        "projected_cost_per_task": Decimal("1"),
        "cost_per_success": Decimal("1"),
    }
    CandidateSummary(**kw)
    for field in ("sim_cost", "projected_cost_per_task", "cost_per_success"):
        with pytest.raises(ValidationError):
            CandidateSummary(**{**kw, field: 1.0})


def test_cost_estimate_totals_are_decimal_sums():
    lines = [
        CostLine(
            model="claude-haiku-5-5",
            role="candidate",
            requests=2,
            input_tokens=10,
            output_tokens=5,
            expected_usd=Decimal("0.10"),
            upper_bound_usd=Decimal("0.30"),
            estimated_tokens=False,
        ),
        CostLine(
            model="claude-sonnet-5-5",
            role="judge",
            requests=2,
            input_tokens=10,
            output_tokens=5,
            expected_usd=Decimal("0.20"),
            upper_bound_usd=Decimal("0.40"),
            estimated_tokens=True,
        ),
    ]
    est = CostEstimate(lines=lines)
    assert est.expected_usd == Decimal("0.30")
    assert est.upper_bound_usd == Decimal("0.70")
    assert CostEstimate(lines=[]).expected_usd == Decimal(0)
    with pytest.raises(ValidationError):
        CostLine(**{**lines[0].model_dump(), "expected_usd": 0.1})
