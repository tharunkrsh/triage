"""LLM judge (`simulate/eval_judge.py`), run through the Batch API."""

import json
from decimal import Decimal

import pytest

from tests.simulate_fakes import (
    CUSTOM_ID_PATTERN,
    SAMPLING_PARAMS,
    FakeAnthropic,
    FakeUsage,
    errored,
    make_outcome,
    make_task,
    succeeded,
    text_block,
    verdict_json,
)
from tokenize_cli.models import UsageRecord
from tokenize_cli.pricing import PricingTable
from tokenize_cli.simulate.eval_judge import (
    JUDGE_SCHEMA,
    build_judge_request,
    parse_verdict,
    run_judge,
)
from tokenize_cli.simulate.models import ResponseContent

JUDGE = "claude-sonnet-5-5"
CANDIDATE = "claude-haiku-5-5"


@pytest.fixture
def pricing() -> PricingTable:
    return PricingTable.load()


def _no_sleep(_: float) -> None:
    raise AssertionError("must not sleep when the batch has already ended")


def _replies(text: str, usage: FakeUsage | None = None):
    return lambda _, params: succeeded(params["model"], text_block(text), usage=usage)


# --------------------------------------------------------------------------- request


@pytest.mark.parametrize("judge_model", [JUDGE, "claude-opus-5-5", "claude-sonnet-4-5"])
def test_judge_request_never_sends_sampling_params(pricing, judge_model):
    request = build_judge_request(make_task(temperature=0.9), make_outcome(), judge_model, pricing)

    assert SAMPLING_PARAMS.isdisjoint(request.params)


def test_submitted_judge_batch_never_contains_sampling_params(pricing):
    client = FakeAnthropic()

    run_judge(
        client, [make_task(temperature=0.5)], [make_outcome()], JUDGE, pricing, sleep=_no_sleep
    )

    (kwargs,) = client.batches.create_calls
    assert SAMPLING_PARAMS.isdisjoint(kwargs)
    assert "extra_body" not in kwargs
    (request,) = client.batches.requests
    assert SAMPLING_PARAMS.isdisjoint(request["params"])


def test_judge_request_uses_structured_output_current_models_accept(pricing):
    request = build_judge_request(make_task(), make_outcome(), "sonnet-5-5", pricing)
    params = request.params

    assert request.model == params["model"] == JUDGE
    assert CUSTOM_ID_PATTERN.match(request.custom_id)
    assert params["max_tokens"] > 0
    assert params["output_config"]["format"] == {"type": "json_schema", "schema": JUDGE_SCHEMA}
    # Forced tool use, disabled thinking and assistant prefill are 400s on current models.
    assert "tool_choice" not in params
    assert "thinking" not in params
    assert params["messages"][-1]["role"] == "user"


def test_judge_schema_is_strict_and_uses_supported_keywords():
    assert JUDGE_SCHEMA["type"] == "object"
    assert JUDGE_SCHEMA["additionalProperties"] is False
    assert set(JUDGE_SCHEMA["required"]) == {"passed", "score", "rationale"}
    assert JUDGE_SCHEMA["properties"]["score"]["enum"] == [1, 2, 3, 4, 5]
    # Structured outputs don't support numeric range keywords.
    assert not {"minimum", "maximum"} & set(JUDGE_SCHEMA["properties"]["score"])


def test_judge_prompt_shows_the_task_baseline_and_candidate(pricing):
    task = make_task(
        system="SYSTEM-MARKER",
        messages=[{"role": "user", "content": "QUESTION-MARKER"}],
        baseline_response=ResponseContent(text="BASELINE-MARKER"),
    )

    request = build_judge_request(task, make_outcome(text="CANDIDATE-MARKER"), JUDGE, pricing)

    prompt = json.dumps(request.params["messages"])
    for marker in ("SYSTEM-MARKER", "QUESTION-MARKER", "BASELINE-MARKER", "CANDIDATE-MARKER"):
        assert marker in prompt


def test_judge_prompt_includes_tool_calls(pricing):
    task = make_task(baseline_response={"tool_calls": [{"name": "baseline_tool", "input": {}}]})
    outcome = make_outcome(
        response=ResponseContent(tool_calls=[{"name": "candidate_tool", "input": {}}])
    )

    prompt = json.dumps(build_judge_request(task, outcome, JUDGE, pricing).params["messages"])

    assert "baseline_tool" in prompt
    assert "candidate_tool" in prompt


# --------------------------------------------------------------------------- parsing


def test_parse_verdict_reads_schema_conforming_json():
    verdict = parse_verdict(verdict_json(passed=False, score=2, rationale="wrong number"))

    assert (verdict.passed, verdict.score, verdict.rationale) == (False, 2, "wrong number")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I think it passes.",
        "[]",
        '{"passed": true}',
        '{"passed": true, "score": 9, "rationale": "x"}',
        '{"passed": true, "score": 4, "rationale": "x", "extra": 1}',
    ],
)
def test_parse_verdict_returns_none_for_unusable_output(text):
    assert parse_verdict(text) is None


# --------------------------------------------------------------------------- running


def test_verdicts_are_matched_to_their_own_outcome(pricing):
    def respond(_, params):
        passed = "MARK-PASS" in json.dumps(params["messages"])
        text = verdict_json(passed=passed, score=5 if passed else 1)
        return succeeded(params["model"], text_block(text))

    outcomes = [
        make_outcome("t1", text="MARK-PASS"),
        make_outcome("t2", text="MARK-FAIL"),
        make_outcome("t1", "claude-haiku-5-5", text="MARK-FAIL"),
    ]
    client = FakeAnthropic(respond)

    run = run_judge(
        client, [make_task("t1"), make_task("t2")], outcomes, JUDGE, pricing, sleep=_no_sleep
    )

    assert len(client.batches.create_calls) == 1
    assert len({r["custom_id"] for r in client.batches.requests}) == 3
    judged = {(o.task_id, o.model): o.judge.passed for o in run.outcomes}
    assert judged == {
        ("t1", "claude-sonnet-5-5"): True,
        ("t2", "claude-sonnet-5-5"): False,
        ("t1", "claude-haiku-5-5"): False,
    }


def test_judging_preserves_order_and_other_fields(pricing):
    outcomes = [
        make_outcome("t1", det=True, sim="0.001", projected="0.002"),
        make_outcome("t2", status="errored"),
    ]

    run = run_judge(
        FakeAnthropic(),
        [make_task("t1"), make_task("t2")],
        outcomes,
        JUDGE,
        pricing,
        sleep=_no_sleep,
    )

    assert [o.task_id for o in run.outcomes] == ["t1", "t2"]
    assert run.outcomes[0].model_copy(update={"judge": None}) == outcomes[0]
    assert run.outcomes[0].judge.passed is True
    assert run.outcomes[1] == outcomes[1]


def test_only_successful_replays_are_sent_to_the_judge(pricing):
    client = FakeAnthropic()
    outcomes = [make_outcome("t1"), make_outcome("t2", status="expired")]

    run_judge(client, [make_task("t1"), make_task("t2")], outcomes, JUDGE, pricing, sleep=_no_sleep)

    assert len(client.batches.requests) == 1


def test_nothing_to_judge_submits_no_batch(pricing):
    client = FakeAnthropic()
    outcomes = [make_outcome(status="errored")]

    run = run_judge(client, [make_task()], outcomes, JUDGE, pricing, sleep=_no_sleep)

    assert client.batches.create_calls == []
    assert run.outcomes == outcomes
    assert run.cost == Decimal(0)


def test_judge_cost_is_batch_priced_from_usage(pricing):
    usage = FakeUsage(input_tokens=2000, output_tokens=300)
    client = FakeAnthropic(_replies(verdict_json(), usage))

    run = run_judge(
        client,
        [make_task("t1"), make_task("t2")],
        [make_outcome("t1"), make_outcome("t2")],
        JUDGE,
        pricing,
        sleep=_no_sleep,
    )

    one = UsageRecord(model=JUDGE, input_tokens=2000, output_tokens=300, is_batch=True)
    assert run.cost == 2 * pricing.cost(one).total


def test_unparseable_verdict_leaves_outcome_unjudged_but_still_costs(pricing):
    client = FakeAnthropic(_replies("not json"))

    run = run_judge(client, [make_task()], [make_outcome()], JUDGE, pricing, sleep=_no_sleep)

    assert run.outcomes[0].judge is None
    assert run.outcomes[0].passed is None
    assert run.cost > 0


def test_refused_verdict_is_not_trusted(pricing):
    client = FakeAnthropic(
        lambda _, p: succeeded(p["model"], text_block(verdict_json()), stop_reason="refusal")
    )

    run = run_judge(client, [make_task()], [make_outcome()], JUDGE, pricing, sleep=_no_sleep)

    assert run.outcomes[0].judge is None


@pytest.mark.parametrize("respond", [lambda *_: errored(), lambda *_: None])
def test_failed_or_missing_judge_result_leaves_outcome_unjudged(pricing, respond):
    run = run_judge(
        FakeAnthropic(respond),
        [make_task()],
        [make_outcome(det=True)],
        JUDGE,
        pricing,
        sleep=_no_sleep,
    )

    assert run.outcomes[0].judge is None
    assert run.outcomes[0].passed is True  # deterministic grade survives
    assert run.cost == Decimal(0)


def test_judge_polls_until_the_batch_ends(pricing):
    client = FakeAnthropic(polls_until_ended=2)
    sleeps: list[float] = []

    run = run_judge(
        client,
        [make_task()],
        [make_outcome()],
        JUDGE,
        pricing,
        sleep=sleeps.append,
        poll_interval=3.0,
    )

    assert len(client.batches.retrieve_calls) == 2
    assert set(sleeps) == {3.0}
    assert run.outcomes[0].judge is not None
