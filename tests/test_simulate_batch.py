"""Batch replay (`simulate/batch.py`) and pre-flight spend (`simulate/estimate.py`)."""

import sys
from decimal import Decimal
from types import ModuleType

import pytest

from tests.simulate_fakes import (
    CUSTOM_ID_PATTERN,
    SAMPLING_PARAMS,
    FakeAnthropic,
    FakeBlock,
    FakeCacheCreation,
    FakeUsage,
    default_responder,
    errored,
    forbid_client,
    make_task,
    make_usage,
    succeeded,
    text_block,
    tool_block,
)
from tokenize_cli.models import UsageRecord
from tokenize_cli.pricing import PricingTable, UnknownModelError
from tokenize_cli.simulate import batch
from tokenize_cli.simulate.batch import DEFAULT_MAX_TOKENS, build_batch_requests, run_replay
from tokenize_cli.simulate.estimate import estimate_cost

SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"
TOOL = {"name": "get_weather", "input_schema": {"type": "object"}}


@pytest.fixture
def pricing() -> PricingTable:
    return PricingTable.load()


def _no_sleep(_: float) -> None:
    raise AssertionError("must not sleep when the batch has already ended")


def _by_key(outcomes):
    return {(o.task_id, o.model): o for o in outcomes}


# --------------------------------------------------------------------------- request building


@pytest.mark.parametrize("target", [SONNET, HAIKU, "claude-opus-4-5"])
def test_requests_never_send_sampling_params(pricing, target):
    tasks = [make_task("a", temperature=0.2), make_task("b", temperature=0.0)]

    requests = build_batch_requests(tasks, [target], pricing)

    assert len(requests) == 2
    for request in requests:
        assert SAMPLING_PARAMS.isdisjoint(request.params)


def test_submitted_batch_never_contains_sampling_params(pricing):
    client = FakeAnthropic()

    run_replay(client, [make_task(temperature=1.0)], [SONNET, HAIKU], pricing, sleep=_no_sleep)

    assert client.batches.requests
    for kwargs in client.batches.create_calls:
        assert SAMPLING_PARAMS.isdisjoint(kwargs)
        assert "extra_body" not in kwargs
    for request in client.batches.requests:
        assert SAMPLING_PARAMS.isdisjoint(request["params"])


def test_request_params_replay_the_logged_request(pricing):
    task = make_task(
        system="Be brief.",
        tools=[TOOL],
        tool_choice={"type": "auto"},
        max_tokens=321,
        temperature=0.7,
    )

    (request,) = build_batch_requests([task], [SONNET], pricing)

    assert request.task_id == "t1"
    assert request.model == SONNET
    assert request.params == {
        "model": SONNET,
        "max_tokens": 321,
        "system": "Be brief.",
        "messages": task.messages,
        "tools": [TOOL],
        "tool_choice": {"type": "auto"},
    }


def test_unset_fields_are_omitted_and_max_tokens_defaults(pricing):
    (request,) = build_batch_requests([make_task()], [SONNET], pricing)

    assert set(request.params) == {"model", "max_tokens", "messages"}
    assert request.params["max_tokens"] == DEFAULT_MAX_TOKENS > 0


def test_custom_ids_are_api_safe_and_unique(pricing):
    ids = ["feature/long id.with spaces", "feature.long id/with spaces", "x" * 200, "x" * 201]

    requests = build_batch_requests([make_task(i) for i in ids], [SONNET, HAIKU], pricing)

    custom_ids = [r.custom_id for r in requests]
    assert len(set(custom_ids)) == len(ids) * 2
    assert all(CUSTOM_ID_PATTERN.match(c) for c in custom_ids)
    assert {(r.task_id, r.model) for r in requests} == {
        (i, m) for i in ids for m in (SONNET, HAIKU)
    }


def test_targets_are_resolved_and_deduplicated(pricing):
    requests = build_batch_requests([make_task()], ["sonnet-5-5", SONNET], pricing)

    assert [r.params["model"] for r in requests] == [SONNET]


def test_unknown_target_fails_before_anything_is_submitted(pricing):
    client = FakeAnthropic()

    with pytest.raises(UnknownModelError):
        run_replay(client, [make_task()], [SONNET, "claude-nope"], pricing, sleep=_no_sleep)

    assert client.batches.create_calls == []


# --------------------------------------------------------------------------- running a batch


def test_results_are_matched_by_custom_id_not_position(pricing):
    def echo(custom_id, params):
        return succeeded(params["model"], text_block(params["messages"][0]["content"]))

    client = FakeAnthropic(echo)
    tasks = [make_task("a/1"), make_task("b/2")]

    outcomes = run_replay(client, tasks, [SONNET, HAIKU], pricing, sleep=_no_sleep)

    assert len(outcomes) == 4
    assert len(client.batches.create_calls) == 1
    for outcome in outcomes:
        assert outcome.response.text == f"Summarise item {outcome.task_id}."
        assert outcome.usage.model == outcome.model


def test_polls_until_the_batch_ends(pricing):
    client = FakeAnthropic(polls_until_ended=3)
    sleeps: list[float] = []

    outcomes = run_replay(
        client, [make_task()], [SONNET], pricing, sleep=sleeps.append, poll_interval=5.0
    )

    assert len(client.batches.retrieve_calls) == 3
    assert len(sleeps) >= 2
    assert set(sleeps) == {5.0}
    assert [o.status for o in outcomes] == ["succeeded"]


def test_no_tasks_submits_nothing(pricing):
    client = FakeAnthropic()

    assert run_replay(client, [], [SONNET], pricing, sleep=_no_sleep) == []
    assert client.batches.create_calls == []


def test_costs_come_from_response_usage(pricing):
    usage = FakeUsage(
        input_tokens=1000,
        output_tokens=200,
        cache_creation_input_tokens=700,
        cache_read_input_tokens=5000,
        cache_creation=FakeCacheCreation(
            ephemeral_5m_input_tokens=300, ephemeral_1h_input_tokens=400
        ),
    )
    client = FakeAnthropic(lambda _, p: succeeded(p["model"], text_block("ok"), usage=usage))
    expected = UsageRecord(
        model=SONNET,
        input_tokens=1000,
        output_tokens=200,
        cache_creation_5m_input_tokens=300,
        cache_creation_1h_input_tokens=400,
        cache_read_input_tokens=5000,
    )

    (outcome,) = run_replay(client, [make_task()], [SONNET], pricing, sleep=_no_sleep)

    batched = expected.model_copy(update={"is_batch": True})
    assert outcome.sim_cost == pricing.cost(batched).total
    assert outcome.projected_cost == pricing.cost(expected).total
    assert 0 < outcome.sim_cost < outcome.projected_cost
    assert outcome.usage.is_batch is True
    assert outcome.usage.cache_creation_1h_input_tokens == 400


def test_null_cache_fields_count_as_zero(pricing):
    usage = FakeUsage(cache_creation_input_tokens=None, cache_read_input_tokens=None)
    client = FakeAnthropic(lambda _, p: succeeded(p["model"], usage=usage))

    (outcome,) = run_replay(client, [make_task()], [SONNET], pricing, sleep=_no_sleep)

    assert outcome.usage.cache_read_input_tokens == 0
    assert outcome.usage.cache_creation_5m_input_tokens == 0


@pytest.mark.parametrize("kind", ["errored", "expired", "canceled"])
def test_unsuccessful_results_carry_no_response_or_cost(pricing, kind):
    def respond(custom_id, params):
        result = errored()
        return result.model_copy(update={"type": kind})

    client = FakeAnthropic(respond)

    (outcome,) = run_replay(client, [make_task()], [SONNET], pricing, sleep=_no_sleep)

    assert outcome.status == kind
    assert outcome.response is None
    assert outcome.passed is None
    assert outcome.sim_cost == outcome.projected_cost == Decimal(0)


def test_a_missing_result_is_reported_as_errored(pricing):
    def respond(custom_id, params):
        return None if params["model"] == HAIKU else default_responder(custom_id, params)

    client = FakeAnthropic(respond)

    outcomes = _by_key(run_replay(client, [make_task()], [SONNET, HAIKU], pricing, sleep=_no_sleep))

    assert outcomes[("t1", SONNET)].status == "succeeded"
    assert outcomes[("t1", HAIKU)].status == "errored"


def test_outcomes_are_graded_deterministically(pricing):
    task = make_task(
        "tool",
        feature="weather",
        tools=[TOOL],
        baseline_response={"tool_calls": [{"name": "get_weather", "input": {"city": "Oslo"}}]},
    )

    def respond(custom_id, params):
        city = "Oslo" if params["model"] == SONNET else "Paris"
        return succeeded(
            params["model"],
            FakeBlock(type="thinking", thinking="", signature="sig"),
            tool_block("get_weather", {"city": city}),
            stop_reason="tool_use",
        )

    outcomes = _by_key(
        run_replay(FakeAnthropic(respond), [task], [SONNET, HAIKU], pricing, sleep=_no_sleep)
    )

    good, bad = outcomes[("tool", SONNET)], outcomes[("tool", HAIKU)]
    assert good.det.passed is True
    assert bad.det.passed is False
    assert good.response.stop_reason == "tool_use"
    assert good.response.text == ""
    assert (good.feature, good.prompt_version) == ("weather", "v1")


# --------------------------------------------------------------------------- client factory


def test_create_client_imports_the_sdk_lazily_and_hardcodes_no_key(monkeypatch):
    calls: list[dict] = []

    class Client:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

    module = ModuleType("anthropic")
    module.Anthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", module)

    assert isinstance(batch.create_client(), Client)
    assert len(calls) == 1
    assert "api_key" not in calls[0]


# --------------------------------------------------------------------------- estimate


def _batch_cost(pricing, model, input_tokens, output_tokens) -> Decimal:
    record = UsageRecord(
        model=model, input_tokens=input_tokens, output_tokens=output_tokens, is_batch=True
    )
    return pricing.cost(record).total


def test_estimate_prices_baseline_usage_at_batch_rates(pricing):
    tasks = [
        make_task("a", baseline_usage=make_usage(input_tokens=900, output_tokens=40)),
        make_task(
            "b",
            max_tokens=500,
            baseline_usage=make_usage(
                input_tokens=100, cache_read_input_tokens=300, output_tokens=60
            ),
        ),
    ]

    estimate = estimate_cost(tasks, [SONNET], pricing)

    (line,) = estimate.lines
    assert (line.model, line.role, line.requests) == (SONNET, "candidate", 2)
    assert line.estimated_tokens is False
    assert line.input_tokens == 900 + 400
    assert line.output_tokens == 40 + 60
    expected = _batch_cost(pricing, SONNET, 900, 40) + _batch_cost(pricing, SONNET, 400, 60)
    upper = _batch_cost(pricing, SONNET, 900, DEFAULT_MAX_TOKENS) + _batch_cost(
        pricing, SONNET, 400, 500
    )
    assert estimate.expected_usd == expected
    assert estimate.upper_bound_usd == upper
    full_price = pricing.cost(UsageRecord(model=SONNET, input_tokens=1300, output_tokens=100))
    assert estimate.expected_usd == full_price.total * pricing.config.batch_discount


def test_estimate_goes_through_pricing_table_cost_with_is_batch(pricing, monkeypatch):
    records: list[UsageRecord] = []
    original = PricingTable.cost

    def spy(self, record, **kwargs):
        records.append(record)
        return original(self, record, **kwargs)

    monkeypatch.setattr(PricingTable, "cost", spy)

    estimate_cost([make_task()], [SONNET, HAIKU], pricing, judge_model=SONNET)

    assert records
    assert all(record.is_batch for record in records)


def test_estimate_has_no_hardcoded_rates(tmp_path):
    config = tmp_path / "pricing.yaml"
    config.write_text(
        """
version: 1
batch_discount: 0.25
cache_multipliers: {write_5m: 1.25, write_1h: 2.0, read: 0.1}
inference_geo_multipliers: {global: 1.0}
models:
  alpha: {input: 7.0, output: 11.0}
""",
        encoding="utf-8",
    )
    task = make_task(
        baseline_model="alpha",
        max_tokens=500,
        baseline_usage=make_usage("alpha", input_tokens=1000, output_tokens=100),
    )

    estimate = estimate_cost([task], ["alpha"], PricingTable.load(config))

    assert estimate.expected_usd == Decimal("0.002025")  # (1000*7 + 100*11) / 1e6 * 0.25
    assert estimate.upper_bound_usd == Decimal("0.003125")  # (1000*7 + 500*11) / 1e6 * 0.25


def test_estimate_selects_price_tiers_per_request_not_per_total(pricing):
    def task(task_id, tokens):
        return make_task(task_id, baseline_usage=make_usage(input_tokens=tokens, output_tokens=10))

    below = estimate_cost([task("a", 60_000), task("b", 60_000)], [HAIKU], pricing)
    above = estimate_cost([task("c", 150_000)], [HAIKU], pricing)

    assert below.expected_usd == 2 * _batch_cost(pricing, HAIKU, 60_000, 10)
    assert above.expected_usd == _batch_cost(pricing, HAIKU, 150_000, 10)


def test_estimate_upper_bound_is_never_below_expected(pricing):
    task = make_task(max_tokens=10, baseline_usage=make_usage(output_tokens=900))

    estimate = estimate_cost([task], [SONNET], pricing)

    assert estimate.upper_bound_usd >= estimate.expected_usd


def test_estimate_falls_back_to_character_counts(pricing):
    short = estimate_cost([make_task()], [SONNET], pricing).lines[0]
    long_task = make_task(system="s" * 4000, messages=[{"role": "user", "content": "word " * 2000}])
    long = estimate_cost([long_task], [SONNET], pricing).lines[0]

    assert short.estimated_tokens is True
    assert long.estimated_tokens is True
    assert 0 < short.input_tokens < long.input_tokens
    assert short.output_tokens > 0


def test_estimate_has_one_line_per_target_plus_one_for_the_judge(pricing):
    tasks = [make_task("a"), make_task("b")]

    estimate = estimate_cost(tasks, ["sonnet-5-5", HAIKU, SONNET], pricing, judge_model="opus-5-5")

    assert [(x.model, x.role, x.requests) for x in estimate.lines] == [
        (SONNET, "candidate", 2),
        (HAIKU, "candidate", 2),
        ("claude-opus-5-5", "judge", 4),
    ]
    judge = estimate.lines[-1]
    assert judge.estimated_tokens is True  # candidate responses don't exist yet
    assert judge.input_tokens > 0
    assert 0 < judge.expected_usd <= judge.upper_bound_usd
    assert estimate.expected_usd == sum(x.expected_usd for x in estimate.lines)


def test_estimate_without_a_judge_has_no_judge_line(pricing):
    estimate = estimate_cost([make_task()], [SONNET], pricing, judge_model=None)

    assert {line.role for line in estimate.lines} == {"candidate"}


def test_estimate_needs_no_api_key_or_client(pricing, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    attempts = forbid_client(monkeypatch)

    estimate = estimate_cost([make_task()], [SONNET], pricing, judge_model=SONNET)

    assert estimate.expected_usd > 0
    assert attempts == []
