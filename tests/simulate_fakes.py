"""Offline stand-ins for the Anthropic SDK plus builders for simulate tests.

Response objects are pydantic models, like the SDK's, so code under test may use attribute
access or `model_dump()`. Nothing here imports `anthropic` or touches the network.

Seam contract: production code builds its client only through
`triage.simulate.batch.create_client`, looked up on the module at call time, and
`create_client` imports `anthropic` lazily.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from decimal import Decimal
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from triage.models import UsageRecord
from triage.simulate.models import (
    CandidateOutcome,
    DeterministicResult,
    JudgeVerdict,
    ReplayTask,
    ResponseContent,
)

SAMPLING_PARAMS = frozenset({"temperature", "top_p", "top_k"})
CUSTOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# --------------------------------------------------------------------------- SDK shapes


class _Fake(BaseModel):
    model_config = ConfigDict(extra="allow")


class FakeBlock(_Fake):
    type: str


class FakeCacheCreation(_Fake):
    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0


class FakeUsage(_Fake):
    input_tokens: int = 100
    output_tokens: int = 20
    cache_creation_input_tokens: int | None = 0
    cache_read_input_tokens: int | None = 0
    cache_creation: FakeCacheCreation | None = None
    service_tier: str | None = "batch"
    inference_geo: str | None = "not_available"


class FakeMessage(_Fake):
    id: str = "msg_fake"
    type: str = "message"
    role: str = "assistant"
    model: str
    content: list[FakeBlock]
    stop_reason: str | None = "end_turn"
    usage: FakeUsage


class FakeError(_Fake):
    type: str = "invalid_request"
    message: str = "bad request"


class FakeResult(_Fake):
    type: str
    message: FakeMessage | None = None
    error: FakeError | None = None


class FakeEntry(_Fake):
    custom_id: str
    result: FakeResult


class FakeBatch(_Fake):
    id: str
    processing_status: str


# --------------------------------------------------------------------------- result builders


def text_block(text: str) -> FakeBlock:
    return FakeBlock(type="text", text=text)


def tool_block(name: str, tool_input: dict[str, Any]) -> FakeBlock:
    return FakeBlock(type="tool_use", id="toolu_fake", name=name, input=tool_input)


def succeeded(
    model: str,
    *blocks: FakeBlock,
    usage: FakeUsage | None = None,
    stop_reason: str = "end_turn",
) -> FakeResult:
    message = FakeMessage(
        model=model,
        content=list(blocks) or [text_block("ok")],
        usage=usage or FakeUsage(),
        stop_reason=stop_reason,
    )
    return FakeResult(type="succeeded", message=message)


def errored() -> FakeResult:
    return FakeResult(type="errored", error=FakeError())


def verdict_json(passed: bool = True, score: int = 5, rationale: str = "matches") -> str:
    return json.dumps({"passed": passed, "score": score, "rationale": rationale})


def is_judge_request(params: dict[str, Any]) -> bool:
    return "format" in (params.get("output_config") or {})


Responder = Callable[[str, dict[str, Any]], "FakeResult | None"]


def default_responder(custom_id: str, params: dict[str, Any]) -> FakeResult:
    if is_judge_request(params):
        return succeeded(params["model"], text_block(verdict_json()))
    return succeeded(params["model"], text_block("ok"))


def fixture_perfect_responder(custom_id: str, params: dict[str, Any]) -> FakeResult:
    """Passes every deterministic check in fixtures/simulate/replay.jsonl, and the judge."""
    if is_judge_request(params):
        return default_responder(custom_id, params)
    if params.get("tools"):
        block = tool_block("get_weather", {"city": "Paris", "unit": "c"})
        return succeeded(params["model"], block, stop_reason="tool_use")
    return succeeded(params["model"], text_block('{"name": "Ada Lovelace", "age": 36}'))


# --------------------------------------------------------------------------- client


def _plain(value: Any) -> Any:
    """Deep copy through JSON: also proves the request payload is serialisable."""
    return json.loads(json.dumps(value))


class FakeBatches:
    """`polls_until_ended` retrieve() calls are needed before a batch reports "ended"."""

    def __init__(self, responder: Responder, polls_until_ended: int) -> None:
        self._responder = responder
        self._polls = polls_until_ended
        self._requests: dict[str, list[dict[str, Any]]] = {}
        self._remaining: dict[str, int] = {}
        self.create_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[str] = []

    @property
    def requests(self) -> list[dict[str, Any]]:
        """Every `{custom_id, params}` submitted, across all batches."""
        return [request for batch in self._requests.values() for request in batch]

    def create(self, **kwargs: Any) -> FakeBatch:
        self.create_calls.append(kwargs)
        batch_id = f"msgbatch_{len(self.create_calls)}"
        self._requests[batch_id] = [_plain(dict(r)) for r in kwargs["requests"]]
        self._remaining[batch_id] = self._polls
        return self._batch(batch_id)

    def retrieve(self, message_batch_id: str) -> FakeBatch:
        self.retrieve_calls.append(message_batch_id)
        self._remaining[message_batch_id] = max(0, self._remaining[message_batch_id] - 1)
        return self._batch(message_batch_id)

    def results(self, message_batch_id: str) -> list[FakeEntry]:
        if self._remaining[message_batch_id]:
            raise AssertionError("results() called before the batch ended")
        entries = []
        # Reversed on purpose: the API returns results in any order.
        for request in reversed(self._requests[message_batch_id]):
            result = self._responder(request["custom_id"], request["params"])
            if result is not None:  # None: the entry is missing from the results
                entries.append(FakeEntry(custom_id=request["custom_id"], result=result))
        return entries

    def _batch(self, batch_id: str) -> FakeBatch:
        status = "in_progress" if self._remaining[batch_id] else "ended"
        return FakeBatch(id=batch_id, processing_status=status)


class FakeMessages:
    def __init__(self, batches: FakeBatches) -> None:
        self.batches = batches

    def create(self, **kwargs: Any) -> None:
        raise AssertionError("simulate must use the Batch API, not messages.create")


class FakeAnthropic:
    def __init__(
        self, responder: Responder = default_responder, polls_until_ended: int = 0
    ) -> None:
        self.messages = FakeMessages(FakeBatches(responder, polls_until_ended))

    @property
    def batches(self) -> FakeBatches:
        return self.messages.batches


def install_client(monkeypatch: pytest.MonkeyPatch, client: FakeAnthropic) -> list[FakeAnthropic]:
    """Route `batch.create_client()` to `client`; the returned list grows per construction."""
    constructed: list[FakeAnthropic] = []

    def factory(*args: Any, **kwargs: Any) -> FakeAnthropic:
        constructed.append(client)
        return client

    monkeypatch.setattr("triage.simulate.batch.create_client", factory)
    return constructed


def forbid_client(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fail on any client construction, through the seam or the SDK. Returns the attempts."""
    attempts: list[str] = []

    def explode(name: str) -> Callable[..., None]:
        def factory(*args: Any, **kwargs: Any) -> None:
            attempts.append(name)
            raise AssertionError(f"{name} must not be called")

        return factory

    module = ModuleType("anthropic")
    module.Anthropic = explode("anthropic.Anthropic")
    module.AsyncAnthropic = explode("anthropic.AsyncAnthropic")
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setattr("triage.simulate.batch.create_client", explode("create_client"))
    return attempts


# --------------------------------------------------------------------------- domain builders


def make_task(task_id: str = "t1", **overrides: Any) -> ReplayTask:
    fields: dict[str, Any] = {
        "task_id": task_id,
        "feature": "summarise",
        "prompt_version": "v1",
        "messages": [{"role": "user", "content": f"Summarise item {task_id}."}],
        "baseline_model": "claude-opus-5-5",
        "baseline_response": ResponseContent(text="A summary."),
    }
    fields.update(overrides)
    return ReplayTask(**fields)


def make_usage(model: str = "claude-opus-5-5", **overrides: Any) -> UsageRecord:
    fields: dict[str, Any] = {"model": model, "input_tokens": 1000, "output_tokens": 100}
    fields.update(overrides)
    return UsageRecord(**fields)


def make_outcome(
    task_id: str = "t1",
    model: str = "claude-sonnet-5-5",
    *,
    det: bool | None = None,
    judge: bool | None = None,
    status: str = "succeeded",
    projected: str = "0",
    sim: str = "0",
    text: str = "answer",
    **overrides: Any,
) -> CandidateOutcome:
    """`det` / `judge` set to None means that grader did not run."""
    checks = [] if det is None else [{"name": "json_parse", "passed": det}]
    verdict = None
    if judge is not None:
        verdict = JudgeVerdict(passed=judge, score=5 if judge else 1, rationale="because")
    fields: dict[str, Any] = {
        "task_id": task_id,
        "feature": "summarise",
        "prompt_version": "v1",
        "model": model,
        "status": status,
        "response": ResponseContent(text=text) if status == "succeeded" else None,
        "sim_cost": Decimal(sim),
        "projected_cost": Decimal(projected),
        "det": DeterministicResult(checks=checks),
        "judge": verdict,
    }
    fields.update(overrides)
    return CandidateOutcome(**fields)
