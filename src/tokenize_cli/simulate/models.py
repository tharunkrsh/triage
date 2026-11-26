"""Pydantic v2 schemas for `tokenize simulate`: replay tasks, check results, outcomes, summaries.

USD amounts are `Decimal` only: raw floats are rejected so binary rounding drift can't enter
cost math. Proportions and confidence bounds are plain floats.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, JsonValue, field_validator

from tokenize_cli.models import UsageRecord


def _reject_float(value: Any) -> Any:
    if isinstance(value, float):
        raise ValueError("USD amounts must be Decimal, str or int, not float")
    return value


USD = Annotated[Decimal, BeforeValidator(_reject_float), Field(ge=0)]
JsonObject = dict[str, JsonValue]
CheckName = Literal["json_parse", "json_schema", "tool_call"]
OutcomeStatus = Literal["succeeded", "errored", "expired", "canceled"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- responses


class ToolCall(_Strict):
    name: str
    input: JsonObject = Field(default_factory=dict)


class ResponseContent(_Strict):
    """The parts of an assistant message the evaluators look at."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str | None = None

    @classmethod
    def from_api(cls, value: Any) -> ResponseContent:
        """Normalise a logged response: plain text, a content-block list, or a full message."""
        if isinstance(value, str):
            return cls(text=value)
        if isinstance(value, dict):
            if not isinstance(value.get("content"), list | str):
                raise ValueError("response object has no `content`")
            inner = cls.from_api(value["content"])
            return inner.model_copy(update={"stop_reason": value.get("stop_reason")})
        if isinstance(value, list):
            texts: list[str] = []
            calls: list[ToolCall] = []
            for block in value:
                if not isinstance(block, dict):
                    raise ValueError("content blocks must be objects")
                if block.get("type") == "text":
                    texts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_use":
                    calls.append(ToolCall(name=block["name"], input=block.get("input") or {}))
            return cls(text="".join(texts), tool_calls=calls)
        raise ValueError(f"unsupported response type {type(value).__name__}")


# --------------------------------------------------------------------------- replay input


def _has_tool_result(message: JsonObject) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


class ReplayTask(_Strict):
    """One historical single-turn request to replay against candidate models."""

    task_id: str
    feature: str | None = None
    prompt_version: str | None = None
    system: str | list[JsonObject] | None = None
    messages: list[JsonObject]
    tools: list[JsonObject] | None = None
    tool_choice: JsonObject | None = None
    expected_schema: JsonObject | None = None
    baseline_model: str
    baseline_response: ResponseContent
    baseline_usage: UsageRecord | None = None
    max_tokens: Annotated[int, Field(gt=0)] | None = None
    temperature: float | None = None  # logged metadata only; never sent to the API

    @field_validator("messages")
    @classmethod
    def _single_turn(cls, messages: list[JsonObject]) -> list[JsonObject]:
        if not messages:
            raise ValueError("messages must not be empty")
        if messages[-1].get("role") != "user":
            raise ValueError("the last message must have role 'user'")
        if any(_has_tool_result(m) for m in messages):
            raise ValueError("multi-turn replay is out of scope for v1 (tool_result blocks found)")
        return messages


# --------------------------------------------------------------------------- evaluation


class CheckResult(_Strict):
    name: CheckName
    passed: bool
    detail: str | None = None


class DeterministicResult(_Strict):
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def applicable(self) -> bool:
        return bool(self.checks)

    @property
    def passed(self) -> bool | None:
        """None when no deterministic check applies to the task."""
        return all(c.passed for c in self.checks) if self.checks else None


class JudgeVerdict(_Strict):
    passed: bool
    score: Annotated[int, Field(ge=1, le=5)]
    rationale: str


class WilsonInterval(_Strict):
    successes: int
    n: int
    p_hat: float | None
    low: float
    high: float
    z: float


class CandidateOutcome(_Strict):
    """One (task, candidate model) replay with its costs and grades."""

    task_id: str
    feature: str | None = None
    prompt_version: str | None = None
    model: str
    status: OutcomeStatus
    response: ResponseContent | None = None
    usage: UsageRecord | None = None
    sim_cost: USD = Decimal(0)  # actual Batch API spend
    projected_cost: USD = Decimal(0)  # same tokens at standard (non-batch) rates
    det: DeterministicResult | None = None
    judge: JudgeVerdict | None = None

    @property
    def passed(self) -> bool | None:
        """None if the replay didn't succeed or nothing graded it."""
        if self.status != "succeeded":
            return None
        det = self.det.passed if self.det is not None else None
        judge = self.judge.passed if self.judge is not None else None
        if det is None and judge is None:
            return None
        return det is not False and judge is not False


class CandidateSummary(_Strict):
    """Aggregate for one (feature, prompt_version, model) group."""

    feature: str | None
    prompt_version: str | None
    model: str
    n: int
    passes: int
    errored: int = 0
    pass_rate: WilsonInterval
    sim_cost: USD
    projected_cost_per_task: USD
    cost_per_success: USD | None
    agreement: WilsonInterval | None = None
    on_frontier: bool = False
    is_baseline: bool = False


# --------------------------------------------------------------------------- cost + result


class CostLine(_Strict):
    model: str
    role: Literal["candidate", "judge"]
    requests: int
    input_tokens: int
    output_tokens: int
    expected_usd: USD
    upper_bound_usd: USD
    estimated_tokens: bool  # True when the chars-per-token fallback was used


class CostEstimate(_Strict):
    lines: list[CostLine] = Field(default_factory=list)

    @property
    def expected_usd(self) -> Decimal:
        return sum((line.expected_usd for line in self.lines), Decimal(0))

    @property
    def upper_bound_usd(self) -> Decimal:
        return sum((line.upper_bound_usd for line in self.lines), Decimal(0))


class SimulationResult(_Strict):
    source: str | None = None
    targets: list[str]
    judge_model: str | None = None
    summaries: list[CandidateSummary] = Field(default_factory=list)
    overall: list[CandidateSummary] = Field(default_factory=list)  # per model, groups pooled
    agreement: WilsonInterval | None = None
    total_sim_cost: USD = Decimal(0)
