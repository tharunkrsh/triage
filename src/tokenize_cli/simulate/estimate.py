"""Pre-flight Batch API spend estimate. Offline: no client, no key, no network.

Every request is priced on its own (so tier cliffs apply per request) through
`PricingTable.cost(is_batch=True)`. Prompt tokens come from the logged baseline usage and are
all priced as uncached input; without usage they are approximated from character counts. The
upper bound assumes each response runs to max_tokens; its input counts are still estimates.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Literal, NamedTuple

from tokenize_cli.models import UsageRecord
from tokenize_cli.pricing import PricingTable
from tokenize_cli.simulate.batch import DEFAULT_MAX_TOKENS, resolve_targets
from tokenize_cli.simulate.eval_judge import JUDGE_MAX_TOKENS, build_judge_request
from tokenize_cli.simulate.models import CandidateOutcome, CostEstimate, CostLine, ReplayTask

CHARS_PER_TOKEN = 4
JUDGE_EXPECTED_OUTPUT_TOKENS = 500  # verdict plus some thinking


class _Size(NamedTuple):
    input_tokens: int
    output_tokens: int
    upper_input_tokens: int
    upper_output_tokens: int
    estimated: bool


def _approx_tokens(value: Any) -> int:
    return max(1, math.ceil(len(json.dumps(value, ensure_ascii=False)) / CHARS_PER_TOKEN))


def _candidate_size(task: ReplayTask) -> _Size:
    usage = task.baseline_usage
    logged_in = usage.prompt_tokens if usage else 0
    logged_out = usage.output_tokens if usage else 0
    tokens_in = logged_in or _approx_tokens([task.system, task.messages, task.tools])
    tokens_out = logged_out or _approx_tokens(task.baseline_response.model_dump())
    upper_out = max(task.max_tokens or DEFAULT_MAX_TOKENS, tokens_out)
    return _Size(tokens_in, tokens_out, tokens_in, upper_out, not (logged_in and logged_out))


def _judge_size(task: ReplayTask, model: str, pricing: PricingTable) -> _Size:
    # Candidate responses don't exist yet: stand the baseline response in for them.
    stand_in = CandidateOutcome(
        task_id=task.task_id, model=model, status="succeeded", response=task.baseline_response
    )
    params = build_judge_request(task, stand_in, model, pricing).params
    candidate = _candidate_size(task)
    # The judge rereads the request plus both responses. Logged prompt tokens can far exceed
    # what the visible text suggests, so take whichever approximation is larger.
    tokens_in = max(
        _approx_tokens([params["system"], params["messages"]]),
        candidate.input_tokens + 2 * candidate.output_tokens + _approx_tokens(params["system"]),
    )
    return _Size(
        input_tokens=tokens_in,
        output_tokens=JUDGE_EXPECTED_OUTPUT_TOKENS,
        upper_input_tokens=tokens_in + candidate.upper_output_tokens,
        upper_output_tokens=JUDGE_MAX_TOKENS,
        estimated=True,
    )


def _line(
    model: str, role: Literal["candidate", "judge"], sizes: list[_Size], pricing: PricingTable
) -> CostLine:
    def price(tokens_in: int, tokens_out: int) -> Decimal:
        record = UsageRecord(
            model=model, input_tokens=tokens_in, output_tokens=tokens_out, is_batch=True
        )
        return pricing.cost(record).total

    return CostLine(
        model=model,
        role=role,
        requests=len(sizes),
        input_tokens=sum(s.input_tokens for s in sizes),
        output_tokens=sum(s.output_tokens for s in sizes),
        expected_usd=sum((price(s.input_tokens, s.output_tokens) for s in sizes), Decimal(0)),
        upper_bound_usd=sum(
            (price(s.upper_input_tokens, s.upper_output_tokens) for s in sizes), Decimal(0)
        ),
        estimated_tokens=any(s.estimated for s in sizes),
    )


def estimate_cost(
    tasks: Sequence[ReplayTask],
    targets: Sequence[str],
    pricing: PricingTable,
    *,
    judge_model: str | None = None,
) -> CostEstimate:
    """One line per target model, plus one for the judge when `judge_model` is given."""
    models = resolve_targets(targets, pricing)
    candidate_sizes = [_candidate_size(task) for task in tasks]
    lines = [_line(model, "candidate", candidate_sizes, pricing) for model in models]
    if judge_model is not None:
        judge = pricing.resolve(judge_model)
        sizes = [_judge_size(task, judge, pricing) for task in tasks] * len(models)
        lines.append(_line(judge, "judge", sizes, pricing))
    return CostEstimate(lines=lines)
