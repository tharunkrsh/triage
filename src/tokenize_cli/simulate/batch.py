"""Anthropic Message Batches: build replay requests, submit, poll and collect outcomes.

Sampling parameters (`temperature`, `top_p`, `top_k`) are never sent: API 2026-10-21 rejects
them with HTTP 400. A task's logged `temperature` is metadata only.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from tokenize_cli.ingest.jsonl import map_line
from tokenize_cli.models import UsageRecord
from tokenize_cli.pricing import PricingTable
from tokenize_cli.simulate.eval_det import evaluate
from tokenize_cli.simulate.models import CandidateOutcome, JsonObject, ReplayTask, ResponseContent

DEFAULT_MAX_TOKENS = 4096  # used when the logged request carries no max_tokens
POLL_INTERVAL_SECONDS = 30.0


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    custom_id: str
    task_id: str
    model: str
    params: JsonObject


def create_client() -> Any:
    """The only place a client is constructed. Credentials come from the SDK's environment."""
    import anthropic  # lazy: dry runs never need the SDK or a key

    return anthropic.Anthropic()


def resolve_targets(targets: Sequence[str], pricing: PricingTable) -> list[str]:
    """Canonical model ids in first-seen order; raises UnknownModelError."""
    return list(dict.fromkeys(pricing.resolve(target) for target in targets))


def check_unique_task_ids(tasks: Sequence[ReplayTask]) -> None:
    """Results and judge verdicts are keyed by task_id, so duplicates would be mis-graded."""
    repeated = [task_id for task_id, n in Counter(t.task_id for t in tasks).items() if n > 1]
    if repeated:
        raise ValueError(f"duplicate task_id in replay input: {', '.join(repeated[:5])}")


def _params(task: ReplayTask, model: str) -> JsonObject:
    optional = {"system": task.system, "tools": task.tools, "tool_choice": task.tool_choice}
    return {
        "model": model,
        "max_tokens": task.max_tokens or DEFAULT_MAX_TOKENS,
        "messages": task.messages,
        **{key: value for key, value in optional.items() if value is not None},
    }


def build_batch_requests(
    tasks: Sequence[ReplayTask], targets: Sequence[str], pricing: PricingTable
) -> list[BatchRequest]:
    """One request per (task, target model), in task then target order."""
    check_unique_task_ids(tasks)
    models = resolve_targets(targets, pricing)
    return [
        # task ids are free text, so custom_id is positional (API: [a-zA-Z0-9_-]{1,64}).
        BatchRequest(
            custom_id=f"task{i}-model{j}",
            task_id=task.task_id,
            model=model,
            params=_params(task, model),
        )
        for i, task in enumerate(tasks)
        for j, model in enumerate(models)
    ]


def submit_and_collect(
    client: Any,
    requests: Sequence[BatchRequest],
    *,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    on_submit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one batch to completion; returns SDK result objects keyed by custom_id."""
    if not requests:
        return {}
    batches = client.messages.batches
    batch = batches.create(
        requests=[{"custom_id": r.custom_id, "params": r.params} for r in requests]
    )
    if on_submit is not None:
        on_submit(batch.id)
    while batch.processing_status != "ended":
        sleep(poll_interval)
        batch = batches.retrieve(batch.id)
    return {entry.custom_id: entry.result for entry in batches.results(batch.id)}


def usage_from_message(model: str, message: Any) -> UsageRecord:
    """Batch-priced usage for a response, attributed to the requested (canonical) model."""
    row, _ = map_line({"usage": message.usage.model_dump()})
    return UsageRecord.model_validate({**row, "model": model, "is_batch": True})


def _outcome(task: ReplayTask, model: str, result: Any, pricing: PricingTable) -> CandidateOutcome:
    base = {
        "task_id": task.task_id,
        "feature": task.feature,
        "prompt_version": task.prompt_version,
        "model": model,
    }
    if result is None or result.type != "succeeded":
        return CandidateOutcome(**base, status="errored" if result is None else result.type)
    response = ResponseContent.from_api(result.message.model_dump())
    usage = usage_from_message(model, result.message)
    return CandidateOutcome(
        **base,
        status="succeeded",
        response=response,
        usage=usage,
        sim_cost=pricing.cost(usage).total,
        projected_cost=pricing.cost(usage.model_copy(update={"is_batch": False})).total,
        det=evaluate(task, response),
    )


def run_replay(
    client: Any,
    tasks: Sequence[ReplayTask],
    targets: Sequence[str],
    pricing: PricingTable,
    *,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    on_submit: Callable[[str], None] | None = None,
) -> list[CandidateOutcome]:
    """Replay every task against every target and grade the deterministic checks."""
    requests = build_batch_requests(tasks, targets, pricing)
    results = submit_and_collect(
        client, requests, sleep=sleep, poll_interval=poll_interval, on_submit=on_submit
    )
    by_id = {task.task_id: task for task in tasks}
    return [
        _outcome(by_id[r.task_id], r.model, results.get(r.custom_id), pricing) for r in requests
    ]
