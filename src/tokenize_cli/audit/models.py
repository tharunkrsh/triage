"""Pydantic result models for `tokenize audit`."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Result(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PriorityUsage(_Result):
    """Priority Tier tokens: $0 marginal cost, reported separately."""

    rows: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_tokens


class UncachedRepeat(_Result):
    """Identical system prompts sent uncached; savings assume 5-minute prompt caching."""

    system_prompt_hash: str
    model: str
    requests: int
    cacheable_tokens: int  # shortest prompt in the group: a safe lower bound on the prefix
    current_cost: Decimal
    cached_cost: Decimal
    estimated_savings: Decimal


class CliffFinding(_Result):
    """A single request priced at a long-context tier (e.g. Haiku over 100k prompt tokens)."""

    request_id: str | None
    model: str
    prompt_tokens: int
    threshold: int
    excess_tokens: int
    rate_multiplier: Decimal
    actual_cost: Decimal
    base_cost: Decimal  # same tokens at base (under-cliff) rates
    estimated_savings: Decimal


class BloatedMaxTokens(_Result):
    """A request whose max_tokens far exceeds its output (rate-limit reservation, not cost)."""

    request_id: str | None
    model: str
    max_tokens: int
    output_tokens: int
    unused_tokens: int
    ratio: float | None  # None when output_tokens == 0
    estimated_savings: Decimal = Decimal(0)


class SpendShare(_Result):
    key: str
    spend: Decimal
    share: float  # 0..1; 0 when total spend is 0


class Concentration(_Result):
    by_model: list[SpendShare] = Field(default_factory=list)
    by_feature: list[SpendShare] = Field(default_factory=list)
    by_api_key: list[SpendShare] = Field(default_factory=list)
    by_workspace: list[SpendShare] = Field(default_factory=list)


Detector = Literal["uncached_repeat", "haiku_cliff", "bloated_max_tokens"]


class Offender(_Result):
    detector: Detector
    label: str
    model: str
    estimated_savings: Decimal


class AuditResult(_Result):
    rows: int = 0
    total_spend: Decimal = Decimal(0)
    total_estimated_savings: Decimal = Decimal(0)
    top_n: int
    priority: PriorityUsage = Field(default_factory=PriorityUsage)
    uncached_repeats: list[UncachedRepeat] = Field(default_factory=list)
    haiku_cliff: list[CliffFinding] = Field(default_factory=list)
    bloated_max_tokens: list[BloatedMaxTokens] = Field(default_factory=list)
    concentration: Concentration = Field(default_factory=Concentration)
    top_offenders: list[Offender] = Field(default_factory=list)

    @property
    def finding_count(self) -> int:
        return len(self.uncached_repeats) + len(self.haiku_cliff) + len(self.bloated_max_tokens)
