"""Canonical pydantic v2 schemas shared across triage modules."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


def _to_decimal(value: Any) -> Any:
    # YAML yields floats; go through str() so 0.1 becomes Decimal("0.1"), not its binary expansion.
    return Decimal(str(value)) if isinstance(value, float) else value


Money = Annotated[Decimal, BeforeValidator(_to_decimal), Field(ge=0)]
Multiplier = Annotated[Decimal, BeforeValidator(_to_decimal), Field(gt=0)]
Factor = Annotated[Decimal, BeforeValidator(_to_decimal), Field(ge=0)]  # may be 0 (free)
TokenCount = Annotated[int, Field(ge=0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- pricing schema


class PriceTier(_Strict):
    """Alternate base rates applied when a request's prompt exceeds a token threshold."""

    above_prompt_tokens: TokenCount
    input: Money
    output: Money


class ModelPricing(_Strict):
    input: Money
    output: Money
    aliases: list[str] = Field(default_factory=list)
    cache_read_multiplier: Multiplier | None = None
    fast_multiplier: Multiplier | None = None  # None: fast mode not offered for this model
    geo_supported: bool = False
    tiers: list[PriceTier] = Field(default_factory=list)
    retired: bool = False
    price_source: Literal["docs", "historical"] = "docs"

    @model_validator(mode="after")
    def _tiers_sorted(self) -> ModelPricing:
        thresholds = [t.above_prompt_tokens for t in self.tiers]
        if thresholds != sorted(set(thresholds)):
            raise ValueError("tiers must be strictly ascending by above_prompt_tokens")
        return self


class CacheMultipliers(_Strict):
    write_5m: Multiplier
    write_1h: Multiplier
    read: Multiplier


class AuditThresholds(_Strict):
    """Bleed-detector thresholds (`audit:` in pricing.yaml)."""

    uncached_min_input_tokens: TokenCount
    uncached_min_repeats: Annotated[int, Field(ge=2)]
    cache_5m_ttl_seconds: Annotated[int, Field(gt=0)]
    bloated_max_tokens_ratio: Multiplier
    top_n: Annotated[int, Field(ge=1)]


class PricingConfig(_Strict):
    version: int
    source: str | None = None
    retrieved: str | None = None
    batch_discount: Annotated[Decimal, BeforeValidator(_to_decimal), Field(gt=0, le=1)]
    cache_multipliers: CacheMultipliers
    inference_geo_multipliers: dict[str, Multiplier]
    service_tier_multipliers: dict[str, Factor] = Field(default_factory=dict)
    audit: AuditThresholds | None = None
    models: dict[str, ModelPricing]

    @model_validator(mode="before")
    @classmethod
    def _coerce_retrieved(cls, data: Any) -> Any:
        # YAML parses an unquoted 2026-11-25 as a date; keep it as an ISO string.
        if isinstance(data, dict) and data.get("retrieved") is not None:
            data = {**data, "retrieved": str(data["retrieved"])}
        return data

    @model_validator(mode="after")
    def _names_unique(self) -> PricingConfig:
        seen: set[str] = set(self.models)
        for model_id, model in self.models.items():
            for alias in model.aliases:
                if alias in seen:
                    raise ValueError(f"alias {alias!r} of {model_id!r} collides with another name")
                seen.add(alias)
        return self


# --------------------------------------------------------------------------- usage + cost


_NO_GEO = {None, "", "not_available"}


class UsageRecord(_Strict):
    """Token usage normalised from JSONL logs or Console exports.

    A JSONL log line is one request (`request_count == 1`). A Console usage-report row is an
    aggregate over a time bucket: `request_count` is the bucket's request total, or `None` when
    the export doesn't report one (legacy usage report). Per-request detectors must skip
    aggregated rows (`is_aggregated`).
    """

    model: str
    input_tokens: TokenCount = 0
    output_tokens: TokenCount = 0
    cache_creation_5m_input_tokens: TokenCount = 0
    cache_creation_1h_input_tokens: TokenCount = 0
    cache_read_input_tokens: TokenCount = 0
    request_id: str | None = None
    timestamp: datetime | None = None
    max_tokens: TokenCount | None = None
    is_batch: bool = False
    inference_geo: str = "global"
    feature: str | None = None
    api_key_id: str | None = None
    system_prompt_hash: str | None = None
    request_count: Annotated[int, Field(ge=1)] | None = 1
    period_end: datetime | None = None
    long_context: bool | None = None  # Console `context_window` bucket; overrides tier guessing
    service_tier: str | None = None
    speed: Literal["standard", "fast"] = "standard"
    workspace_id: str | None = None
    actor_id: str | None = None
    product: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalise_inputs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # Console/API report "not_available" (or null) for models without geo routing.
        if "inference_geo" in data and data["inference_geo"] in _NO_GEO:
            data["inference_geo"] = "global"
        if "speed" in data:
            speed = data["speed"]
            data["speed"] = (
                speed.strip().lower() if isinstance(speed, str) and speed else "standard"
            )
        if "cache_creation_input_tokens" not in data:
            return data
        flat = data.pop("cache_creation_input_tokens")
        has_split = (
            "cache_creation_5m_input_tokens" in data or "cache_creation_1h_input_tokens" in data
        )
        if not has_split and flat is not None:
            data["cache_creation_5m_input_tokens"] = flat
        return data

    @property
    def is_aggregated(self) -> bool:
        return self.request_count != 1

    @property
    def prompt_tokens(self) -> int:
        """Total prompt length as Anthropic counts it for tiered pricing (incl. cache tokens)."""
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_5m_input_tokens
            + self.cache_creation_1h_input_tokens
        )

    @property
    def tier_prompt_tokens(self) -> int:
        """Prompt length per request, for tier selection. Averages over aggregated rows;
        an aggregate with no request count gives 0 (base rates) since sums say nothing."""
        if self.request_count is None:
            return 0
        return self.prompt_tokens // self.request_count


class CostBreakdown(_Strict):
    model: str
    input_cost: Decimal
    cache_write_5m_cost: Decimal
    cache_write_1h_cost: Decimal
    cache_read_cost: Decimal
    output_cost: Decimal
    total: Decimal
    tier_applied: int | None = None
    batch_applied: bool = False
    fast_applied: bool = False
    geo_multiplier: Decimal = Decimal(1)
    service_tier_multiplier: Decimal = Decimal(1)
