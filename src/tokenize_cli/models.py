"""Canonical pydantic v2 schemas shared across tokenize modules."""

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


class PricingConfig(_Strict):
    version: int
    source: str | None = None
    retrieved: str | None = None
    batch_discount: Annotated[Decimal, BeforeValidator(_to_decimal), Field(gt=0, le=1)]
    cache_multipliers: CacheMultipliers
    inference_geo_multipliers: dict[str, Multiplier]
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


class UsageRecord(_Strict):
    """One API request's token usage, normalised from JSONL logs or Console exports."""

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

    @model_validator(mode="before")
    @classmethod
    def _split_cache_creation(cls, data: Any) -> Any:
        """Map the API's flat `cache_creation_input_tokens` onto the 5m bucket if no split."""
        if not isinstance(data, dict) or "cache_creation_input_tokens" not in data:
            return data
        data = dict(data)
        flat = data.pop("cache_creation_input_tokens")
        has_split = (
            "cache_creation_5m_input_tokens" in data or "cache_creation_1h_input_tokens" in data
        )
        if not has_split and flat is not None:
            data["cache_creation_5m_input_tokens"] = flat
        return data

    @property
    def prompt_tokens(self) -> int:
        """Total prompt length as Anthropic counts it for tiered pricing (incl. cache tokens)."""
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_5m_input_tokens
            + self.cache_creation_1h_input_tokens
        )


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
    geo_multiplier: Decimal = Decimal(1)
