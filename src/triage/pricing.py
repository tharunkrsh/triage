"""Load pricing.yaml and compute per-request Claude API costs.

All rates, cliffs, cache multipliers and discounts come from pricing.yaml; this module
contains only the arithmetic.
"""

from __future__ import annotations

import os
import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import yaml

from triage.models import CostBreakdown, ModelPricing, PricingConfig, UsageRecord

PRICING_ENV_VAR = "TRIAGE_PRICING"
_PACKAGED_PRICING = Path(__file__).resolve().with_name("pricing.yaml")  # in installed wheels
_REPO_PRICING = Path(__file__).resolve().parents[2] / "pricing.yaml"  # in a source checkout
DEFAULT_PRICING_PATH = _PACKAGED_PRICING if _PACKAGED_PRICING.is_file() else _REPO_PRICING
_PER_MTOK = Decimal(1_000_000)
_DATED_SUFFIX = re.compile(r"-\d{8}$")


class UnknownModelError(KeyError):
    """Raised when a model id is not present in pricing.yaml (by id, alias or dated id)."""

    def __str__(self) -> str:
        return f"unknown model {self.args[0]!r}: add it to pricing.yaml"


class UnknownGeoError(ValueError):
    """Raised when a record's inference_geo has no multiplier in pricing.yaml."""


class UnsupportedSpeedError(ValueError):
    """Raised for a fast-mode record on a model with no fast_multiplier in pricing.yaml."""


class EffectiveRates(NamedTuple):
    """Per-token USD rates for one record with every multiplier applied."""

    model: str
    input: Decimal  # also the base for cache write/read multipliers
    output: Decimal
    cache_read_multiplier: Decimal
    tier_applied: int | None
    batch_applied: bool
    fast_applied: bool
    geo_multiplier: Decimal
    service_tier_multiplier: Decimal


def resolve_pricing_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path).resolve()
    env = os.environ.get(PRICING_ENV_VAR)
    return Path(env).resolve() if env else DEFAULT_PRICING_PATH


@lru_cache(maxsize=8)
def _load(path: Path) -> PricingConfig:
    with path.open(encoding="utf-8") as fh:
        return PricingConfig.model_validate(yaml.safe_load(fh))


def load_pricing(path: Path | str | None = None) -> PricingConfig:
    """Load and validate pricing config. Order: explicit path, $TRIAGE_PRICING, bundled file."""
    return _load(resolve_pricing_path(path))


class PricingTable:
    """Cost calculator over a validated PricingConfig."""

    def __init__(self, config: PricingConfig) -> None:
        self.config = config
        self._names: dict[str, str] = {model_id: model_id for model_id in config.models}
        for model_id, model in config.models.items():
            self._names.update({alias: model_id for alias in model.aliases})

    @classmethod
    def load(cls, path: Path | str | None = None) -> PricingTable:
        return cls(load_pricing(path))

    def resolve(self, model: str) -> str:
        """Return the canonical model id for an id, alias, or dated snapshot id."""
        name = model.strip()
        if name in self._names:
            return self._names[name]
        undated = _DATED_SUFFIX.sub("", name)
        if undated in self._names:
            return self._names[undated]
        raise UnknownModelError(model)

    def get(self, model: str) -> ModelPricing:
        return self.config.models[self.resolve(model)]

    def rates_for(self, model: str, prompt_tokens: int) -> tuple[Decimal, Decimal, int | None]:
        """Effective base (input, output) USD/MTok and the tier threshold applied, if any."""
        pricing = self.get(model)
        rate_in, rate_out, applied = pricing.input, pricing.output, None
        for tier in pricing.tiers:
            if prompt_tokens > tier.above_prompt_tokens:
                rate_in, rate_out, applied = tier.input, tier.output, tier.above_prompt_tokens
        return rate_in, rate_out, applied

    def geo_multiplier(self, model: str, inference_geo: str) -> Decimal:
        multipliers = self.config.inference_geo_multipliers
        if inference_geo not in multipliers:
            raise UnknownGeoError(f"no multiplier for inference_geo {inference_geo!r}")
        if not self.get(model).geo_supported:
            return Decimal(1)
        return multipliers[inference_geo]

    @staticmethod
    def _tier_basis(record: UsageRecord, pricing: ModelPricing) -> int:
        """Prompt length used for tier selection.

        A Console `context_window` label (long_context) is authoritative: it is the billing
        boundary itself, so True selects the first tier and False the base rate.
        """
        if record.long_context is None or not pricing.tiers:
            return record.tier_prompt_tokens
        return pricing.tiers[0].above_prompt_tokens + 1 if record.long_context else 0

    def service_tier_multiplier(self, service_tier: str | None) -> Decimal:
        """Marginal-cost factor for a service tier; tiers not in pricing.yaml pay list price."""
        return self.config.service_tier_multipliers.get(service_tier or "standard", Decimal(1))

    def effective_rates(self, record: UsageRecord, *, tiered: bool = True) -> EffectiveRates:
        """Per-token rates for `record`. `tiered=False` prices it at base (pre-cliff) rates."""
        model_id = self.resolve(record.model)
        pricing = self.config.models[model_id]
        basis = self._tier_basis(record, pricing) if tiered else 0
        rate_in, rate_out, tier = self.rates_for(model_id, basis)

        fast = record.speed == "fast"
        if fast:
            if pricing.fast_multiplier is None:
                raise UnsupportedSpeedError(
                    f"model {model_id!r} has no fast-mode pricing: add fast_multiplier to "
                    "pricing.yaml"
                )
            rate_in *= pricing.fast_multiplier
            rate_out *= pricing.fast_multiplier

        geo = self.geo_multiplier(model_id, record.inference_geo)
        tier_mult = self.service_tier_multiplier(record.service_tier)
        scale = geo * tier_mult / _PER_MTOK
        batch = record.is_batch and not fast  # fast mode is never Batch-discounted
        if batch:
            scale *= self.config.batch_discount
        return EffectiveRates(
            model=model_id,
            input=rate_in * scale,
            output=rate_out * scale,
            cache_read_multiplier=pricing.cache_read_multiplier
            or self.config.cache_multipliers.read,
            tier_applied=tier,
            batch_applied=batch,
            fast_applied=fast,
            geo_multiplier=geo,
            service_tier_multiplier=tier_mult,
        )

    def cost(self, record: UsageRecord, *, tiered: bool = True) -> CostBreakdown:
        rates = self.effective_rates(record, tiered=tiered)
        cache = self.config.cache_multipliers
        parts = {
            "input_cost": record.input_tokens * rates.input,
            "cache_write_5m_cost": record.cache_creation_5m_input_tokens
            * rates.input
            * cache.write_5m,
            "cache_write_1h_cost": record.cache_creation_1h_input_tokens
            * rates.input
            * cache.write_1h,
            "cache_read_cost": record.cache_read_input_tokens
            * rates.input
            * rates.cache_read_multiplier,
            "output_cost": record.output_tokens * rates.output,
        }
        return CostBreakdown(
            model=rates.model,
            total=sum(parts.values(), Decimal(0)),
            tier_applied=rates.tier_applied,
            batch_applied=rates.batch_applied,
            fast_applied=rates.fast_applied,
            geo_multiplier=rates.geo_multiplier,
            service_tier_multiplier=rates.service_tier_multiplier,
            **parts,
        )

    def cliff_thresholds(self) -> dict[str, int]:
        """Lowest pricing-tier threshold per model (for SQL bleed detection)."""
        return {
            model_id: model.tiers[0].above_prompt_tokens
            for model_id, model in self.config.models.items()
            if model.tiers
        }

    def cliff_rate_multipliers(self) -> dict[str, Decimal]:
        """Input-rate jump at each model's first pricing tier (e.g. 5 for Haiku 5.5)."""
        return {
            model_id: model.tiers[0].input / model.input
            for model_id, model in self.config.models.items()
            if model.tiers and model.input
        }
