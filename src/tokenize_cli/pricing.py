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

import yaml

from tokenize_cli.models import CostBreakdown, ModelPricing, PricingConfig, UsageRecord

PRICING_ENV_VAR = "TOKENIZE_PRICING"
DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[2] / "pricing.yaml"
_PER_MTOK = Decimal(1_000_000)
_DATED_SUFFIX = re.compile(r"-\d{8}$")


class UnknownModelError(KeyError):
    """Raised when a model id is not present in pricing.yaml (by id, alias or dated id)."""

    def __str__(self) -> str:
        return f"unknown model {self.args[0]!r}: add it to pricing.yaml"


class UnknownGeoError(ValueError):
    """Raised when a record's inference_geo has no multiplier in pricing.yaml."""


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
    """Load and validate pricing config. Order: explicit path, $TOKENIZE_PRICING, repo root."""
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

    def cost(self, record: UsageRecord) -> CostBreakdown:
        model_id = self.resolve(record.model)
        pricing = self.config.models[model_id]
        cache = self.config.cache_multipliers
        rate_in, rate_out, tier = self.rates_for(model_id, self._tier_basis(record, pricing))
        read_mult = pricing.cache_read_multiplier or cache.read

        scale = self.geo_multiplier(model_id, record.inference_geo) / _PER_MTOK
        if record.is_batch:
            scale *= self.config.batch_discount

        parts = {
            "input_cost": record.input_tokens * rate_in,
            "cache_write_5m_cost": record.cache_creation_5m_input_tokens * rate_in * cache.write_5m,
            "cache_write_1h_cost": record.cache_creation_1h_input_tokens * rate_in * cache.write_1h,
            "cache_read_cost": record.cache_read_input_tokens * rate_in * read_mult,
            "output_cost": record.output_tokens * rate_out,
        }
        parts = {k: v * scale for k, v in parts.items()}
        return CostBreakdown(
            model=model_id,
            total=sum(parts.values(), Decimal(0)),
            tier_applied=tier,
            batch_applied=record.is_batch,
            geo_multiplier=self.geo_multiplier(model_id, record.inference_geo),
            **parts,
        )

    def cliff_thresholds(self) -> dict[str, int]:
        """Lowest pricing-tier threshold per model (for SQL bleed detection)."""
        return {
            model_id: model.tiers[0].above_prompt_tokens
            for model_id, model in self.config.models.items()
            if model.tiers
        }
