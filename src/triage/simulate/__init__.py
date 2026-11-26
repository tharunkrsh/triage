"""Offline downgrade simulator: replay schemas, deterministic checks and statistics."""

from triage.simulate.eval_det import evaluate
from triage.simulate.models import ReplayTask
from triage.simulate.stats import mark_frontier, wilson_interval

__all__ = ["ReplayTask", "evaluate", "mark_frontier", "wilson_interval"]
