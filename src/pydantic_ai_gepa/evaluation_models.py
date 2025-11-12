"""Shared evaluation data structures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .types import RolloutOutput, Trajectory


@dataclass(slots=True)
class EvaluationBatch:
    """Evaluation payload produced by adapters."""

    outputs: list[RolloutOutput[Any]]
    scores: list[float]
    trajectories: Sequence[Trajectory | None] | None = None


__all__ = ["EvaluationBatch"]
