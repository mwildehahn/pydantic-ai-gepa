"""Dependency container for GEPA graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..adapter import PydanticAIGEPAAdapter
from .evaluation import ParallelEvaluator, ParetoFrontManager


@dataclass(slots=True)
class GepaDeps:
    """Runtime dependencies shared across GEPA graph nodes.

    More optional fields will be added as subsequent refactor steps land.
    """

    adapter: PydanticAIGEPAAdapter[Any]
    evaluator: ParallelEvaluator
    pareto_manager: ParetoFrontManager
    seed_candidate: dict[str, str] | None = None
