"""GEPA graph-based implementation package."""

from __future__ import annotations

from . import evaluation, models
from .deps import GepaDeps
from .evaluation import EvaluationResults, ParallelEvaluator, ParetoFrontManager
from .graph import create_gepa_graph
from .helpers import create_deps
from .models import (
    CandidateProgram,
    CandidateSelectorStrategy,
    ComponentValue,
    GepaConfig,
    GepaResult,
    GepaState,
)
from .runtime import optimize

__all__ = [
    "evaluation",
    "models",
    "create_deps",
    "create_gepa_graph",
    "optimize",
    "GepaConfig",
    "GepaState",
    "GepaResult",
    "CandidateProgram",
    "CandidateSelectorStrategy",
    "ComponentValue",
    "GepaDeps",
    "EvaluationResults",
    "ParallelEvaluator",
    "ParetoFrontManager",
]
