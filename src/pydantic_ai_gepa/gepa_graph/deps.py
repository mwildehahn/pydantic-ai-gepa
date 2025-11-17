"""Dependency container for GEPA graph steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai.models import KnownModelName, Model

from ..adapter import Adapter
from .evaluation import ParallelEvaluator, ParetoFrontManager
from .proposal import InstructionProposalGenerator, MergeProposalBuilder
from .selectors import BatchSampler, CandidateSelector, ComponentSelector

@dataclass(slots=True)
class GepaDeps:
    """Runtime dependencies shared across GEPA graph steps."""

    adapter: Adapter[Any, Any, Any]
    evaluator: ParallelEvaluator
    pareto_manager: ParetoFrontManager
    candidate_selector: CandidateSelector
    component_selector: ComponentSelector
    batch_sampler: BatchSampler
    proposal_generator: InstructionProposalGenerator
    merge_builder: MergeProposalBuilder
    reflection_model: Model | KnownModelName | str | None = None
    seed_candidate: dict[str, str] | None = None
