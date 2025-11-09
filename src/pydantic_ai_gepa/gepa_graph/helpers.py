"""Helper utilities for constructing GEPA graph dependencies."""

from __future__ import annotations

from typing import Any

from ..adapter import PydanticAIGEPAAdapter
from .deps import GepaDeps
from .evaluation import ParallelEvaluator, ParetoFrontManager
from .models import GepaConfig
from .proposal import LLMProposalGenerator, MergeProposalBuilder, ReflectiveDatasetBuilder
from .selectors import (
    AllComponentSelector,
    BatchSampler,
    CandidateSelector,
    ComponentSelector,
    CurrentBestCandidateSelector,
    ParetoCandidateSelector,
    RoundRobinComponentSelector,
)


def create_deps(
    adapter: PydanticAIGEPAAdapter[Any],
    config: GepaConfig,
) -> GepaDeps:
    """Construct :class:`GepaDeps` instances for a GEPA run."""
    candidate_selector = _build_candidate_selector(config)
    component_selector = _build_component_selector(config)
    batch_sampler = BatchSampler(seed=config.seed)

    return GepaDeps(
        adapter=adapter,
        evaluator=ParallelEvaluator(),
        pareto_manager=ParetoFrontManager(),
        candidate_selector=candidate_selector,
        component_selector=component_selector,
        batch_sampler=batch_sampler,
        proposal_generator=LLMProposalGenerator(),
        reflective_dataset_builder=ReflectiveDatasetBuilder(
            sampler=getattr(adapter, "reflection_sampler", None)
        ),
        merge_builder=MergeProposalBuilder(seed=config.seed),
        reflection_model=getattr(adapter, "reflection_model", None),
    )


def _build_candidate_selector(config: GepaConfig) -> CandidateSelector:
    if config.candidate_selector == "pareto":
        return ParetoCandidateSelector(seed=config.seed)
    if config.candidate_selector == "current_best":
        return CurrentBestCandidateSelector()
    raise ValueError(f"Unsupported candidate selector '{config.candidate_selector}'.")


def _build_component_selector(config: GepaConfig) -> ComponentSelector:
    if config.component_selector == "round_robin":
        return RoundRobinComponentSelector()
    if config.component_selector == "all":
        return AllComponentSelector()
    raise ValueError(f"Unsupported component selector '{config.component_selector}'.")


__all__ = ["create_deps"]
