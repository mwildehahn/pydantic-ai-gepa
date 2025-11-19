"""Helper utilities for constructing GEPA graph dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .deps import GepaDeps
from .evaluation import ParallelEvaluator, ParetoFrontManager
from .models import CandidateMap, CandidateSelectorStrategy, GepaConfig
from .selectors import (
    AllComponentSelector,
    BatchSampler,
    CandidateSelector,
    ComponentSelector,
    CurrentBestCandidateSelector,
    ParetoCandidateSelector,
    RoundRobinComponentSelector,
)

if TYPE_CHECKING:
    from ..adapter import Adapter


def create_deps(
    adapter: "Adapter[Any, Any, Any]",
    config: GepaConfig,
    *,
    seed_candidate: CandidateMap | None = None,
) -> GepaDeps:
    """Construct :class:`GepaDeps` instances for a GEPA run.

    Args:
        adapter: Implementation of the Adapter protocol powering evaluations.
        config: Immutable optimization configuration.
        seed_candidate: Optional initial candidate mapping injected into ``GepaDeps``
            for consumption by :class:`StartStep`.
    """
    from .proposal import InstructionProposalGenerator, MergeProposalBuilder

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
        proposal_generator=InstructionProposalGenerator(
            include_hypothesis_metadata=config.track_component_hypotheses,
            model_settings=config.reflection_model_settings,
        ),
        merge_builder=MergeProposalBuilder(seed=config.seed),
        reflection_model=config.reflection_model,
        seed_candidate=seed_candidate,
    )


def _build_candidate_selector(config: GepaConfig) -> CandidateSelector:
    if config.candidate_selector is CandidateSelectorStrategy.PARETO:
        return ParetoCandidateSelector(seed=config.seed)
    if config.candidate_selector is CandidateSelectorStrategy.CURRENT_BEST:
        return CurrentBestCandidateSelector()
    raise ValueError(f"Unsupported candidate selector '{config.candidate_selector}'.")


def _build_component_selector(config: GepaConfig) -> ComponentSelector:
    if config.component_selector == "round_robin":
        return RoundRobinComponentSelector()
    if config.component_selector == "all":
        return AllComponentSelector()
    raise ValueError(f"Unsupported component selector '{config.component_selector}'.")


__all__ = ["create_deps"]
