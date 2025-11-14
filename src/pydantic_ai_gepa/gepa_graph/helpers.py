"""Helper utilities for constructing GEPA graph dependencies."""

from __future__ import annotations

from collections.abc import Mapping

from ..adapter import Adapter
from ..types import DataInstT
from .deps import GepaDeps
from .evaluation import ParallelEvaluator, ParetoFrontManager
from .models import CandidateSelectorStrategy, GepaConfig
from .proposal import InstructionProposalGenerator, MergeProposalBuilder
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
    adapter: Adapter[DataInstT],
    config: GepaConfig,
    *,
    seed_candidate: Mapping[str, str] | None = None,
) -> GepaDeps[DataInstT]:
    """Construct :class:`GepaDeps` instances for a GEPA run.

    Args:
        adapter: Implementation of the Adapter protocol powering evaluations.
        config: Immutable optimization configuration.
        seed_candidate: Optional initial candidate mapping injected into ``GepaDeps``
            for consumption by :class:`StartStep`.
    """
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
        proposal_generator=InstructionProposalGenerator(),
        merge_builder=MergeProposalBuilder(seed=config.seed),
        reflection_model=config.reflection_model,
        seed_candidate=dict(seed_candidate) if seed_candidate is not None else None,
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
