"""Start step - initializes the GEPA optimization run."""

from __future__ import annotations

from typing import Mapping

from pydantic_graph.beta import StepContext

from ..deps import GepaDeps
from ..models import CandidateProgram, ComponentValue, GepaState


async def start_step(ctx: StepContext[GepaState, GepaDeps, None]) -> None:
    """Initialize the GEPA optimization by adding the seed candidate."""

    state = ctx.state

    if state.candidates:
        if state.iteration < 0:
            # Checkpoints can restore candidates before the first run; normalize the counter.
            state.iteration = 0
        return None

    seed_components = _determine_seed_components(ctx.deps)
    candidate = _build_candidate(state, seed_components)
    state.add_candidate(candidate)
    state.iteration = 0
    return None


def _determine_seed_components(deps: GepaDeps) -> Mapping[str, str]:
    if deps.seed_candidate:
        return deps.seed_candidate

    components = deps.adapter.get_components()
    deps.seed_candidate = dict(components)
    return deps.seed_candidate


def _build_candidate(
    state: GepaState,
    components: Mapping[str, str],
) -> CandidateProgram:
    component_models = {
        name: ComponentValue(name=name, text=str(text))
        for name, text in components.items()
    }
    return CandidateProgram(
        idx=len(state.candidates),
        components=component_models,
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=state.total_evaluations,
    )


__all__ = ["start_step"]
