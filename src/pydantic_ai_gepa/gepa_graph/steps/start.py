"""Start step - initializes the GEPA optimization run."""

from __future__ import annotations

from pydantic_graph.beta import StepContext

from ..deps import GepaDeps
from ..models import CandidateMap, CandidateProgram, ComponentValue, GepaState


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


def _determine_seed_components(deps: GepaDeps) -> CandidateMap:
    if deps.seed_candidate:
        return deps.seed_candidate

    raw_components = deps.adapter.get_components()
    components = {
        name: component if isinstance(component, ComponentValue) else ComponentValue(name=name, text=str(component))
        for name, component in raw_components.items()
    }
    deps.seed_candidate = {name: component.model_copy() for name, component in components.items()}
    return deps.seed_candidate


def _build_candidate(
    state: GepaState,
    components: CandidateMap,
) -> CandidateProgram:
    component_models = {
        name: component if isinstance(component, ComponentValue) else ComponentValue(name=name, text=str(component))
        for name, component in components.items()
    }
    return CandidateProgram(
        idx=len(state.candidates),
        components=component_models,
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=state.total_evaluations,
    )


__all__ = ["start_step"]
