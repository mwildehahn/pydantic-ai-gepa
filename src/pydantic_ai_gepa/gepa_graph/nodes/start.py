"""Start node - initializes the GEPA optimization run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING
from ...gepa_graph.models import CandidateProgram, ComponentValue, GepaState
from ..deps import GepaDeps
from .base import GepaNode, GepaRunContext

if TYPE_CHECKING:
    from .evaluate import EvaluateNode


@dataclass(slots=True)
class StartNode(GepaNode):
    """Initialize the GEPA optimization by adding the seed candidate."""

    async def run(self, ctx: GepaRunContext) -> "EvaluateNode":
        from .evaluate import EvaluateNode  # Local import to avoid circular dependency

        state = ctx.state

        if state.candidates:
            if state.iteration < 0:
                # Checkpoints can restore candidates before StartNode runs; normalize the counter.
                state.iteration = 0
            return EvaluateNode()

        seed_components = self._determine_seed_components(ctx.deps)
        candidate = self._build_candidate(state, seed_components)
        state.add_candidate(candidate)
        state.iteration = 0
        return EvaluateNode()

    def _determine_seed_components(self, deps: GepaDeps) -> Mapping[str, str]:
        seed_candidate = deps.seed_candidate
        if seed_candidate is None:
            raise RuntimeError(
                "GepaDeps.seed_candidate must be provided before StartNode runs. "
                "Set it via create_deps(..., seed_candidate=...) or assign it "
                "directly on the deps object."
            )
        return seed_candidate

    @staticmethod
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


__all__ = ["StartNode"]
