"""Start node - initializes the GEPA optimization run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

from ...components import extract_seed_candidate_with_signature
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
                state.iteration = 0
            return EvaluateNode()

        seed_components = self._determine_seed_components(ctx.deps)
        candidate = self._build_candidate(state, seed_components)
        state.add_candidate(candidate)
        state.iteration = 0
        return EvaluateNode()

    def _determine_seed_components(self, deps: GepaDeps) -> Mapping[str, str]:
        if deps.seed_candidate:
            return deps.seed_candidate
        adapter = deps.adapter
        return extract_seed_candidate_with_signature(
            agent=adapter.agent,
            input_type=adapter.input_spec,
        )

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
