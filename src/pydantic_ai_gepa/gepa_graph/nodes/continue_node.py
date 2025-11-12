"""Continue node - decides whether to stop, merge, or reflect."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..models import GepaState
from .base import End, GepaNode, GepaResult, GepaRunContext

if TYPE_CHECKING:
    from .merge import MergeNode
    from .reflect import ReflectNode


@dataclass(slots=True)
class ContinueNode(GepaNode):
    """Decision point for the GEPA optimization loop."""

    async def run(self, ctx: GepaRunContext) -> ReflectNode | MergeNode | End[GepaResult]:
        from .merge import MergeNode
        from .reflect import ReflectNode

        state = ctx.state

        if self._should_stop(state):
            state.stopped = True
            result = GepaResult.from_state(state)
            return End(result)

        state.iteration += 1

        if (
            state.config.use_merge
            and state.merge_scheduled > 0
            and state.last_accepted
        ):
            return MergeNode()

        return ReflectNode()

    def _should_stop(self, state: GepaState) -> bool:
        if state.stopped:
            return True

        if state.total_evaluations >= state.config.max_evaluations:
            state.stop_reason = "Max evaluations reached"
            return True

        if state.config.max_iterations is not None and state.iteration >= state.config.max_iterations:
            state.stop_reason = "Max iterations reached"
            return True

        return False


__all__ = ["ContinueNode"]
