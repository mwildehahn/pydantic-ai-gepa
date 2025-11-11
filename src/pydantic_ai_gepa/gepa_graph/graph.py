"""Graph construction helpers for the native GEPA implementation."""

from __future__ import annotations

from pydantic_graph import Graph
from pydantic_graph.beta import GraphBuilder, StepContext

from ..adapter import AgentAdapter
from ..types import DataInstT
from .deps import GepaDeps
from .models import GepaConfig, GepaResult, GepaState
from .nodes import (
    ContinueNode,
    EvaluateNode,
    MergeNode,
    ReflectNode,
    StartNode,
    continue_node as continue_module,
    evaluate as evaluate_module,
    merge as merge_module,
    reflect as reflect_module,
    start as start_module,
)

def create_gepa_graph(
    adapter: AgentAdapter[DataInstT],
    config: GepaConfig,
) -> Graph[GepaState, GepaDeps[DataInstT], GepaResult]:
    """Create the GEPA graph definition based on the provided configuration.

    Args:
        adapter: Adapter used for prompt optimization (currently unused but part of the public API
            to stay aligned with the dependency construction helper).
        config: Immutable optimization configuration.
    """
    _ = adapter  # Ensures the signature stays aligned with create_deps while unused for now.

    _ensure_forward_refs()

    return Graph(
        nodes=(
            StartNode,
            EvaluateNode,
            ContinueNode,
            ReflectNode,
            MergeNode,
        ),
        name="gepa_graph",
        state_type=GepaState,
        run_end_type=GepaResult,
    )


__all__ = ["create_gepa_graph"]


def _ensure_forward_refs() -> None:
    """Make forward-referenced node types resolvable for typing.get_type_hints."""

    setattr(start_module, "EvaluateNode", EvaluateNode)
    setattr(evaluate_module, "ContinueNode", ContinueNode)

    setattr(reflect_module, "EvaluateNode", EvaluateNode)
    setattr(reflect_module, "ContinueNode", ContinueNode)

    setattr(merge_module, "EvaluateNode", EvaluateNode)
    setattr(merge_module, "ContinueNode", ContinueNode)

    setattr(continue_module, "ReflectNode", ReflectNode)
    setattr(continue_module, "MergeNode", MergeNode)
