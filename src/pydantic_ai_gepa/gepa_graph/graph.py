"""Graph construction helpers for the native GEPA implementation."""

from __future__ import annotations

from typing import Any

from pydantic_graph import Graph

from ..adapter import PydanticAIGEPAAdapter
from .deps import GepaDeps
from .models import GepaConfig, GepaResult, GepaState
from .nodes import ContinueNode, EvaluateNode, MergeNode, ReflectNode, StartNode

def create_gepa_graph(
    adapter: PydanticAIGEPAAdapter[Any],
    config: GepaConfig,
) -> Graph[GepaState, GepaDeps, GepaResult]:
    """Create the GEPA graph definition based on the provided configuration.

    Args:
        adapter: Adapter used for prompt optimization (currently unused but part of the public API
            to stay aligned with the dependency construction helper).
        config: Immutable optimization configuration.
    """
    _ = adapter  # Ensures the signature stays aligned with create_deps while unused for now.

    nodes: list[type] = [
        StartNode,
        EvaluateNode,
        ContinueNode,
        ReflectNode,
    ]

    if config.use_merge:
        nodes.append(MergeNode)

    return Graph(
        nodes=tuple(nodes),
        name="gepa_graph",
        state_type=GepaState,
        run_end_type=GepaResult,
    )


__all__ = ["create_gepa_graph"]
