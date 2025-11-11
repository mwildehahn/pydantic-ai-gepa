"""Runtime helpers for executing the GEPA graph."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_graph import Graph

from ..adapter import AgentAdapter
from ..types import DataInstT
from .deps import GepaDeps
from .graph import create_gepa_graph
from .helpers import create_deps
from .models import GepaConfig, GepaResult, GepaState
from .nodes import StartNode
from .nodes.base import GepaNode


async def optimize(
    *,
    adapter: AgentAdapter[DataInstT],
    config: GepaConfig,
    trainset: Sequence[DataInstT],
    valset: Sequence[DataInstT] | None = None,
    deps: GepaDeps[DataInstT] | None = None,
    graph: Graph[GepaState, GepaDeps[DataInstT], GepaResult] | None = None,
    start_node: GepaNode | None = None,
) -> GepaResult:
    """Execute the GEPA graph end-to-end and return the resulting GepaResult."""

    resolved_deps = deps if deps is not None else create_deps(adapter, config)
    resolved_graph = (
        graph
        if graph is not None
        else create_gepa_graph(adapter=adapter, config=config)
    )
    state = GepaState(config=config, training_set=trainset, validation_set=valset)
    start = start_node if start_node is not None else StartNode()

    async with resolved_graph.iter(start, state=state, deps=resolved_deps) as run:
        async for _ in run:
            pass

    run_result = run.result
    if run_result is None:
        raise RuntimeError("GEPA graph run did not complete.")

    return run_result.output


__all__ = ["optimize"]
