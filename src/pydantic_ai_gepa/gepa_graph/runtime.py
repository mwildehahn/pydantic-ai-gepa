"""Runtime helpers for executing the GEPA graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic_graph import Graph

from ..adapter import Adapter
from ..types import DataInstT
from .deps import GepaDeps
from .graph import create_gepa_graph
from .helpers import create_deps
from .models import GepaConfig, GepaResult, GepaState
from .nodes import StartNode
from .nodes.base import GepaNode
from ..progress import OptimizationProgress


async def optimize(
    *,
    adapter: Adapter[DataInstT],
    config: GepaConfig,
    trainset: Sequence[DataInstT],
    valset: Sequence[DataInstT] | None = None,
    seed_candidate: Mapping[str, str] | None = None,
    deps: GepaDeps[DataInstT] | None = None,
    graph: Graph[GepaState, GepaDeps[DataInstT], GepaResult] | None = None,
    start_node: GepaNode | None = None,
    show_progress: bool = False,
) -> GepaResult:
    """Execute the GEPA graph end-to-end and return the resulting ``GepaResult``.

    Args:
        adapter: Implementation of the Adapter protocol that powers evaluation/reflection.
        config: Immutable optimization configuration.
        trainset: Training dataset used for minibatch reflections.
        valset: Optional validation dataset; defaults to ``trainset`` when omitted.
        seed_candidate: Mapping of component names to their initial text. Required unless
            already attached to ``deps``.
        deps: Preconstructed dependency bundle; ``create_deps`` used when omitted.
        graph: Custom graph definition; ``create_gepa_graph`` used when omitted.
        start_node: Alternative start node for advanced scenarios.
        show_progress: When True, display a Rich progress bar that tracks the evaluation budget.
    """

    if deps is None:
        resolved_deps = create_deps(
            adapter,
            config,
            seed_candidate=seed_candidate,
        )
    else:
        resolved_deps = deps
        if seed_candidate is not None:
            resolved_deps.seed_candidate = dict(seed_candidate)

    resolved_graph = (
        graph
        if graph is not None
        else create_gepa_graph(adapter=adapter, config=config)
    )
    state = GepaState(config=config, training_set=trainset, validation_set=valset)
    start = start_node if start_node is not None else StartNode()

    run_result = None
    with OptimizationProgress(
        total=config.max_evaluations,
        description="GEPA optimize",
        enabled=show_progress,
    ) as progress_bar:
        previous_node_name: str | None = None
        async with resolved_graph.iter(start, state=state, deps=resolved_deps) as run:
            async for node in run:
                current_node_name = node.__class__.__name__
                progress_bar.update(
                    state.total_evaluations,
                    current_node=current_node_name,
                    previous_node=previous_node_name,
                )
                previous_node_name = current_node_name
            run_result = run.result
        progress_bar.update(state.total_evaluations)

    if run_result is None:
        raise RuntimeError("GEPA graph run did not complete.")

    return run_result.output


__all__ = ["optimize"]
