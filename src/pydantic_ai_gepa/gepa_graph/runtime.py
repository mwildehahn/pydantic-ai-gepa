"""Runtime helpers for executing the GEPA graph."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from pydantic_graph import Graph

from ..adapter import Adapter
from ..exceptions import UsageBudgetExceeded
from ..types import DataInstT
from .deps import GepaDeps
from .datasets import DatasetInput, resolve_dataset
from .graph import create_gepa_graph
from .helpers import create_deps
from .models import GepaConfig, GepaResult, GepaState
from .nodes import StartNode
from .nodes.base import GepaNode
from ..progress import OptimizationProgress


logger = logging.getLogger(__name__)


async def optimize(
    *,
    adapter: Adapter[DataInstT],
    config: GepaConfig,
    trainset: DatasetInput,
    valset: DatasetInput | None = None,
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
        trainset: Training dataset or loader used for minibatch reflections.
        valset: Optional validation dataset specification; defaults to ``trainset`` when omitted.
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
    training_loader = await resolve_dataset(trainset, name="trainset")
    validation_loader = await resolve_dataset(valset, name="valset") if valset is not None else None

    state = GepaState(
        config=config,
        training_set=training_loader,
        validation_set=validation_loader,
    )
    start = start_node if start_node is not None else StartNode()

    run_result = None
    try:
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
                        best_score=state.best_score,
                    )
                    previous_node_name = current_node_name
                run_result = run.result
            progress_bar.update(
                state.total_evaluations,
                best_score=state.best_score,
            )
    except UsageBudgetExceeded:
        state.mark_stopped(reason="Usage budget exceeded")
        logger.info("GEPA run stopped early due to usage budget limit.")
        return GepaResult.from_state(state)

    if run_result is None:
        raise RuntimeError("GEPA graph run did not complete.")

    return run_result.output


__all__ = ["optimize"]
