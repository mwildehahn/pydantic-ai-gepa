"""Runtime helpers for executing the GEPA graph."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_graph.beta import Graph
from pydantic_graph.beta.graph import EndMarker, GraphTask

from ..adapter import Adapter
from ..exceptions import UsageBudgetExceeded
from .deps import GepaDeps
from .datasets import DatasetInput, resolve_dataset
from .graph import create_gepa_graph
from .helpers import create_deps
from .models import GepaConfig, GepaResult, GepaState
from ..progress import OptimizationProgress


logger = logging.getLogger(__name__)


async def optimize(
    *,
    adapter: Adapter[Any, Any, Any],
    config: GepaConfig,
    trainset: DatasetInput,
    valset: DatasetInput | None = None,
    seed_candidate: Mapping[str, str] | None = None,
    deps: GepaDeps | None = None,
        graph: Graph[GepaState, GepaDeps, None, GepaResult] | None = None,
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
        else create_gepa_graph(config=config)
    )
    training_loader = await resolve_dataset(trainset, name="trainset")
    validation_loader = await resolve_dataset(valset, name="valset") if valset is not None else None

    state = GepaState(
        config=config,
        training_set=training_loader,
        validation_set=validation_loader,
    )
    run_output = None
    try:
        with OptimizationProgress(
            total=config.max_evaluations,
            description="GEPA optimize",
            enabled=show_progress,
        ) as progress_bar:
            previous_node_name: str | None = None
            async with resolved_graph.iter(state=state, deps=resolved_deps) as run:
                async for event in run:
                    current_node_name = _describe_event(resolved_graph, event)
                    progress_bar.update(
                        state.total_evaluations,
                        current_node=current_node_name,
                        previous_node=previous_node_name,
                        best_score=state.best_score,
                    )
                    if current_node_name:
                        previous_node_name = current_node_name
                run_output = run.output
            progress_bar.update(
                state.total_evaluations,
                best_score=state.best_score,
            )
    except UsageBudgetExceeded:
        state.mark_stopped(reason="Usage budget exceeded")
        logger.info("GEPA run stopped early due to usage budget limit.")
        return GepaResult.from_state(state)

    if run_output is None:
        raise RuntimeError("GEPA graph run did not complete.")

    return run_output


def _describe_event(
    graph: Graph[GepaState, GepaDeps, None, GepaResult],
    event: EndMarker[GepaResult] | Sequence[GraphTask],
) -> str | None:
    if isinstance(event, EndMarker):
        return "End"

    node_ids = {task.node_id for task in event}
    if not node_ids:
        return None

    names = sorted(_node_label(graph, node_id) for node_id in node_ids)
    return ", ".join(names)


def _node_label(
    graph: Graph[GepaState, GepaDeps, None, GepaResult],
    node_id,
) -> str:
    node = graph.nodes.get(node_id)
    if node is None:
        return str(node_id)
    label = getattr(node, "label", None)
    if label:
        return label
    node_identifier = getattr(node, "id", None)
    if node_identifier is not None:
        return str(node_identifier)
    if hasattr(node, "__class__"):
        return node.__class__.__name__
    return str(node_id)


__all__ = ["optimize"]
