"""Graph construction helpers for the native GEPA implementation."""

from __future__ import annotations

from types import NoneType
from typing import Any, Literal

from pydantic_graph.beta import Graph, GraphBuilder, StepContext
from pydantic_graph.beta.util import TypeExpression

from .deps import GepaDeps
from .models import GepaConfig, GepaResult, GepaState
from .nodes import StopSignal, continue_node, evaluate_node, merge_node, reflect_node, start_node


def create_gepa_graph(
    *,
    config: GepaConfig,
) -> Graph[GepaState, GepaDeps[Any], None, GepaResult]:
    """Create the GEPA graph definition based on the provided configuration."""

    builder = GraphBuilder(
        name="gepa_graph",
        state_type=GepaState,
        deps_type=GepaDeps[Any],
        input_type=NoneType,
        output_type=GepaResult,
    )

    start_step = builder.step(start_node, node_id="StartNode")
    evaluate_step = builder.step(evaluate_node, node_id="EvaluateNode")
    continue_step = builder.step(continue_node, node_id="ContinueNode")
    reflect_step = builder.step(reflect_node, node_id="ReflectNode")
    merge_step = builder.step(merge_node, node_id="MergeNode")

    continue_decision = (
        builder.decision(node_id="ContinueDecision")
        .branch(builder.match(StopSignal).transform(_stop_signal_to_result).to(builder.end_node))
        .branch(
            builder.match(TypeExpression[Literal["merge"]])
            .transform(_drop_input)
            .to(merge_step)
        )
        .branch(
            builder.match(TypeExpression[Literal["reflect"]])
            .transform(_drop_input)
            .to(reflect_step)
        )
    )

    reflect_decision = _iteration_decision(builder, evaluate_step, continue_step, node_id="ReflectIterationDecision")
    merge_decision = _iteration_decision(builder, evaluate_step, continue_step, node_id="MergeIterationDecision")

    builder.add(
        builder.edge_from(builder.start_node).to(start_step),
        builder.edge_from(start_step).to(evaluate_step),
        builder.edge_from(evaluate_step).to(continue_step),
        builder.edge_from(continue_step).to(continue_decision),
        builder.edge_from(reflect_step).to(reflect_decision),
        builder.edge_from(merge_step).to(merge_decision),
    )

    return builder.build()


def _iteration_decision(
    builder: GraphBuilder[Any, Any, Any, Any],
    evaluate_step,
    continue_step,
    *,
    node_id: str,
):
    return (
        builder.decision(node_id=node_id)
        .branch(
            builder.match(TypeExpression[Literal["evaluate"]])
            .transform(_drop_input)
            .to(evaluate_step)
        )
        .branch(
            builder.match(TypeExpression[Literal["continue"]])
            .transform(_drop_input)
            .to(continue_step)
        )
    )


def _stop_signal_to_result(
    ctx: StepContext[GepaState, GepaDeps[Any], StopSignal]
) -> GepaResult:
    return ctx.inputs.result


def _drop_input(
    ctx: StepContext[GepaState, GepaDeps[Any], object]
) -> None:
    return None


__all__ = ["create_gepa_graph"]
