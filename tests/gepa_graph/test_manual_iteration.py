"""End-to-end manual iteration tests for the GEPA graph."""

from __future__ import annotations

from typing import Any, cast

import pytest

from pydantic_ai_gepa.adapter import AgentAdapter
from pydantic_ai_gepa.gepa_graph import create_deps, create_gepa_graph
from pydantic_ai_gepa.gepa_graph.models import GepaConfig, GepaState
from pydantic_ai_gepa.gepa_graph.nodes import StartNode
from pydantic_ai_gepa.types import DataInst
from tests.gepa_graph.utils import (
    AdapterStub,
    ProposalGeneratorStub,
    make_dataset,
)


@pytest.mark.asyncio
async def test_manual_iteration_flow() -> None:
    adapter = cast(AgentAdapter[DataInst], AdapterStub())
    config = GepaConfig(
        max_evaluations=30,
        minibatch_size=2,
        seed=17,
        reflection_model="reflection-model",
    )
    deps = create_deps(adapter, config)
    deps.proposal_generator = cast(Any, ProposalGeneratorStub())

    graph = create_gepa_graph(adapter=adapter, config=config)
    dataset = make_dataset()
    state = GepaState(config=config, training_set=dataset, validation_set=dataset)

    executed_nodes: list[str] = []
    async with graph.iter(StartNode(), state=state, deps=deps) as run:
        async for node in run:
            executed_nodes.append(node.__class__.__name__)
            if state.best_score is not None and state.best_score >= 0.8:
                state.stopped = True
                state.stop_reason = "target score met"

    run_result = run.result
    assert run_result is not None
    result = run_result.output

    assert "StartNode" in executed_nodes
    assert "ReflectNode" in executed_nodes
    assert state.stop_reason == "target score met"
    assert result.stopped is True
    assert result.best_score is not None
    assert result.original_score is not None
    assert result.best_score > result.original_score
    assert len(result.candidates) >= 2
