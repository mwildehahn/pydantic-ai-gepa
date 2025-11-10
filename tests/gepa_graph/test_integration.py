"""Integration tests that exercise checkpointing and resumption."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic_graph import FullStatePersistence

from pydantic_ai_gepa.adapter import AgentAdapter
from pydantic_ai_gepa.gepa_graph import create_deps, create_gepa_graph
from pydantic_ai_gepa.gepa_graph.models import GepaConfig, GepaState
from pydantic_ai_gepa.gepa_graph.nodes import EvaluateNode, StartNode
from tests.gepa_graph.utils import AdapterStub, ProposalGeneratorStub, make_dataset


@pytest.mark.asyncio
async def test_checkpoint_resume_restores_progress() -> None:
    adapter = cast(AgentAdapter[Any], AdapterStub())
    config = GepaConfig(max_evaluations=40, minibatch_size=2, seed=42)
    deps = create_deps(adapter, config)
    deps.proposal_generator = cast(Any, ProposalGeneratorStub())

    graph = create_gepa_graph(adapter=adapter, config=config)
    dataset = make_dataset()
    state = GepaState(config=config, training_set=dataset, validation_set=dataset)

    persistence = FullStatePersistence()

    executed_before: list[str] = []
    async with graph.iter(
        StartNode(), state=state, deps=deps, persistence=persistence
    ) as run:
        async for node in run:
            executed_before.append(node.__class__.__name__)
            if isinstance(node, EvaluateNode) and state.iteration >= 1:
                break

    assert run.result is None
    assert "ReflectNode" in executed_before
    assert len(persistence.history) > 0

    resumed_nodes: list[str] = []
    async with graph.iter_from_persistence(persistence, deps=deps) as resumed_run:
        async for node in resumed_run:
            resumed_nodes.append(node.__class__.__name__)

    resume_result = resumed_run.result
    assert resume_result is not None
    outcome = resume_result.output
    assert outcome.stopped is True
    assert outcome.best_score is not None
    assert outcome.original_score is not None
    assert outcome.best_score > outcome.original_score
    assert any(name == "ContinueNode" for name in resumed_nodes), (
        "resumed run should continue past saved node"
    )
