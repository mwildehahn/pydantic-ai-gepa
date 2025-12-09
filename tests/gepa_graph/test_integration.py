"""Integration tests for the GEPA graph end-to-end run."""

from __future__ import annotations

from typing import Any, cast

import pytest

from pydantic_ai_gepa.adapter import Adapter
from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph import create_deps, create_gepa_graph
from pydantic_ai_gepa.gepa_graph.models import ComponentValue, GepaConfig, GepaState
from pydantic_ai_gepa.types import ReflectionConfig
from tests.gepa_graph.utils import AdapterStub, ProposalGeneratorStub, make_dataset


@pytest.mark.asyncio
async def test_graph_run_produces_improved_candidate() -> None:
    adapter = cast(Adapter[str, str, dict[str, str]], AdapterStub())
    config = GepaConfig(
        max_evaluations=40,
        minibatch_size=2,
        seed=42,
        reflection_config=ReflectionConfig(model="reflection-model"),
    )
    deps = create_deps(
        adapter,
        config,
        seed_candidate={
            "instructions": ComponentValue(
                name="instructions", text="seed instructions"
            )
        },
    )
    deps.proposal_generator = cast(Any, ProposalGeneratorStub())

    graph = create_gepa_graph(config=config)
    dataset = make_dataset()
    state = GepaState(
        config=config,
        training_set=ListDataLoader(dataset),
        validation_set=ListDataLoader(dataset),
    )

    result = await graph.run(state=state, deps=deps)

    assert result.stopped is True
    assert result.best_score is not None
    assert result.original_score is not None
    assert result.best_score >= result.original_score
    assert state.total_evaluations > 0
