"""Tests for the high-level GEPA runtime helper."""

from __future__ import annotations

from typing import cast

import pytest

from pydantic_ai_gepa.adapter import Adapter
from pydantic_ai_gepa.gepa_graph import GepaConfig, optimize
from pydantic_ai_gepa.types import DataInst
from tests.gepa_graph.utils import AdapterStub, make_dataset


@pytest.mark.asyncio
async def test_optimize_completes_successfully() -> None:
    adapter = cast(Adapter[DataInst], AdapterStub())
    config = GepaConfig(
        max_evaluations=30,
        minibatch_size=2,
        seed=11,
        reflection_model="reflection-model",
    )
    dataset = make_dataset()

    result = await optimize(
        adapter=adapter,
        config=config,
        trainset=dataset,
        valset=dataset,
        seed_candidate={"instructions": "seed instructions"},
    )

    assert result.stopped is True
    assert result.best_score is not None
    assert result.original_score is not None
    assert result.best_score >= result.original_score
