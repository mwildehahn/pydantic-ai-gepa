"""Tests for the high-level GEPA runtime helper."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from pydantic_ai_gepa.adapter import Adapter
from pydantic_ai_gepa.gepa_graph import GepaConfig, optimize
from pydantic_ai_gepa.types import ReflectionConfig
from tests.gepa_graph.utils import AdapterStub, make_dataset


@pytest.mark.asyncio
async def test_optimize_completes_successfully() -> None:
    adapter = cast(Adapter[str, str, dict[str, str]], AdapterStub())
    config = GepaConfig(
        max_evaluations=30,
        minibatch_size=2,
        seed=11,
        reflection_config=ReflectionConfig(model="reflection-model"),
    )
    dataset = make_dataset()

    result = await optimize(
        adapter=adapter,
        config=config,
        trainset=dataset,
        valset=dataset,
    )

    assert result.stopped is True
    assert result.best_score is not None
    assert result.original_score is not None
    assert result.best_score >= result.original_score


@pytest.mark.asyncio
async def test_optimize_supports_async_dataset_loader() -> None:
    adapter = cast(Adapter[str, str, dict[str, str]], AdapterStub())
    config = GepaConfig(
        max_evaluations=20,
        minibatch_size=1,
        seed=3,
        reflection_config=ReflectionConfig(model="reflection-model"),
    )

    async def load_remote_dataset():
        await asyncio.sleep(0)
        return make_dataset(2)

    result = await optimize(
        adapter=adapter,
        config=config,
        trainset=load_remote_dataset,
    )

    assert result.best_score is not None
