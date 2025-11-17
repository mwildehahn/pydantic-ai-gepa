"""Tests for deterministic batch sampling."""

from __future__ import annotations

from collections import Counter

import pytest

from pydantic_evals import Case

from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph.models import GepaConfig, GepaState
from pydantic_ai_gepa.gepa_graph.selectors import BatchSampler


def _make_training(count: int) -> list[Case[str, str, dict[str, str]]]:
    return [
        Case(name=f"case-{idx}", inputs=f"prompt-{idx}", metadata={"label": "stub"})
        for idx in range(count)
    ]


def _make_state(training_size: int) -> GepaState:
    training = _make_training(training_size)
    return GepaState(config=GepaConfig(), training_set=ListDataLoader(training))


@pytest.mark.asyncio
async def test_batch_sampler_deterministic_sequence() -> None:
    state = _make_state(5)
    sampler_a = BatchSampler(seed=7)
    sampler_b = BatchSampler(seed=7)

    batches_a = [await sampler_a.sample(state.training_set, state, 2) for _ in range(3)]
    batches_b = [await sampler_b.sample(state.training_set, state, 2) for _ in range(3)]

    ids_a = [[inst.name for inst in batch] for batch in batches_a]
    ids_b = [[inst.name for inst in batch] for batch in batches_b]

    assert ids_a == ids_b


@pytest.mark.asyncio
async def test_batch_sampler_padding_prefers_least_used() -> None:
    state = _make_state(3)
    sampler = BatchSampler(seed=1)

    batches = [await sampler.sample(state.training_set, state, 2) for _ in range(2)]
    counts = Counter(inst.name for batch in batches for inst in batch)

    # Because padding repeats only one element, ensure usage skew is at most 1.
    assert max(counts.values()) - min(counts.values()) <= 1


@pytest.mark.asyncio
async def test_batch_sampler_rejects_invalid_inputs() -> None:
    state = _make_state(1)
    sampler = BatchSampler()

    with pytest.raises(ValueError):
        await sampler.sample(ListDataLoader([]), state, 1)

    with pytest.raises(ValueError):
        await sampler.sample(state.training_set, state, 0)
