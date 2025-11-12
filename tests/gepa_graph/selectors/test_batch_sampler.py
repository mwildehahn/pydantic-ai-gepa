"""Tests for deterministic batch sampling."""

from __future__ import annotations

from collections import Counter

from pydantic_ai.messages import UserPromptPart

from pydantic_ai_gepa.gepa_graph.models import GepaConfig, GepaState
from pydantic_ai_gepa.gepa_graph.selectors import BatchSampler
from pydantic_ai_gepa.types import DataInstWithPrompt


def _make_training(count: int) -> list[DataInstWithPrompt]:
    return [
        DataInstWithPrompt(
            user_prompt=UserPromptPart(content=f"prompt-{idx}"),
            message_history=None,
            metadata={},
            case_id=str(idx),
        )
        for idx in range(count)
    ]


def _make_state(training_size: int) -> GepaState:
    training = _make_training(training_size)
    return GepaState(config=GepaConfig(), training_set=training)


def test_batch_sampler_deterministic_sequence() -> None:
    state = _make_state(5)
    sampler_a = BatchSampler(seed=7)
    sampler_b = BatchSampler(seed=7)

    batches_a = [sampler_a.sample(state.training_set, state, 2) for _ in range(3)]
    batches_b = [sampler_b.sample(state.training_set, state, 2) for _ in range(3)]

    ids_a = [[inst.case_id for inst in batch] for batch in batches_a]
    ids_b = [[inst.case_id for inst in batch] for batch in batches_b]

    assert ids_a == ids_b


def test_batch_sampler_padding_prefers_least_used() -> None:
    state = _make_state(3)
    sampler = BatchSampler(seed=1)

    batches = [sampler.sample(state.training_set, state, 2) for _ in range(2)]
    counts = Counter(inst.case_id for batch in batches for inst in batch)

    # Because padding repeats only one element, ensure usage skew is at most 1.
    assert max(counts.values()) - min(counts.values()) <= 1


def test_batch_sampler_rejects_invalid_inputs() -> None:
    state = _make_state(1)
    sampler = BatchSampler()

    try:
        sampler.sample([], state, 1)  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty training set.")

    try:
        sampler.sample(state.training_set, state, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for zero batch size.")
