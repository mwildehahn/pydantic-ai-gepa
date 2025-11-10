"""Tests for helper utilities in the GEPA graph package."""

from __future__ import annotations

from typing import Any, cast
import random

from pydantic_ai_gepa.adapter import AgentAdapter
from pydantic_ai_gepa.gepa_graph import create_deps
from pydantic_ai_gepa.gepa_graph.deps import GepaDeps
from pydantic_ai_gepa.gepa_graph.models import GepaConfig, GepaState
from pydantic_ai_gepa.gepa_graph.selectors import (
    AllComponentSelector,
    BatchSampler,
    CurrentBestCandidateSelector,
    ParetoCandidateSelector,
    RoundRobinComponentSelector,
)
from pydantic_ai_gepa.types import DataInstWithPrompt
from pydantic_ai.messages import UserPromptPart


class _AgentStub:
    def __init__(self) -> None:
        self._instructions = "seed"


class _AdapterStub:
    def __init__(self) -> None:
        self.agent = _AgentStub()
        self.input_spec = None
        self.reflection_model = "reflection-model"
        self.reflection_sampler = object()


def _make_adapter() -> AgentAdapter[Any]:
    return cast(AgentAdapter[Any], _AdapterStub())


def _make_state(config: GepaConfig) -> GepaState:
    dataset = [
        DataInstWithPrompt(
            user_prompt=UserPromptPart(content=f"prompt-{idx}"),
            message_history=None,
            metadata={},
            case_id=str(idx),
        )
        for idx in range(3)
    ]
    return GepaState(config=config, training_set=dataset, validation_set=dataset)


def test_create_deps_defaults() -> None:
    adapter = _make_adapter()
    config = GepaConfig(seed=7)

    deps = create_deps(adapter, config)

    assert isinstance(deps, GepaDeps)
    assert deps.adapter is adapter
    assert isinstance(deps.candidate_selector, ParetoCandidateSelector)
    assert isinstance(deps.component_selector, RoundRobinComponentSelector)
    assert isinstance(deps.batch_sampler, BatchSampler)
    assert isinstance(deps.merge_builder.seed, int)
    assert deps.merge_builder.seed == config.seed
    assert deps.reflection_model == adapter.reflection_model
    sampler = getattr(deps.reflective_dataset_builder, "_sampler")
    assert sampler is adapter.reflection_sampler

    # Batch sampler should respect the config seed for determinism.
    sampler_rng = getattr(deps.batch_sampler, "_rng")
    assert abs(sampler_rng.random() - random.Random(config.seed).random()) < 1e-9


def test_create_deps_supports_alternate_selectors() -> None:
    adapter = _make_adapter()
    config = GepaConfig(
        candidate_selector="current_best",
        component_selector="all",
    )

    deps = create_deps(adapter, config)

    assert isinstance(deps.candidate_selector, CurrentBestCandidateSelector)
    assert isinstance(deps.component_selector, AllComponentSelector)

    # Ensure helper returns usable dependencies for sampling.
    state = _make_state(config)
    batch = deps.batch_sampler.sample(state.training_set, state, size=2)
    assert len(batch) == 2
