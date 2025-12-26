from __future__ import annotations

import pytest

from pydantic_evals import Case

from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph.models import GepaConfig, GepaState


def _make_state() -> GepaState:
    dataset = [Case(name="case-1", inputs="x", metadata=None)]
    return GepaState(
        config=GepaConfig(),
        training_set=ListDataLoader(dataset),
        validation_set=ListDataLoader(dataset),
    )


def test_state_skill_activation_registry_defaults_empty() -> None:
    state = _make_state()
    assert state.active_skill_paths == set()


def test_state_activate_skill_path_is_idempotent() -> None:
    state = _make_state()
    assert state.activate_skill_path("index/tasks") is True
    assert state.activate_skill_path("index/tasks") is False
    assert state.is_skill_active("index/tasks") is True
    assert state.is_skill_active("index/missing") is False


def test_state_activate_skill_path_rejects_empty() -> None:
    state = _make_state()
    with pytest.raises(ValueError):
        state.activate_skill_path("")
