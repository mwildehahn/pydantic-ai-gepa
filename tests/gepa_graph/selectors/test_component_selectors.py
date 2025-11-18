"""Tests for component selection strategies."""

from __future__ import annotations

from pydantic_evals import Case

from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph.models import (
    CandidateProgram,
    ComponentValue,
    GepaConfig,
    GepaState,
)
from pydantic_ai_gepa.gepa_graph.selectors import (
    AllComponentSelector,
    RoundRobinComponentSelector,
)


def _make_state() -> GepaState:
    training = [
        Case(name=f"case-{idx}", inputs=f"prompt-{idx}", metadata={})
        for idx in range(2)
    ]
    state = GepaState(config=GepaConfig(), training_set=ListDataLoader(training))
    candidate = CandidateProgram(
        idx=0,
        components={
            "system": ComponentValue(name="system", text="sys"),
            "user": ComponentValue(name="user", text="usr"),
            "instructions": ComponentValue(name="instructions", text="inst"),
        },
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )
    state.add_candidate(candidate, auto_assign_idx=False)
    return state


def test_round_robin_component_selector_cycles() -> None:
    state = _make_state()
    selector = RoundRobinComponentSelector()

    selections = [selector.select(state, 0)[0] for _ in range(5)]
    assert selections[:3] == ["system", "user", "instructions"]
    assert selections[3:] == ["system", "user"]


def test_all_component_selector_returns_all_components() -> None:
    state = _make_state()
    selector = AllComponentSelector()

    selected = selector.select(state, 0)
    assert set(selected) == {"system", "user", "instructions"}
