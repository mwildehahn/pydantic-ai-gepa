"""Tests for candidate selection strategies."""

from __future__ import annotations

from pydantic_ai.messages import UserPromptPart

from pydantic_ai_gepa.gepa_graph.models import (
    CandidateProgram,
    ComponentValue,
    GepaConfig,
    GepaState,
    ParetoFrontEntry,
)
from pydantic_ai_gepa.gepa_graph.selectors import (
    CurrentBestCandidateSelector,
    ParetoCandidateSelector,
)
from pydantic_ai_gepa.types import DataInstWithPrompt


def _make_data_inst(case_id: str) -> DataInstWithPrompt:
    return DataInstWithPrompt(
        user_prompt=UserPromptPart(content=f"prompt-{case_id}"),
        message_history=None,
        metadata={},
        case_id=case_id,
    )


def _build_state(num_candidates: int = 3) -> GepaState:
    config = GepaConfig()
    training = [_make_data_inst(str(i)) for i in range(5)]
    state = GepaState(config=config, training_set=training)
    for idx in range(num_candidates):
        candidate = CandidateProgram(
            idx=idx,
            components={"system": ComponentValue(name="system", text=f"sys-{idx}")},
            creation_type="seed" if idx == 0 else "reflection",
            discovered_at_iteration=idx,
            discovered_at_evaluation=idx,
        )
        state.add_candidate(candidate, auto_assign_idx=False)
    state.best_candidate_idx = 0 if num_candidates else None
    return state


def test_pareto_candidate_selector_is_deterministic() -> None:
    state = _build_state()
    state.pareto_front = {
        "case-1": ParetoFrontEntry(data_id="case-1", best_score=0.8, candidate_indices={0, 1}),
        "case-2": ParetoFrontEntry(data_id="case-2", best_score=0.9, candidate_indices={1}),
    }

    selector_a = ParetoCandidateSelector(seed=42)
    selector_b = ParetoCandidateSelector(seed=42)

    seq_a = [selector_a.select(state) for _ in range(5)]
    seq_b = [selector_b.select(state) for _ in range(5)]

    assert seq_a == seq_b
    assert set(seq_a).issubset({0, 1})


def test_pareto_candidate_selector_falls_back_to_best() -> None:
    state = _build_state()
    state.pareto_front = {}
    state.best_candidate_idx = 2

    selector = ParetoCandidateSelector(seed=0)
    assert selector.select(state) == 2


def test_current_best_candidate_selector_defaults_to_zero() -> None:
    state = _build_state()
    selector = CurrentBestCandidateSelector()
    state.best_candidate_idx = None

    assert selector.select(state) == 0
