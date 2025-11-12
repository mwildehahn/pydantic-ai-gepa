"""Tests for the GEPA state model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import UserPromptPart

from pydantic_ai_gepa.gepa_graph.models import (
    CandidateProgram,
    ComponentValue,
    GepaConfig,
    GepaState,
)
from pydantic_ai_gepa.types import DataInstWithPrompt, RolloutOutput


def _make_data_inst(case_id: str) -> DataInstWithPrompt:
    return DataInstWithPrompt(
        user_prompt=UserPromptPart(content=f"prompt-{case_id}"),
        message_history=None,
        metadata={},
        case_id=case_id,
    )


def test_state_requires_training_set() -> None:
    config = GepaConfig()
    with pytest.raises(ValidationError):
        GepaState(config=config, training_set=[])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_state_defaults_validation_set_to_training_set() -> None:
    config = GepaConfig()
    training_set = [_make_data_inst("1"), _make_data_inst("2")]

    state = GepaState(
        config=config,
        training_set=training_set,
    )

    assert len(state.training_set) == 2
    validation_loader = state.validation_set
    assert validation_loader is not None
    assert validation_loader is state.training_set
    ids = await validation_loader.all_ids()
    fetched = await validation_loader.fetch(ids)
    assert fetched == training_set


def test_state_add_candidate_and_genealogy() -> None:
    config = GepaConfig()
    training_set = [_make_data_inst("1")]
    state = GepaState(config=config, training_set=training_set)

    candidate = CandidateProgram(
        idx=99,  # intentionally wrong, should be reassigned
        components={"system": ComponentValue(name="system", text="Hello")},
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )

    stored = state.add_candidate(candidate)
    assert stored.idx == 0
    assert len(state.candidates) == 1
    assert state.genealogy[0].candidate_idx == 0
    assert state.genealogy[0].creation_type == "seed"


def test_state_recompute_best_candidate() -> None:
    config = GepaConfig()
    training_set = [_make_data_inst("1"), _make_data_inst("2"), _make_data_inst("3")]
    state = GepaState(config=config, training_set=training_set)

    cand_a = CandidateProgram(
        idx=0,
        components={"system": ComponentValue(name="system", text="A")},
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )
    cand_a.record_validation(
        data_id="1",
        score=0.5,
        output=RolloutOutput.from_success("A1"),
    )

    cand_b = CandidateProgram(
        idx=1,
        components={"system": ComponentValue(name="system", text="B")},
        creation_type="reflection",
        parent_indices=[0],
        discovered_at_iteration=1,
        discovered_at_evaluation=1,
    )
    cand_b.record_validation(
        data_id="1",
        score=0.8,
        output=RolloutOutput.from_success("B1"),
    )
    cand_b.record_validation(
        data_id="2",
        score=0.9,
        output=RolloutOutput.from_success("B2"),
    )

    state.add_candidate(cand_a, auto_assign_idx=False)
    state.add_candidate(cand_b, auto_assign_idx=False)

    best = state.recompute_best_candidate()
    assert best is cand_b
    assert state.best_candidate_idx == 1
    assert state.best_score == pytest.approx(0.85)
