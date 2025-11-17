from __future__ import annotations

from typing import Literal

import pytest
from pydantic_evals import Case

from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph.models import CandidateProgram, ComponentValue, GepaConfig, GepaState
from pydantic_ai_gepa.gepa_graph.proposal import MergeProposalBuilder
from pydantic_ai_gepa.types import RolloutOutput


def _make_data_inst(case_id: str) -> Case[str, str, dict[str, str]]:
    return Case(name=case_id, inputs=f"prompt-{case_id}", metadata={})


def _make_state(
    *,
    min_shared_validation: int = 4,
    merge_subsample_size: int = 5,
) -> GepaState:
    training = [_make_data_inst(f"case-{idx}") for idx in range(6)]
    config = GepaConfig(
        max_evaluations=100,
        use_merge=True,
        min_shared_validation=min_shared_validation,
        merge_subsample_size=merge_subsample_size,
    )
    return GepaState(
        config=config,
        training_set=ListDataLoader(training),
        validation_set=ListDataLoader(training),
    )


def _add_candidate(
    state: GepaState,
    *,
    instructions: str,
    tools: str,
    creation_type: Literal["seed", "reflection", "merge"],
    parent_indices: list[int],
    iteration: int,
) -> CandidateProgram:
    idx = len(state.candidates)
    candidate = CandidateProgram(
        idx=idx,
        components={
            "instructions": ComponentValue(name="instructions", text=instructions, version=iteration),
            "tools": ComponentValue(name="tools", text=tools, version=iteration),
        },
        creation_type=creation_type,
        parent_indices=parent_indices,
        discovered_at_iteration=iteration,
        discovered_at_evaluation=iteration,
    )
    state.add_candidate(candidate, auto_assign_idx=False)
    return candidate


def _populate_scores(candidate: CandidateProgram, scores: list[float]) -> None:
    for idx, score in enumerate(scores):
        candidate.record_validation(
            data_id=f"case-{idx}",
            score=score,
            output=RolloutOutput.from_success(f"{candidate.idx}-{idx}"),
        )


def _build_lineage(state: GepaState) -> tuple[CandidateProgram, CandidateProgram, CandidateProgram]:
    ancestor = _add_candidate(
        state,
        instructions="Seed instructions",
        tools="Seed tools",
        creation_type="seed",
        parent_indices=[],
        iteration=0,
    )
    _populate_scores(ancestor, [0.5, 0.5, 0.4, 0.4, 0.3, 0.3])

    parent1 = _add_candidate(
        state,
        instructions="Parent1 instructions",
        tools="Seed tools",
        creation_type="reflection",
        parent_indices=[ancestor.idx],
        iteration=1,
    )
    _populate_scores(parent1, [0.7, 0.6, 0.5, 0.5, 0.4, 0.4])

    parent2 = _add_candidate(
        state,
        instructions="Seed instructions",
        tools="Parent2 tools",
        creation_type="reflection",
        parent_indices=[ancestor.idx],
        iteration=2,
    )
    _populate_scores(parent2, [0.6, 0.7, 0.6, 0.4, 0.5, 0.5])

    return ancestor, parent1, parent2


def test_find_merge_pair_orders_indices() -> None:
    state = _make_state()
    builder = MergeProposalBuilder(seed=0)
    dominators = [5, 3, 9, 1]
    pair = builder.find_merge_pair(state, dominators)
    assert pair is not None
    assert pair[0] < pair[1]
    assert pair[0] in dominators and pair[1] in dominators


def test_find_common_ancestor_requires_desirable_predictor() -> None:
    state = _make_state()
    ancestor, parent1, parent2 = _build_lineage(state)
    builder = MergeProposalBuilder(seed=42)

    result = builder.find_common_ancestor(state, parent1.idx, parent2.idx)
    assert result == ancestor.idx

    # If parents identical to ancestor there is no desirable predictor.
    clone1 = _add_candidate(
        state,
        instructions="Seed instructions",
        tools="Seed tools",
        creation_type="reflection",
        parent_indices=[ancestor.idx],
        iteration=3,
    )
    _populate_scores(clone1, [0.6] * 6)

    clone2 = _add_candidate(
        state,
        instructions="Seed instructions",
        tools="Seed tools",
        creation_type="reflection",
        parent_indices=[ancestor.idx],
        iteration=4,
    )
    _populate_scores(clone2, [0.6] * 6)

    assert builder.find_common_ancestor(state, clone1.idx, clone2.idx) is None


def test_build_merged_candidate_combines_components() -> None:
    state = _make_state()
    ancestor, parent1, parent2 = _build_lineage(state)
    builder = MergeProposalBuilder(seed=1)

    merged = builder.build_merged_candidate(
        state=state,
        parent1_idx=parent1.idx,
        parent2_idx=parent2.idx,
        ancestor_idx=ancestor.idx,
    )

    assert merged.creation_type == "merge"
    assert merged.parent_indices == [parent1.idx, parent2.idx]
    assert merged.components["instructions"].text == parent1.components["instructions"].text
    assert merged.components["tools"].text == parent2.components["tools"].text


@pytest.mark.asyncio
async def test_select_merge_subsample_stratifies_scores() -> None:
    state = _make_state(min_shared_validation=4, merge_subsample_size=5)
    _, parent1, parent2 = _build_lineage(state)
    builder = MergeProposalBuilder(seed=5)

    subsample = await builder.select_merge_subsample(state, parent1_idx=parent1.idx, parent2_idx=parent2.idx)
    assert len(subsample) == state.config.merge_subsample_size
    case_ids = {inst.name for _, inst in subsample}
    assert case_ids.issubset({f"case-{idx}" for idx in range(6)})


@pytest.mark.asyncio
async def test_select_merge_subsample_returns_empty_when_insufficient_overlap() -> None:
    state = _make_state(min_shared_validation=5)
    ancestor = _add_candidate(
        state,
        instructions="Seed instructions",
        tools="Seed tools",
        creation_type="seed",
        parent_indices=[],
        iteration=0,
    )
    _populate_scores(ancestor, [0.5] * 3)
    parent1 = _add_candidate(
        state,
        instructions="Parent1 instructions",
        tools="Seed tools",
        creation_type="reflection",
        parent_indices=[ancestor.idx],
        iteration=1,
    )
    _populate_scores(parent1, [0.6] * 3)
    parent2 = _add_candidate(
        state,
        instructions="Seed instructions",
        tools="Parent2 tools",
        creation_type="reflection",
        parent_indices=[ancestor.idx],
        iteration=2,
    )
    _populate_scores(parent2, [0.7] * 2)  # Less overlap than required

    builder = MergeProposalBuilder(seed=7)
    subsample = await builder.select_merge_subsample(state, parent1_idx=parent1.idx, parent2_idx=parent2.idx)
    assert subsample == []


def test_register_candidate_deduplicates_same_merge() -> None:
    state = _make_state()
    ancestor, parent1, parent2 = _build_lineage(state)
    builder = MergeProposalBuilder(seed=2)
    merged = builder.build_merged_candidate(state, parent1.idx, parent2.idx, ancestor.idx)

    assert builder.register_candidate(candidate=merged, parent1_idx=parent1.idx, parent2_idx=parent2.idx) is True
    assert builder.register_candidate(candidate=merged, parent1_idx=parent1.idx, parent2_idx=parent2.idx) is False
