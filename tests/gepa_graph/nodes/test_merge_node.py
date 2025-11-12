"""Tests for the MergeNode implementation."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import UserPromptPart
from pydantic_graph import GraphRunContext

from typing import Literal, Sequence, cast

from pydantic_ai_gepa.gepa_graph.deps import GepaDeps
from pydantic_ai_gepa.gepa_graph.evaluation import (
    EvaluationResults,
    ParetoFrontManager,
    ParallelEvaluator,
)
from pydantic_ai_gepa.gepa_graph.models import (
    CandidateProgram,
    ComponentValue,
    GepaConfig,
    GepaState,
)
from pydantic_ai_gepa.gepa_graph.nodes import ContinueNode, EvaluateNode, MergeNode
from pydantic_ai_gepa.gepa_graph.proposal import (
    InstructionProposalGenerator,
    MergeProposalBuilder,
)
from pydantic_ai_gepa.gepa_graph.selectors import (
    BatchSampler,
    CurrentBestCandidateSelector,
    RoundRobinComponentSelector,
)
from pydantic_ai_gepa.types import DataInst, DataInstWithPrompt, RolloutOutput
from pydantic_ai_gepa.adapter import Adapter, SharedReflectiveDataset


def _make_data_inst(case_id: str) -> DataInstWithPrompt:
    return DataInstWithPrompt(
        user_prompt=UserPromptPart(content=f"prompt-{case_id}"),
        message_history=None,
        metadata={},
        case_id=case_id,
    )


def _make_state() -> GepaState:
    validation = [_make_data_inst(f"case-{idx}") for idx in range(5)]
    config = GepaConfig(
        max_evaluations=100,
        use_merge=True,
        min_shared_validation=3,
    )
    return GepaState(config=config, training_set=validation, validation_set=validation)


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
            "instructions": ComponentValue(
                name="instructions", text=instructions, version=iteration
            ),
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


def _build_lineage(state: GepaState) -> tuple[int, int, int]:
    ancestor = _add_candidate(
        state,
        instructions="Seed instructions",
        tools="Seed tools",
        creation_type="seed",
        parent_indices=[],
        iteration=0,
    )
    _populate_scores(ancestor, [0.5, 0.5, 0.4, 0.3, 0.3])

    parent1 = _add_candidate(
        state,
        instructions="Parent1 instructions",
        tools="Seed tools",
        creation_type="reflection",
        parent_indices=[ancestor.idx],
        iteration=1,
    )
    _populate_scores(parent1, [0.7, 0.6, 0.5, 0.5, 0.4])

    parent2 = _add_candidate(
        state,
        instructions="Seed instructions",
        tools="Parent2 tools",
        creation_type="reflection",
        parent_indices=[ancestor.idx],
        iteration=2,
    )
    _populate_scores(parent2, [0.6, 0.7, 0.6, 0.4, 0.5])

    return ancestor.idx, parent1.idx, parent2.idx


class _StubAdapter:
    reflection_model = "test-model"
    reflection_sampler = None
    agent = type("Agent", (), {"_instructions": "seed"})()
    input_spec = None

    async def evaluate(self, batch, candidate, capture_traces):  # pragma: no cover
        raise RuntimeError("evaluate should not be called in MergeNode tests")

    def make_reflective_dataset(
        self,
        *,
        candidate,
        eval_batch,
        components_to_update: Sequence[str],
    ) -> SharedReflectiveDataset:  # pragma: no cover
        return SharedReflectiveDataset(records=[])

    def get_components(self) -> dict[str, str]:  # pragma: no cover
        return {"instructions": "seed"}


class _StubEvaluator(ParallelEvaluator):
    def __init__(self, result: EvaluationResults[str]) -> None:
        self._result = result
        self.calls = 0

    async def evaluate_batch(self, **kwargs):
        self.calls += 1
        return self._result


class _StubMergeBuilder(MergeProposalBuilder):
    def __init__(
        self,
        *,
        pair: tuple[int, int],
        ancestor_idx: int,
        merged_candidate: CandidateProgram,
        subsample: list[DataInstWithPrompt],
        register_returns: bool = True,
    ) -> None:
        super().__init__(seed=0)
        self._pair = pair
        self._ancestor_idx = ancestor_idx
        self._merged_candidate = merged_candidate
        self._subsample = [cast(DataInst, inst) for inst in subsample]
        self._register_returns = register_returns

    def find_merge_pair(
        self, state: GepaState, dominators: Sequence[int]
    ) -> tuple[int, int] | None:  # noqa: D401
        return self._pair

    def find_common_ancestor(
        self, state: GepaState, idx1: int, idx2: int
    ) -> int | None:  # noqa: D401
        return self._ancestor_idx

    def build_merged_candidate(  # noqa: D401
        self,
        state: GepaState,
        parent1_idx: int,
        parent2_idx: int,
        ancestor_idx: int,
    ) -> CandidateProgram:
        return self._merged_candidate

    def select_merge_subsample(
        self,
        state: GepaState,
        parent1_idx: int,
        parent2_idx: int,
    ) -> list[DataInst]:
        return list(self._subsample)

    def register_candidate(  # noqa: D401
        self,
        *,
        candidate: CandidateProgram,
        parent1_idx: int,
        parent2_idx: int,
    ) -> bool:
        return self._register_returns


def _make_deps(
    *,
    merge_builder: MergeProposalBuilder,
    evaluator: ParallelEvaluator,
) -> GepaDeps[DataInst]:
    return GepaDeps(
        adapter=cast(Adapter[DataInst], _StubAdapter()),
        evaluator=evaluator,
        pareto_manager=ParetoFrontManager(),
        candidate_selector=CurrentBestCandidateSelector(),
        component_selector=RoundRobinComponentSelector(),
        batch_sampler=BatchSampler(),
        proposal_generator=InstructionProposalGenerator(),
        merge_builder=merge_builder,
        reflection_model="test-model",
    )


def _evaluation_results(
    subsample: Sequence[DataInstWithPrompt], scores: list[float]
) -> EvaluationResults[str]:
    return EvaluationResults(
        data_ids=[inst.case_id for inst in subsample],
        scores=list(scores),
        outputs=[RolloutOutput.from_success("merged")] * len(subsample),
        trajectories=None,
    )


@pytest.mark.asyncio
async def test_merge_node_accepts_when_scores_non_strictly_better() -> None:
    state = _make_state()
    ancestor_idx, parent1_idx, parent2_idx = _build_lineage(state)
    validation_set = cast(Sequence[DataInstWithPrompt], state.validation_set)
    subsample: list[DataInstWithPrompt] = list(validation_set)[:3]

    merged_candidate = CandidateProgram(
        idx=len(state.candidates),
        components={
            "instructions": ComponentValue(
                name="instructions", text="Parent1 instructions", version=3
            ),
            "tools": ComponentValue(name="tools", text="Parent2 tools", version=3),
        },
        creation_type="merge",
        parent_indices=[parent1_idx, parent2_idx],
        discovered_at_iteration=3,
        discovered_at_evaluation=10,
    )

    results = _evaluation_results(subsample, [0.7, 0.65, 0.6])
    evaluator = _StubEvaluator(results)
    builder = _StubMergeBuilder(
        pair=(parent1_idx, parent2_idx),
        ancestor_idx=ancestor_idx,
        merged_candidate=merged_candidate,
        subsample=subsample,
    )
    deps = _make_deps(merge_builder=builder, evaluator=evaluator)
    ctx = GraphRunContext(state=state, deps=deps)

    node = MergeNode()
    next_node = await node.run(ctx)

    assert isinstance(next_node, EvaluateNode)
    assert evaluator.calls == 1
    assert state.last_accepted is True
    assert len(state.candidates) == 4
    assert state.merge_attempts == 1
    new_candidate = state.candidates[-1]
    assert new_candidate.creation_type == "merge"
    assert new_candidate.parent_indices == [parent1_idx, parent2_idx]
    assert new_candidate.minibatch_scores == results.scores
    assert set(new_candidate.validation_scores) == {inst.case_id for inst in subsample}


@pytest.mark.asyncio
async def test_merge_node_rejects_when_merged_scores_lower() -> None:
    state = _make_state()
    ancestor_idx, parent1_idx, parent2_idx = _build_lineage(state)
    validation_set = cast(Sequence[DataInstWithPrompt], state.validation_set)
    subsample: list[DataInstWithPrompt] = list(validation_set)[:3]

    merged_candidate = CandidateProgram(
        idx=len(state.candidates),
        components={
            "instructions": ComponentValue(
                name="instructions", text="Parent1 instructions", version=3
            ),
            "tools": ComponentValue(name="tools", text="Parent2 tools", version=3),
        },
        creation_type="merge",
        parent_indices=[parent1_idx, parent2_idx],
        discovered_at_iteration=3,
        discovered_at_evaluation=10,
    )

    results = _evaluation_results(subsample, [0.5, 0.5, 0.45])
    evaluator = _StubEvaluator(results)
    builder = _StubMergeBuilder(
        pair=(parent1_idx, parent2_idx),
        ancestor_idx=ancestor_idx,
        merged_candidate=merged_candidate,
        subsample=subsample,
    )
    deps = _make_deps(merge_builder=builder, evaluator=evaluator)
    ctx = GraphRunContext(state=state, deps=deps)

    node = MergeNode()
    next_node = await node.run(ctx)

    assert isinstance(next_node, ContinueNode)
    assert evaluator.calls == 1
    assert state.last_accepted is False
    assert len(state.candidates) == 3
    assert state.merge_attempts == 1


@pytest.mark.asyncio
async def test_merge_node_skips_when_duplicate_detected() -> None:
    state = _make_state()
    ancestor_idx, parent1_idx, parent2_idx = _build_lineage(state)
    validation_set = cast(Sequence[DataInstWithPrompt], state.validation_set)
    subsample: list[DataInstWithPrompt] = list(validation_set)[:3]

    merged_candidate = CandidateProgram(
        idx=len(state.candidates),
        components={
            "instructions": ComponentValue(
                name="instructions", text="Parent1 instructions", version=3
            ),
            "tools": ComponentValue(name="tools", text="Parent2 tools", version=3),
        },
        creation_type="merge",
        parent_indices=[parent1_idx, parent2_idx],
        discovered_at_iteration=3,
        discovered_at_evaluation=10,
    )

    results = _evaluation_results(subsample, [0.7, 0.65, 0.6])
    evaluator = _StubEvaluator(results)
    builder = _StubMergeBuilder(
        pair=(parent1_idx, parent2_idx),
        ancestor_idx=ancestor_idx,
        merged_candidate=merged_candidate,
        subsample=subsample,
        register_returns=False,
    )
    deps = _make_deps(merge_builder=builder, evaluator=evaluator)
    ctx = GraphRunContext(state=state, deps=deps)

    node = MergeNode()
    next_node = await node.run(ctx)

    assert isinstance(next_node, ContinueNode)
    assert evaluator.calls == 0
    assert state.last_accepted is False
    assert len(state.candidates) == 3
    assert state.merge_attempts == 1
