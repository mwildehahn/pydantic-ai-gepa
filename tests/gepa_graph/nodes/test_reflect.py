"""Tests for the ReflectNode implementation."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic_graph import GraphRunContext
from pydantic_ai.messages import UserPromptPart

from pydantic_ai_gepa.adapter import AgentAdapter

from pydantic_ai_gepa.gepa_graph.deps import GepaDeps
from pydantic_ai_gepa.gepa_graph.evaluation import (
    EvaluationResults,
    ParallelEvaluator,
    ParetoFrontManager,
)
from pydantic_ai_gepa.gepa_graph.models import (
    CandidateProgram,
    ComponentValue,
    GepaConfig,
    GepaState,
)
from pydantic_ai_gepa.gepa_graph.nodes import ContinueNode, EvaluateNode, ReflectNode
from pydantic_ai_gepa.gepa_graph.proposal import (
    InstructionProposalGenerator,
    MergeProposalBuilder,
    ReflectiveDatasetBuilder,
)
from pydantic_ai_gepa.gepa_graph.selectors import (
    BatchSampler,
    CurrentBestCandidateSelector,
    RoundRobinComponentSelector,
)
from pydantic_ai_gepa.types import DataInstWithPrompt, RolloutOutput


def _make_data(case_id: str) -> DataInstWithPrompt:
    return DataInstWithPrompt(
        user_prompt=UserPromptPart(content=f"prompt-{case_id}"),
        message_history=None,
        metadata={},
        case_id=case_id,
    )


def _make_state(*, minibatch_size: int = 2) -> GepaState:
    config = GepaConfig(
        max_evaluations=100,
        minibatch_size=minibatch_size,
        merges_per_accept=2,
        perfect_score=1.0,
        skip_perfect_score=True,
    )
    training = [_make_data("a"), _make_data("b")]
    state = GepaState(config=config, training_set=training, validation_set=training)
    seed = CandidateProgram(
        idx=0,
        components={
            "instructions": ComponentValue(
                name="instructions", text="Seed instructions"
            ),
        },
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )
    state.add_candidate(seed, auto_assign_idx=False)
    state.iteration = 1
    return state


def _eval_results(scores: list[float]) -> EvaluationResults[str]:
    outputs = [
        RolloutOutput.from_success(f"result-{idx}") for idx, _ in enumerate(scores)
    ]
    case_ids = [f"case-{idx}" for idx in range(len(scores))]
    return EvaluationResults(
        data_ids=case_ids,
        scores=list(scores),
        outputs=outputs,
        trajectories=None,
    )


class _StubAdapter:
    def __init__(self, reflection_model: str | None = "reflection-model") -> None:
        self.agent = type("Agent", (), {"_instructions": "seed"})()
        self.input_spec = None
        self.reflection_model = reflection_model
        self.reflection_sampler = None


class _StubBatchSampler(BatchSampler):
    def __init__(self, batch):
        super().__init__(seed=0)
        self._batch = list(batch)
        self.calls = 0

    def sample(self, training_set, state, size):
        self.calls += 1
        return list(self._batch)


class _StubDatasetBuilder(ReflectiveDatasetBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def build_dataset(self, *, eval_results, components):
        self.calls += 1
        return {component: [{"feedback": "needs work"}] for component in components}


class _StubProposalGenerator(InstructionProposalGenerator):
    def __init__(self, updates: dict[str, str]) -> None:
        super().__init__()
        self._updates = updates
        self.calls = 0

    async def propose_texts(self, *, candidate, reflective_data, components, model):
        self.calls += 1
        return {
            component: self._updates.get(
                component, candidate.components[component].text
            )
            for component in components
        }


class _StubEvaluator(ParallelEvaluator):
    def __init__(self, results: list[EvaluationResults[str]]) -> None:
        self._results = list(results)
        self.calls = 0

    async def evaluate_batch(self, **kwargs):
        if not self._results:
            raise AssertionError("No more evaluation results scheduled.")
        self.calls += 1
        return self._results.pop(0)


def _make_deps(
    *,
    evaluator: ParallelEvaluator,
    batch_sampler: BatchSampler,
    proposal_generator: InstructionProposalGenerator,
    dataset_builder: ReflectiveDatasetBuilder,
    reflection_model: str | None = "reflection-model",
) -> GepaDeps:
    return GepaDeps(
        adapter=cast(
            AgentAdapter[Any], _StubAdapter(reflection_model=reflection_model)
        ),
        evaluator=evaluator,
        pareto_manager=ParetoFrontManager(),
        candidate_selector=CurrentBestCandidateSelector(),
        component_selector=RoundRobinComponentSelector(),
        batch_sampler=batch_sampler,
        proposal_generator=proposal_generator,
        reflective_dataset_builder=dataset_builder,
        merge_builder=MergeProposalBuilder(),
        reflection_model=reflection_model,
    )


@pytest.mark.asyncio
async def test_reflect_node_accepts_strict_improvement() -> None:
    state = _make_state()
    minibatch = list(state.training_set)
    evaluator = _StubEvaluator([_eval_results([0.4, 0.5]), _eval_results([0.6, 0.7])])
    batch_sampler = _StubBatchSampler(minibatch)
    builder = _StubDatasetBuilder()
    generator = _StubProposalGenerator({"instructions": "Improved text"})
    deps = _make_deps(
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
        dataset_builder=builder,
    )
    ctx = GraphRunContext(state=state, deps=deps)

    node = ReflectNode()
    result = await node.run(ctx)

    assert isinstance(result, EvaluateNode)
    assert state.last_accepted is True
    assert len(state.candidates) == 2
    new_candidate = state.candidates[-1]
    assert new_candidate.components["instructions"].text == "Improved text"
    assert new_candidate.components["instructions"].version == 1
    assert new_candidate.parent_indices == [0]
    assert new_candidate.minibatch_scores == [0.6, 0.7]
    assert state.merge_scheduled == state.config.merges_per_accept
    assert state.total_evaluations == 4
    assert builder.calls == 1
    assert generator.calls == 1
    assert evaluator.calls == 2
    assert batch_sampler.calls == 1


@pytest.mark.asyncio
async def test_reflect_node_rejects_when_not_improved() -> None:
    state = _make_state()
    minibatch = list(state.training_set)
    evaluator = _StubEvaluator([_eval_results([0.6, 0.6]), _eval_results([0.6, 0.6])])
    batch_sampler = _StubBatchSampler(minibatch)
    builder = _StubDatasetBuilder()
    generator = _StubProposalGenerator({"instructions": "Same text"})
    deps = _make_deps(
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
        dataset_builder=builder,
    )
    ctx = GraphRunContext(state=state, deps=deps)

    node = ReflectNode()
    result = await node.run(ctx)

    assert isinstance(result, ContinueNode)
    assert state.last_accepted is False
    assert len(state.candidates) == 1
    assert state.merge_scheduled == 0
    assert state.total_evaluations == 4


@pytest.mark.asyncio
async def test_reflect_node_skips_when_batch_is_perfect() -> None:
    state = _make_state()
    minibatch = list(state.training_set)
    evaluator = _StubEvaluator([_eval_results([1.0, 1.0])])
    batch_sampler = _StubBatchSampler(minibatch)
    builder = _StubDatasetBuilder()
    generator = _StubProposalGenerator({"instructions": "Unused"})
    deps = _make_deps(
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
        dataset_builder=builder,
    )
    ctx = GraphRunContext(state=state, deps=deps)

    node = ReflectNode()
    result = await node.run(ctx)

    assert isinstance(result, ContinueNode)
    assert state.last_accepted is False
    assert len(state.candidates) == 1
    assert builder.calls == 0
    assert generator.calls == 0
    assert evaluator.calls == 1
    assert state.total_evaluations == 2


@pytest.mark.asyncio
async def test_reflect_node_requires_reflection_model() -> None:
    state = _make_state()
    minibatch = list(state.training_set)
    evaluator = _StubEvaluator([_eval_results([0.3, 0.4])])
    batch_sampler = _StubBatchSampler(minibatch)
    builder = _StubDatasetBuilder()
    generator = _StubProposalGenerator({"instructions": "Improved"})
    deps = _make_deps(
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
        dataset_builder=builder,
        reflection_model=None,
    )
    ctx = GraphRunContext(state=state, deps=deps)

    node = ReflectNode()
    with pytest.raises(ValueError, match="reflection model"):
        await node.run(ctx)
