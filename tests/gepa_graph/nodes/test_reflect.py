"""Tests for the ReflectNode implementation."""

from __future__ import annotations

from typing import Mapping, Sequence, cast

import pytest
from pydantic_graph import GraphRunContext
from pydantic_ai.messages import UserPromptPart

from pydantic_ai_gepa.adapter import Adapter

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
)
from pydantic_ai_gepa.gepa_graph.selectors import (
    BatchSampler,
    CurrentBestCandidateSelector,
    RoundRobinComponentSelector,
)
from pydantic_ai_gepa.types import DataInst, DataInstWithPrompt, RolloutOutput


def _make_data(case_id: str) -> DataInstWithPrompt:
    return DataInstWithPrompt(
        user_prompt=UserPromptPart(content=f"prompt-{case_id}"),
        message_history=None,
        metadata={},
        case_id=case_id,
    )


def _make_state(
    *,
    minibatch_size: int = 2,
    config: GepaConfig | None = None,
) -> GepaState:
    if config is None:
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
    def __init__(
        self,
        *,
        dataset: dict[str, list[dict[str, str]]] | None = None,
    ) -> None:
        self.agent = type("Agent", (), {"_instructions": "seed"})()
        self.input_spec = None
        self._dataset = dataset or {"instructions": [{"feedback": "needs work"}]}
        self.dataset_calls = 0

    async def evaluate(self, batch, candidate, capture_traces):  # pragma: no cover
        raise RuntimeError("evaluate should not be called in ReflectNode tests")

    def make_reflective_dataset(self, *, candidate, eval_batch, components_to_update):
        self.dataset_calls += 1
        return {
            component: list(self._dataset.get(component, self._dataset["instructions"]))
            for component in components_to_update
        }


class _StubBatchSampler(BatchSampler):
    def __init__(self, batch):
        super().__init__(seed=0)
        self._batch = list(batch)
        self.calls = 0

    def sample(self, training_set, state, size):
        self.calls += 1
        return list(self._batch)


class _StubProposalGenerator(InstructionProposalGenerator):
    def __init__(self, updates: dict[str, str]) -> None:
        super().__init__()
        self._updates = updates
        self.calls = 0
        self.last_reflective_data: Mapping[str, Sequence[Mapping[str, object]]] | None = None

    async def propose_texts(self, *, candidate, reflective_data, components, model):
        self.calls += 1
        self.last_reflective_data = {k: list(v) for k, v in reflective_data.items()}
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
    adapter: Adapter[DataInst],
    evaluator: ParallelEvaluator,
    batch_sampler: BatchSampler,
    proposal_generator: InstructionProposalGenerator,
    reflection_model: str | None = "reflection-model",
) -> GepaDeps[DataInst]:
    return GepaDeps(
        adapter=adapter,
        evaluator=evaluator,
        pareto_manager=ParetoFrontManager(),
        candidate_selector=CurrentBestCandidateSelector(),
        component_selector=RoundRobinComponentSelector(),
        batch_sampler=batch_sampler,
        proposal_generator=proposal_generator,
        merge_builder=MergeProposalBuilder(),
        reflection_model=reflection_model,
    )


@pytest.mark.asyncio
async def test_reflect_node_accepts_strict_improvement() -> None:
    state = _make_state()
    minibatch = list(state.training_set)
    evaluator = _StubEvaluator([_eval_results([0.4, 0.5]), _eval_results([0.6, 0.7])])
    batch_sampler = _StubBatchSampler(minibatch)
    stub_adapter = _StubAdapter()
    adapter = cast(Adapter[DataInst], stub_adapter)
    generator = _StubProposalGenerator({"instructions": "Improved text"})
    deps = _make_deps(
        adapter=adapter,
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
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
    assert stub_adapter.dataset_calls == 1
    assert generator.calls == 1
    assert evaluator.calls == 2
    assert batch_sampler.calls == 1


@pytest.mark.asyncio
async def test_reflect_node_applies_config_sampler() -> None:
    sampler_calls: list[tuple[int, int]] = []

    def sampler(records: list[dict[str, object]], max_records: int) -> list[dict[str, object]]:
        sampler_calls.append((len(records), max_records))
        return records[:max_records]

    config = GepaConfig(
        max_evaluations=50,
        minibatch_size=2,
        merges_per_accept=1,
        perfect_score=1.0,
        skip_perfect_score=True,
        reflection_sampler=sampler,
        reflection_sampler_max_records=1,
    )
    state = _make_state(config=config)
    minibatch = list(state.training_set)
    evaluator = _StubEvaluator(
        [_eval_results([0.4, 0.5]), _eval_results([0.6, 0.7])]
    )
    batch_sampler = _StubBatchSampler(minibatch)
    dataset = {"instructions": [{"feedback": "a"}, {"feedback": "b"}]}
    stub_adapter = _StubAdapter(dataset=dataset)
    adapter = cast(Adapter[DataInst], stub_adapter)
    generator = _StubProposalGenerator({"instructions": "Improved text"})
    deps = _make_deps(
        adapter=adapter,
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
    )
    ctx = GraphRunContext(state=state, deps=deps)

    node = ReflectNode()
    await node.run(ctx)

    assert sampler_calls == [(2, 1)]
    assert generator.last_reflective_data is not None
    assert len(generator.last_reflective_data["instructions"]) == 1


@pytest.mark.asyncio
async def test_reflect_node_rejects_when_not_improved() -> None:
    state = _make_state()
    minibatch = list(state.training_set)
    evaluator = _StubEvaluator([_eval_results([0.6, 0.6]), _eval_results([0.6, 0.6])])
    batch_sampler = _StubBatchSampler(minibatch)
    stub_adapter = _StubAdapter()
    adapter = cast(Adapter[DataInst], stub_adapter)
    generator = _StubProposalGenerator({"instructions": "Same text"})
    deps = _make_deps(
        adapter=adapter,
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
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
    stub_adapter = _StubAdapter()
    adapter = cast(Adapter[DataInst], stub_adapter)
    generator = _StubProposalGenerator({"instructions": "Unused"})
    deps = _make_deps(
        adapter=adapter,
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
    )
    ctx = GraphRunContext(state=state, deps=deps)

    node = ReflectNode()
    result = await node.run(ctx)

    assert isinstance(result, ContinueNode)
    assert state.last_accepted is False
    assert len(state.candidates) == 1
    assert stub_adapter.dataset_calls == 0
    assert generator.calls == 0
    assert evaluator.calls == 1
    assert state.total_evaluations == 2


@pytest.mark.asyncio
async def test_reflect_node_requires_reflection_model() -> None:
    state = _make_state()
    minibatch = list(state.training_set)
    evaluator = _StubEvaluator([_eval_results([0.3, 0.4])])
    batch_sampler = _StubBatchSampler(minibatch)
    adapter = cast(Adapter[DataInst], _StubAdapter())
    generator = _StubProposalGenerator({"instructions": "Improved"})
    deps = _make_deps(
        adapter=adapter,
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
        reflection_model=None,
    )
    ctx = GraphRunContext(state=state, deps=deps)

    node = ReflectNode()
    with pytest.raises(ValueError, match="reflection_model"):
        await node.run(ctx)
