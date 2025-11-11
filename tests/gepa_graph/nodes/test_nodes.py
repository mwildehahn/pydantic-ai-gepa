"""Tests for the start, evaluate, and continue nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import pytest
from pydantic_graph import End, GraphRunContext
from pydantic_ai.messages import UserPromptPart

from pydantic_ai_gepa.adapter import Adapter
from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapterTrajectory
from pydantic_ai_gepa.gepa_graph.deps import GepaDeps
from pydantic_ai_gepa.gepa_graph.evaluation import ParallelEvaluator, ParetoFrontManager
from pydantic_ai_gepa.gepa_graph.models import (
    CandidateProgram,
    ComponentValue,
    GepaConfig,
    GepaState,
)
from pydantic_ai_gepa.gepa_graph.nodes import (
    ContinueNode,
    EvaluateNode,
    MergeNode,
    ReflectNode,
    StartNode,
)
from pydantic_ai_gepa.types import DataInst, DataInstWithPrompt, RolloutOutput
from pydantic_ai_gepa.gepa_graph.selectors import (
    BatchSampler,
    CurrentBestCandidateSelector,
    RoundRobinComponentSelector,
)
from pydantic_ai_gepa.gepa_graph.proposal import (
    InstructionProposalGenerator,
    MergeProposalBuilder,
)


def _make_data_inst(case_id: str) -> DataInstWithPrompt:
    return DataInstWithPrompt(
        user_prompt=UserPromptPart(content=f"prompt-{case_id}"),
        message_history=None,
        metadata={},
        case_id=case_id,
    )


def _make_state(
    num_instances: int = 2,
    *,
    config: GepaConfig | None = None,
) -> GepaState:
    training = [_make_data_inst(str(i)) for i in range(num_instances)]
    config = config or GepaConfig(max_evaluations=10)
    return GepaState(config=config, training_set=training, validation_set=training)


@dataclass
class _FakeEvaluationBatch:
    outputs: list[RolloutOutput[str]]
    scores: list[float]
    trajectories: list[AgentAdapterTrajectory] | None = None


class _FakeAdapter:
    def __init__(self) -> None:
        self.agent = type("Agent", (), {"_instructions": "seed"})()
        self.input_spec = None
        self.scores: dict[str, float] = {}
        self.reflection_model = "test-model"
        self.reflection_sampler = None

    async def evaluate(self, batch, candidate, capture_traces):
        case_id = batch[0].case_id
        score = self.scores.get(case_id, 0.5)
        return _FakeEvaluationBatch(
            outputs=[RolloutOutput.from_success(candidate["instructions"])],
            scores=[score],
        )

    def make_reflective_dataset(
        self,
        *,
        candidate,
        eval_batch,
        components_to_update: Sequence[str],
    ) -> dict[str, list[dict]]:
        return {component: [] for component in components_to_update}

    def get_components(self) -> dict[str, str]:
        return {"instructions": "seed"}


class _HydratingAdapter(_FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._include_tool = False

    async def evaluate(self, batch, candidate, capture_traces):
        result = await super().evaluate(batch, candidate, capture_traces)
        self._include_tool = True
        return result

    def get_components(self) -> dict[str, str]:
        components = super().get_components()
        if self._include_tool:
            hydrated = dict(components)
            hydrated["tool:new"] = "desc"
            return hydrated
        return components


def _make_deps(
    seed_candidate: dict[str, str] | None = None,
) -> GepaDeps[DataInst]:
    return GepaDeps(
        adapter=cast(Adapter[DataInst], _FakeAdapter()),
        evaluator=ParallelEvaluator(),
        pareto_manager=ParetoFrontManager(),
        candidate_selector=CurrentBestCandidateSelector(),
        component_selector=RoundRobinComponentSelector(),
        batch_sampler=BatchSampler(),
        proposal_generator=InstructionProposalGenerator(),
        merge_builder=MergeProposalBuilder(),
        reflection_model="test-model",
        seed_candidate=seed_candidate,
    )


@pytest.mark.asyncio
async def test_start_node_adds_seed_candidate_from_deps() -> None:
    state = _make_state()
    deps = _make_deps(seed_candidate={"instructions": "hello world"})
    ctx = GraphRunContext(state=state, deps=deps)

    node = StartNode()
    result = await node.run(ctx)

    assert isinstance(result, EvaluateNode)
    assert len(state.candidates) == 1
    candidate = state.candidates[0]
    assert candidate.components["instructions"].text == "hello world"
    assert state.iteration == 0


@pytest.mark.asyncio
async def test_start_node_is_idempotent_when_candidates_exist() -> None:
    state = _make_state()
    deps = _make_deps(seed_candidate={"instructions": "hello"})
    state.add_candidate(
        CandidateProgram(
            idx=0,
            components={
                "instructions": ComponentValue(name="instructions", text="existing")
            },
            creation_type="seed",
            discovered_at_iteration=0,
            discovered_at_evaluation=0,
        )
    )
    state.iteration = 0
    ctx = GraphRunContext(state=state, deps=deps)

    node = StartNode()
    result = await node.run(ctx)

    assert isinstance(result, EvaluateNode)
    assert len(state.candidates) == 1


@pytest.mark.asyncio
async def test_start_node_uses_adapter_snapshot_when_seed_missing() -> None:
    state = _make_state()
    deps = _make_deps(seed_candidate=None)
    ctx = GraphRunContext(state=state, deps=deps)

    node = StartNode()
    result = await node.run(ctx)

    assert isinstance(result, EvaluateNode)
    assert len(state.candidates) == 1
    candidate = state.candidates[0]
    assert candidate.components["instructions"].text == "seed"
    assert deps.seed_candidate == {"instructions": "seed"}


@pytest.mark.asyncio
async def test_evaluate_node_updates_candidate_scores_and_state() -> None:
    state = _make_state(num_instances=3)
    candidate = CandidateProgram(
        idx=0,
        components={"instructions": ComponentValue(name="instructions", text="test")},
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )
    state.add_candidate(candidate)
    ctx = GraphRunContext(state=state, deps=_make_deps())

    node = EvaluateNode()
    result = await node.run(ctx)

    assert isinstance(result, ContinueNode)
    validation_set = state.validation_set
    assert validation_set is not None
    assert len(candidate.validation_scores) == len(validation_set)
    assert state.best_candidate_idx == 0
    assert state.total_evaluations == len(validation_set)
    assert state.full_validations == 1
    assert state.pareto_front


@pytest.mark.asyncio
async def test_evaluate_node_hydrates_new_components() -> None:
    state = _make_state(num_instances=1)
    candidate = CandidateProgram(
        idx=0,
        components={"instructions": ComponentValue(name="instructions", text="test")},
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )
    state.add_candidate(candidate)
    adapter = cast(Adapter[DataInst], _HydratingAdapter())
    deps = _make_deps()
    deps.adapter = adapter
    ctx = GraphRunContext(state=state, deps=deps)

    node = EvaluateNode()
    await node.run(ctx)

    assert "tool:new" in candidate.components
    assert candidate.components["tool:new"].text == "desc"
    assert deps.seed_candidate is not None
    assert deps.seed_candidate["tool:new"] == "desc"


@pytest.mark.asyncio
async def test_continue_node_returns_end_when_budget_spent() -> None:
    state = _make_state()
    state.total_evaluations = state.config.max_evaluations
    deps = _make_deps()
    ctx = GraphRunContext(state=state, deps=deps)

    node = ContinueNode()
    result = await node.run(ctx)

    assert isinstance(result, End)
    assert state.stop_reason == "Max evaluations reached"


@pytest.mark.asyncio
async def test_continue_node_triggers_merge_when_scheduled() -> None:
    config = GepaConfig(max_evaluations=10, use_merge=True)
    state = _make_state(config=config)
    state.iteration = 0
    state.merge_scheduled = 2
    state.last_accepted = True
    deps = _make_deps()
    ctx = GraphRunContext(state=state, deps=deps)

    node = ContinueNode()
    next_node = await node.run(ctx)

    assert isinstance(next_node, MergeNode)
    assert state.merge_scheduled == 1
    assert state.iteration == 1


@pytest.mark.asyncio
async def test_continue_node_defaults_to_reflect() -> None:
    state = _make_state()
    state.iteration = 0
    deps = _make_deps()
    ctx = GraphRunContext(state=state, deps=deps)

    node = ContinueNode()
    next_node = await node.run(ctx)

    assert isinstance(next_node, ReflectNode)
    assert state.iteration == 1
