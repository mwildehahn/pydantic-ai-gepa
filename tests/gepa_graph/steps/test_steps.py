"""Tests for the start, evaluate, and continue steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import pytest
from pydantic_graph.beta import StepContext
from pydantic_evals import Case

from pydantic_ai_gepa.adapter import Adapter, SharedReflectiveDataset
from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapterTrajectory
from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph.deps import GepaDeps
from pydantic_ai_gepa.gepa_graph.evaluation import ParallelEvaluator, ParetoFrontManager
from pydantic_ai_gepa.gepa_graph.models import (
    CandidateMap,
    CandidateProgram,
    ComponentValue,
    GepaConfig,
    GepaState,
)
from pydantic_ai_gepa.gepa_graph.steps import (
    StopSignal,
    continue_step,
    evaluate_step,
    merge_step,
    reflect_step,
    start_step,
)
from pydantic_ai_gepa.types import RolloutOutput
from pydantic_ai_gepa.gepa_graph.selectors import (
    BatchSampler,
    CurrentBestCandidateSelector,
    RoundRobinComponentSelector,
)
from pydantic_ai_gepa.gepa_graph.proposal import (
    InstructionProposalGenerator,
    MergeProposalBuilder,
)


def _make_data_inst(case_id: str) -> Case[str, str, dict[str, str]]:
    return Case(name=case_id, inputs=f"prompt-{case_id}", metadata={})


def _make_state(
    num_instances: int = 2,
    *,
    config: GepaConfig | None = None,
) -> GepaState:
    training = [_make_data_inst(str(i)) for i in range(num_instances)]
    config = config or GepaConfig(max_evaluations=10)
    loader = ListDataLoader(training)
    return GepaState(
        config=config,
        training_set=loader,
        validation_set=ListDataLoader(training),
    )


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
        case_name = batch[0].name or "unnamed"
        score = self.scores.get(case_name, 0.5)
        return _FakeEvaluationBatch(
            outputs=[RolloutOutput.from_success(candidate["instructions"].text)],
            scores=[score],
        )

    def make_reflective_dataset(
        self,
        *,
        candidate,
        eval_batch,
        components_to_update: Sequence[str],
    ) -> SharedReflectiveDataset:
        return SharedReflectiveDataset(records=[])

    def get_components(self) -> CandidateMap:
        return {"instructions": ComponentValue(name="instructions", text="seed")}


class _HydratingAdapter(_FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._include_tool = False

    async def evaluate(self, batch, candidate, capture_traces):
        result = await super().evaluate(batch, candidate, capture_traces)
        self._include_tool = True
        return result

    def get_components(self) -> CandidateMap:
        components = super().get_components()
        if self._include_tool:
            hydrated = {name: value.model_copy() for name, value in components.items()}
            hydrated["tool:new"] = ComponentValue(name="tool:new", text="desc")
            return hydrated
        return components


def _make_deps(
    seed_candidate: CandidateMap | None = None,
) -> GepaDeps:
    return GepaDeps(
        adapter=cast(Adapter[str, str, dict[str, str]], _FakeAdapter()),
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


def _ctx(state: GepaState, deps: GepaDeps) -> StepContext[GepaState, GepaDeps, None]:
    return StepContext(state=state, deps=deps, inputs=None)


@pytest.mark.asyncio
async def test_start_step_adds_seed_candidate_from_deps() -> None:
    state = _make_state()
    deps = _make_deps(
        seed_candidate={
            "instructions": ComponentValue(name="instructions", text="hello world"),
        }
    )
    ctx = _ctx(state, deps)

    await start_step(ctx)

    assert len(state.candidates) == 1
    candidate = state.candidates[0]
    assert candidate.components["instructions"].text == "hello world"
    assert state.iteration == 0


@pytest.mark.asyncio
async def test_start_step_is_idempotent_when_candidates_exist() -> None:
    state = _make_state()
    deps = _make_deps(
        seed_candidate={
            "instructions": ComponentValue(name="instructions", text="hello"),
        }
    )
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
    ctx = _ctx(state, deps)

    await start_step(ctx)

    assert len(state.candidates) == 1


@pytest.mark.asyncio
async def test_start_step_uses_adapter_snapshot_when_seed_missing() -> None:
    state = _make_state()
    deps = _make_deps(seed_candidate=None)
    ctx = _ctx(state, deps)

    await start_step(ctx)

    assert len(state.candidates) == 1
    candidate = state.candidates[0]
    assert candidate.components["instructions"].text == "seed"
    assert deps.seed_candidate == {
        "instructions": ComponentValue(name="instructions", text="seed"),
    }


@pytest.mark.asyncio
async def test_evaluate_step_updates_candidate_scores_and_state() -> None:
    state = _make_state(num_instances=3)
    candidate = CandidateProgram(
        idx=0,
        components={"instructions": ComponentValue(name="instructions", text="test")},
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )
    state.add_candidate(candidate)
    ctx = _ctx(state, _make_deps())

    await evaluate_step(ctx)
    validation_set = state.validation_set
    assert validation_set is not None
    assert len(candidate.validation_scores) == len(validation_set)
    assert state.best_candidate_idx == 0
    assert state.total_evaluations == len(validation_set)
    assert state.full_validations == 1
    assert state.pareto_front


@pytest.mark.asyncio
async def test_evaluate_step_hydrates_new_components() -> None:
    state = _make_state(num_instances=1)
    candidate = CandidateProgram(
        idx=0,
        components={"instructions": ComponentValue(name="instructions", text="test")},
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )
    state.add_candidate(candidate)
    adapter = cast(Adapter[str, str, dict[str, str]], _HydratingAdapter())
    deps = _make_deps()
    deps.adapter = adapter
    ctx = _ctx(state, deps)

    await evaluate_step(ctx)

    assert "tool:new" in candidate.components
    assert candidate.components["tool:new"].text == "desc"
    assert deps.seed_candidate is not None
    assert deps.seed_candidate["tool:new"].text == "desc"


@pytest.mark.asyncio
async def test_continue_step_returns_end_when_budget_spent() -> None:
    state = _make_state()
    state.total_evaluations = state.config.max_evaluations
    deps = _make_deps()
    ctx = _ctx(state, deps)

    result = await continue_step(ctx)

    assert isinstance(result, StopSignal)
    assert state.stop_reason == "Max evaluations reached"


@pytest.mark.asyncio
async def test_continue_step_triggers_merge_when_scheduled() -> None:
    config = GepaConfig(max_evaluations=10, use_merge=True)
    state = _make_state(config=config)
    state.iteration = 0
    state.merge_scheduled = 2
    state.last_accepted = True
    deps = _make_deps()
    ctx = _ctx(state, deps)

    next_node = await continue_step(ctx)

    assert next_node == "merge"
    assert state.merge_attempts == 0
    assert state.merge_scheduled == 2
    assert state.iteration == 1


@pytest.mark.asyncio
async def test_continue_step_defaults_to_reflect() -> None:
    state = _make_state()
    state.iteration = 0
    deps = _make_deps()
    ctx = _ctx(state, deps)

    next_node = await continue_step(ctx)

    assert next_node == "reflect"
    assert state.iteration == 1
