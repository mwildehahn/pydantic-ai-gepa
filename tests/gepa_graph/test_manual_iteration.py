"""End-to-end manual iteration tests for the GEPA graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
from pydantic_ai.messages import UserPromptPart

from pydantic_ai_gepa.adapter import PydanticAIGEPAAdapter
from pydantic_ai_gepa.gepa_graph import create_deps, create_gepa_graph
from pydantic_ai_gepa.gepa_graph.models import GepaConfig, GepaState
from pydantic_ai_gepa.gepa_graph.nodes import StartNode
from pydantic_ai_gepa.types import DataInstWithPrompt, RolloutOutput, Trajectory


def _make_dataset(size: int = 3) -> list[DataInstWithPrompt]:
    return [
        DataInstWithPrompt(
            user_prompt=UserPromptPart(content=f"prompt-{idx}"),
            message_history=None,
            metadata={},
            case_id=str(idx),
        )
        for idx in range(size)
    ]


@dataclass
class _EvaluationBatch:
    outputs: list[RolloutOutput[str]]
    scores: list[float]
    trajectories: list[Trajectory] | None


class _AdapterStub:
    def __init__(self) -> None:
        self.agent = type("Agent", (), {"_instructions": "seed instructions"})()
        self.input_spec = None
        self.reflection_model = "reflection-model"
        self.reflection_sampler = None

    async def evaluate(self, batch, candidate, capture_traces):
        """Return deterministic scores based on the current candidate."""
        text = candidate["instructions"]
        base = 0.85 if text.startswith("improved") else 0.4

        outputs = [RolloutOutput.from_success(f"{text}-{instance.case_id}") for instance in batch]
        trajectories = (
            [
                Trajectory(
                    messages=[],
                    final_output=output.result,
                    instructions=text,
                )
                for output in outputs
            ]
            if capture_traces
            else None
        )

        return _EvaluationBatch(
            outputs=outputs,
            scores=[base for _ in batch],
            trajectories=trajectories,
        )


class _StubProposalGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def propose_texts(self, *, candidate, reflective_data, components, model):
        self.calls += 1
        updates: dict[str, str] = {}
        for component in components:
            if self.calls == 1:
                updates[component] = f"improved {component}"
            else:
                updates[component] = candidate.components[component].text
        return updates


@pytest.mark.asyncio
async def test_manual_iteration_flow() -> None:
    adapter = cast(PydanticAIGEPAAdapter[Any], _AdapterStub())
    config = GepaConfig(max_evaluations=30, minibatch_size=2, seed=17)
    deps = create_deps(adapter, config)
    deps.proposal_generator = cast(Any, _StubProposalGenerator())

    graph = create_gepa_graph(adapter=adapter, config=config)
    dataset = _make_dataset()
    state = GepaState(config=config, training_set=dataset, validation_set=dataset)

    executed_nodes: list[str] = []
    async with graph.iter(StartNode(), state=state, deps=deps) as run:
        async for node in run:
            executed_nodes.append(node.__class__.__name__)
            if state.best_score is not None and state.best_score >= 0.8:
                state.stopped = True
                state.stop_reason = "target score met"

    result = run.result.output

    assert "StartNode" in executed_nodes
    assert "ReflectNode" in executed_nodes
    assert state.stop_reason == "target score met"
    assert result.stopped is True
    assert result.best_score is not None
    assert result.original_score is not None
    assert result.best_score > result.original_score
    assert len(result.candidates) >= 2
