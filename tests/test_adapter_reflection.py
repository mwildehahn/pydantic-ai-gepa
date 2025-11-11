"""Tests for AgentAdapter reflective dataset creation."""

from __future__ import annotations

from typing import Any
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapter, AgentAdapterTrajectory
from pydantic_ai_gepa.evaluation_models import EvaluationBatch
from pydantic_ai_gepa.types import (
    DataInst,
    DataInstWithPrompt,
    MetricResult,
    RolloutOutput,
)


def _make_adapter() -> AgentAdapter[DataInst]:
    agent = Agent(TestModel(), instructions="Base instructions")

    def metric(
        data_inst: DataInst, output: RolloutOutput[Any]
    ) -> MetricResult:
        return MetricResult(score=0.5, feedback="feedback")

    return AgentAdapter(agent, metric)


def _make_trajectory(
    *,
    prompt: str,
    response: str,
    feedback: str | None,
) -> AgentAdapterTrajectory:
    return AgentAdapterTrajectory(
        messages=[
            ModelRequest(
                parts=[UserPromptPart(content=prompt)],
                instructions="Base instructions",
            ),
            ModelResponse(parts=[TextPart(content=response)]),
        ],
        final_output=response,
        instructions="Base instructions",
        metric_feedback=feedback,
        usage={"requests": 1},
    )


def _build_batch() -> EvaluationBatch:
    trajectories = [
        _make_trajectory(prompt="Hello", response="Hi!", feedback="Detailed feedback"),
        _make_trajectory(prompt="Bad", response="Err", feedback=None),
    ]
    outputs = [
        RolloutOutput.from_success("Hi!"),
        RolloutOutput.from_error(ValueError("boom")),
    ]
    scores = [0.9, 0.2]
    return EvaluationBatch(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories,
    )


def test_make_reflective_dataset_includes_feedback_and_errors() -> None:
    adapter = _make_adapter()
    dataset = adapter.make_reflective_dataset(
        candidate={"instructions": "Base instructions"},
        eval_batch=_build_batch(),
        components_to_update=["instructions", "tools"],
    )

    assert dataset["instructions"] is dataset["tools"]
    records = dataset["instructions"]
    assert len(records) == 2
    assert records[0]["feedback"] == "Detailed feedback"
    assert records[1]["feedback"].startswith("Poor response (score: 0.20)")
    assert records[1]["error_message"] == "boom"
    assert records[0]["instructions"] == "Base instructions"


def test_make_reflective_dataset_returns_full_records() -> None:
    adapter = _make_adapter()
    batch = _build_batch()
    dataset = adapter.make_reflective_dataset(
        candidate={"instructions": "seed"},
        eval_batch=batch,
        components_to_update=["instructions"],
    )

    assert len(dataset["instructions"]) == len(batch.outputs)


def test_make_reflective_dataset_handles_missing_trajectories() -> None:
    adapter = _make_adapter()
    batch = EvaluationBatch(outputs=[RolloutOutput.from_success("ok")], scores=[0.5])
    dataset = adapter.make_reflective_dataset(
        candidate={"instructions": "seed"},
        eval_batch=batch,
        components_to_update=["instructions"],
    )
    assert dataset == {"instructions": []}
