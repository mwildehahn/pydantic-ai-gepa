from __future__ import annotations

from typing import Any
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from pydantic_ai_gepa.gepa_graph.evaluation import EvaluationResults
from pydantic_ai_gepa.gepa_graph.proposal import ReflectiveDatasetBuilder
from pydantic_ai_gepa.types import RolloutOutput, Trajectory


def _make_trajectory(
    *,
    prompt: str,
    response: str,
    feedback: str | None,
) -> Trajectory:
    return Trajectory(
        messages=[
            ModelRequest(parts=[UserPromptPart(content=prompt)], instructions="Base instructions"),
            ModelResponse(parts=[TextPart(content=response)]),
        ],
        final_output=response,
        instructions="Base instructions",
        metric_feedback=feedback,
        usage={"requests": 1},
    )


def _make_results() -> EvaluationResults[str]:
    trajectories = [
        _make_trajectory(prompt="Hello", response="Hi!", feedback="Detailed feedback"),
        _make_trajectory(prompt="Bad", response="Err", feedback=None),
    ]
    outputs = [
        RolloutOutput.from_success("Hi!"),
        RolloutOutput.from_error(ValueError("boom")),
    ]
    scores = [0.9, 0.2]
    return EvaluationResults(
        data_ids=["a", "b"],
        scores=scores,
        outputs=outputs,
        trajectories=trajectories,
    )


def test_builder_creates_records_with_feedback() -> None:
    builder = ReflectiveDatasetBuilder()
    dataset = builder.build_dataset(
        eval_results=_make_results(),
        components=["instructions", "tools"],
    )

    # All components share the same reflective data
    assert dataset["instructions"] is dataset["tools"]

    records = dataset["instructions"]
    assert len(records) == 2
    assert records[0]["feedback"] == "Detailed feedback"
    assert records[1]["feedback"].startswith("Poor response (score: 0.20)")
    assert records[1]["error_message"] == "boom"
    assert records[0]["instructions"] == "Base instructions"


def test_builder_applies_sampler() -> None:
    calls: list[tuple[int, int]] = []

    def sampler(records: list[dict[str, Any]], max_records: int) -> list[dict[str, Any]]:
        calls.append((len(records), max_records))
        return records[:1]

    builder = ReflectiveDatasetBuilder(sampler=sampler, max_records=5)
    dataset = builder.build_dataset(eval_results=_make_results(), components=["instructions"])
    assert len(dataset["instructions"]) == 1
    assert calls == [(2, 5)]


def test_builder_handles_missing_trajectories() -> None:
    results = EvaluationResults(
        data_ids=["case-1"],
        scores=[0.5],
        outputs=[RolloutOutput.from_success("ok")],
        trajectories=None,
    )
    builder = ReflectiveDatasetBuilder()
    dataset = builder.build_dataset(eval_results=results, components=["instructions"])
    assert dataset == {"instructions": []}
