"""Shared stubs and helpers for GEPA graph tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from pydantic_ai.messages import UserPromptPart

from pydantic_ai_gepa.adapter import AdapterTrajectory, AgentAdapter
from pydantic_ai_gepa.types import DataInst, DataInstWithPrompt, RolloutOutput

__all__ = [
    "AdapterStub",
    "EvaluationBatchStub",
    "ProposalGeneratorStub",
    "make_adapter_stub",
    "make_dataset",
]


def make_dataset(size: int = 3) -> list[DataInstWithPrompt]:
    """Return a simple dataset for GEPA tests."""
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
class EvaluationBatchStub:
    """Lightweight EvaluationBatch replacement used by adapter stubs."""

    outputs: list[RolloutOutput[str]]
    scores: list[float]
    trajectories: list[AdapterTrajectory] | None


class AdapterStub:
    """Adapter that produces deterministic scores based on component text."""

    def __init__(self) -> None:
        self.agent = type("Agent", (), {"_instructions": "seed instructions"})()
        self.input_spec = None
        self.reflection_model = "reflection-model"
        self.reflection_sampler = None

    async def evaluate(self, batch, candidate, capture_traces):
        text = candidate["instructions"]
        base = 0.85 if text.startswith("improved") else 0.4

        outputs = [
            RolloutOutput.from_success(f"{text}-{instance.case_id}")
            for instance in batch
        ]
        trajectories = (
            [
                AdapterTrajectory(
                    messages=[],
                    final_output=output.result,
                    instructions=text,
                )
                for output in outputs
            ]
            if capture_traces
            else None
        )

        return EvaluationBatchStub(
            outputs=outputs,
            scores=[base for _ in batch],
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self, *, candidate, eval_batch, components_to_update
    ):
        return {
            component: [
                {
                    "feedback": "stub feedback",
                    "score": score,
                    "success": output.success,
                }
                for score, output in zip(eval_batch.scores, eval_batch.outputs)
            ]
            for component in components_to_update
        }


class ProposalGeneratorStub:
    """Proposal generator that improves instructions on the first call only."""

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


def make_adapter_stub() -> AgentAdapter[DataInst]:
    """Return the adapter stub typed as a PydanticAIGEPAAdapter."""
    return cast(AgentAdapter[DataInst], AdapterStub())
