"""Shared stubs and helpers for GEPA graph tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from pydantic_ai.messages import UserPromptPart

from pydantic_ai_gepa.adapter import Adapter, SharedReflectiveDataset
from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapterTrajectory
from pydantic_ai_gepa.gepa_graph.proposal import ProposalResult
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
    trajectories: list[AgentAdapterTrajectory] | None


class AdapterStub:
    """Adapter that produces deterministic scores based on component text."""

    def __init__(self) -> None:
        self.agent = type("Agent", (), {"_instructions": "seed instructions"})()
        self.input_spec = None

    async def evaluate(self, batch, candidate, capture_traces):
        text = candidate["instructions"]
        base = 0.85 if text.startswith("improved") else 0.4

        outputs = [
            RolloutOutput.from_success(f"{text}-{instance.case_id}")
            for instance in batch
        ]
        trajectories = (
            [
                AgentAdapterTrajectory(
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
    ) -> SharedReflectiveDataset:
        records = [
            {
                "feedback": "stub feedback",
                "score": score,
                "success": output.success,
            }
            for score, output in zip(eval_batch.scores, eval_batch.outputs)
        ]
        return SharedReflectiveDataset(records=records)

    def get_components(self) -> dict[str, str]:
        return {"instructions": "seed instructions"}


class ProposalGeneratorStub:
    """Proposal generator that improves instructions on the first call only."""

    def __init__(self) -> None:
        self.calls = 0

    async def propose_texts(
        self,
        *,
        candidate,
        reflective_data,
        components,
        model,
        iteration: int | None = None,
        current_best_score: float | None = None,
        parent_score: float | None = None,
    ):
        self.calls += 1
        updates: dict[str, str] = {}
        for component in components:
            if self.calls == 1:
                updates[component] = f"improved {component}"
            else:
                updates[component] = candidate.components[component].text
        return ProposalResult(texts=updates, component_metadata={}, reasoning=None)


def make_adapter_stub() -> Adapter[DataInst]:
    """Return the adapter stub typed as a PydanticAIGEPAAdapter."""
    return cast(Adapter[DataInst], AdapterStub())
