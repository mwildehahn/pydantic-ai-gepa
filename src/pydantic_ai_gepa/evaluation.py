"""Helpers for evaluating agents/candidates on datasets outside GEPA."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pydantic_ai import UsageLimits
from pydantic_ai.agent import AbstractAgent

from .components import apply_candidate_to_agent
from .adapters.agent_adapter import AgentAdapter
from .types import DataInst


@dataclass(slots=True)
class EvaluationRecord:
    """Structured result for a single dataset evaluation."""

    case_id: str
    score: float
    feedback: str | None
    payload: Mapping[str, Any]


async def evaluate_candidate_dataset(
    *,
    agent: AbstractAgent[Any, Any],
    metric,
    input_type,
    dataset: Sequence[DataInst],
    candidate: Mapping[str, str] | None = None,
    concurrency: int = 20,
    agent_usage_limits: UsageLimits | None = None,
    capture_traces: bool = False,
) -> list[EvaluationRecord]:
    """Evaluate an agent/candidate pair on a dataset in parallel."""

    semaphore = asyncio.Semaphore(max(1, concurrency))
    records: list[EvaluationRecord] = []

    async def run_case(index: int, data_inst: DataInst, adapter: AgentAdapter) -> None:
        async with semaphore:
            result = await adapter.process_data_instance(
                data_inst,
                capture_traces=capture_traces,
            )
            case_id = getattr(data_inst, "case_id", None) or f"case-{index}"
            records.append(
                EvaluationRecord(
                    case_id=case_id,
                    score=float(result.get("score", 0.0)),
                    feedback=result.get("feedback"),
                    payload=result,
                )
            )

    with apply_candidate_to_agent(agent, candidate):
        adapter = AgentAdapter(
            agent=agent,
            metric=metric,
            input_type=input_type,
            cache_manager=None,
            agent_usage_limits=agent_usage_limits,
        )
        await asyncio.gather(*(
            run_case(idx, data_inst, adapter)
            for idx, data_inst in enumerate(dataset)
        ))

    return records


__all__ = [
    "EvaluationRecord",
    "evaluate_candidate_dataset",
]
