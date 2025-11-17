"""Helpers for evaluating agents/candidates on datasets outside GEPA."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pydantic_ai import UsageLimits
from pydantic_ai.agent import AbstractAgent
from pydantic import BaseModel
from pydantic_evals import Case, Dataset

from .adapters.agent_adapter import create_adapter
from .signature import InputSpec


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
    dataset: Sequence[Case[Any, Any, Any]] | Dataset[Any, Any],
    candidate: Mapping[str, str] | None = None,
    concurrency: int = 20,
    agent_usage_limits: UsageLimits | None = None,
    capture_traces: bool = False,
    input_type: InputSpec[BaseModel] | None = None,
) -> list[EvaluationRecord]:
    """Evaluate an agent/candidate pair on a dataset in parallel."""

    semaphore = asyncio.Semaphore(max(1, concurrency))
    records: list[EvaluationRecord] = []

    cases: Sequence[Case[Any, Any, Any]]
    if isinstance(dataset, Dataset):
        cases = dataset.cases
    else:
        cases = dataset

    adapter = create_adapter(
        agent=agent,
        metric=metric,
        input_type=input_type,
        cache_manager=None,
        agent_usage_limits=agent_usage_limits,
    )

    candidate_dict = dict(candidate or {})

    async def run_case(index: int, case: Case[Any, Any, Any]) -> None:
        async with semaphore:
            result = await adapter.process_case(
                case,
                index,
                capture_traces=capture_traces,
                candidate=candidate_dict,
            )
            case_id = case.name or f"case-{index}"
            records.append(
                EvaluationRecord(
                    case_id=case_id,
                    score=float(result.get("score", 0.0)),
                    feedback=result.get("feedback"),
                    payload=result,
                )
            )

    with adapter.apply_candidate(candidate_dict):
        await asyncio.gather(*(run_case(idx, case) for idx, case in enumerate(cases)))

    return records


__all__ = [
    "EvaluationRecord",
    "evaluate_candidate_dataset",
]
