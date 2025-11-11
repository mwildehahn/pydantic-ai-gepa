"""Parallel evaluation utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar

from ...adapter import Adapter
from ...evaluation_models import EvaluationBatch
from ...types import DataInst, RolloutOutput, Trajectory
from ..models import CandidateProgram

DataIdT = TypeVar("DataIdT")
DataInstT = TypeVar("DataInstT", bound=DataInst, contravariant=True)


@dataclass(slots=True, kw_only=True)
class EvaluationResults(Generic[DataIdT]):
    """Container for flattened evaluation outputs."""

    data_ids: list[DataIdT]
    scores: list[float]
    outputs: list[RolloutOutput[Any]]
    trajectories: list[Trajectory] | None = None

    def __post_init__(self) -> None:
        lengths = {len(self.data_ids), len(self.scores), len(self.outputs)}
        if self.trajectories is not None:
            lengths.add(len(self.trajectories))
        if len(lengths) > 1:
            raise ValueError("EvaluationResults fields must have matching lengths.")

    def __len__(self) -> int:
        return len(self.data_ids)

    def __iter__(self):
        return iter(zip(self.data_ids, self.scores, self.outputs))

    def has_trajectories(self) -> bool:
        return self.trajectories is not None


class ParallelEvaluator:
    """Evaluate candidates on datasets with asyncio-powered parallelism."""

    async def evaluate_batch(
        self,
        *,
        candidate: CandidateProgram,
        batch: Sequence[DataInst],
        adapter: Adapter[DataInst],
        max_concurrent: int = 10,
        capture_traces: bool = False,
    ) -> EvaluationResults[str]:
        """Evaluate ``candidate`` for every instance in ``batch``."""
        if not batch:
            return EvaluationResults(
                data_ids=[],
                scores=[],
                outputs=[],
                trajectories=[] if capture_traces else None,
            )

        semaphore = asyncio.Semaphore(max(1, max_concurrent))
        candidate_payload = candidate.to_dict_str()

        async def run_one(index: int, instance: DataInst):
            async with semaphore:
                eval_batch = await self._call_adapter(
                    adapter=adapter,
                    instance=instance,
                    candidate_payload=candidate_payload,
                    capture_traces=capture_traces,
                )
            data_id = self._data_id(instance, index)
            return data_id, eval_batch

        tasks = [run_one(idx, instance) for idx, instance in enumerate(batch)]

        results = await asyncio.gather(*tasks)
        return self._merge_results(results, capture_traces=capture_traces)

    async def _call_adapter(
        self,
        *,
        adapter: Adapter[DataInst],
        instance: DataInst,
        candidate_payload: dict[str, str],
        capture_traces: bool,
    ) -> EvaluationBatch:
        return await adapter.evaluate(
            [instance],
            candidate_payload,
            capture_traces,
        )

    def _merge_results(
        self,
        results: list[tuple[str, EvaluationBatch]],
        *,
        capture_traces: bool,
    ) -> EvaluationResults[str]:
        data_ids: list[str] = []
        scores: list[float] = []
        outputs: list[RolloutOutput[Any]] = []
        trajectories: list[Trajectory] | None = [] if capture_traces else None

        for data_id, batch in results:
            batch_scores = list(batch.scores)
            batch_outputs = list(batch.outputs)

            if len(batch_scores) != len(batch_outputs):
                raise ValueError("Adapter returned mismatched scores and outputs.")

            data_ids.extend([data_id] * len(batch_scores))
            scores.extend(batch_scores)
            outputs.extend(batch_outputs)

            if trajectories is not None:
                batch_traces = batch.trajectories
                if batch_traces is None:
                    trajectories = None
                else:
                    trajectories.extend(batch_traces)

        return EvaluationResults(
            data_ids=data_ids,
            scores=scores,
            outputs=outputs,
            trajectories=trajectories,
        )

    @staticmethod
    def _data_id(instance: DataInst, index: int) -> str:
        case_id = getattr(instance, "case_id", None)
        return str(case_id) if case_id is not None else str(index)


__all__ = ["EvaluationResults", "ParallelEvaluator"]
