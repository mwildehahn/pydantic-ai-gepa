"""Builders for reflection datasets consumed by the GEPA graph."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...adapter import ReflectionSampler
from ...types import RolloutOutput, Trajectory
from ..evaluation import EvaluationResults

DEFAULT_REFLECTION_RECORD_LIMIT = 10


class ReflectiveDatasetBuilder:
    """Convert evaluation trajectories into reflection-ready datasets."""

    def __init__(
        self,
        *,
        sampler: ReflectionSampler | None = None,
        max_records: int = DEFAULT_REFLECTION_RECORD_LIMIT,
    ) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be greater than zero.")
        self._sampler = sampler
        self._max_records = max_records

    def build_dataset(
        self,
        *,
        eval_results: EvaluationResults[str],
        components: Sequence[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build a reflective dataset for every requested component."""
        if not components:
            return {}

        trajectories = eval_results.trajectories
        if not trajectories:
            return {component: [] for component in components}

        records = [
            self._build_record(trajectory, score, output)
            for trajectory, score, output in zip(
                trajectories,
                eval_results.scores,
                eval_results.outputs,
                strict=True,
            )
        ]
        processed_records = self._apply_sampler(records)
        return {component: processed_records for component in components}

    def _build_record(
        self,
        trajectory: Trajectory,
        score: float,
        output: RolloutOutput[Any],
    ) -> dict[str, Any]:
        record = trajectory.to_reflective_record()
        record["score"] = score
        record["success"] = output.success
        if output.error_message:
            record["error_message"] = output.error_message
        if trajectory.instructions:
            record.setdefault("instructions", trajectory.instructions)

        feedback = trajectory.metric_feedback or self._fallback_feedback(
            score, output.error_message
        )
        record["feedback"] = feedback
        return record

    def _apply_sampler(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []
        if self._sampler is None:
            return records
        return self._sampler(records, max_records=self._max_records)

    @staticmethod
    def _fallback_feedback(score: float, error_message: str | None) -> str:
        if score >= 0.8:
            return "Good response"
        if score >= 0.5:
            return "Adequate response, could be improved"
        base = f"Poor response (score: {score:.2f})"
        if error_message:
            base += f" - Error: {error_message}"
        return base


__all__ = ["ReflectiveDatasetBuilder", "DEFAULT_REFLECTION_RECORD_LIMIT"]
