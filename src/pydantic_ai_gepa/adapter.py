"""Adapter protocol shared by all GEPA adapter implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .evaluation_models import EvaluationBatch
from .types import DataInstT


class Adapter(Protocol[DataInstT]):
    """Protocol describing the minimal surface required by the GEPA engine."""

    async def evaluate(
        self,
        batch: Sequence[DataInstT],
        candidate: dict[str, str],
        capture_traces: bool,
    ) -> EvaluationBatch:
        ...

    def make_reflective_dataset(
        self,
        *,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: Sequence[str],
    ) -> dict[str, list[dict[str, Any]]]:
        ...

    def get_components(self) -> dict[str, str]:
        """Return the adapter's current candidate component mapping."""
        ...


__all__ = ["Adapter"]
