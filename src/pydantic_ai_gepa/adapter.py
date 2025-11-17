"""Adapter protocol shared by all GEPA adapter implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from pydantic_evals import Case

from .evaluation_models import EvaluationBatch

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
MetadataT = TypeVar("MetadataT")


@dataclass(slots=True)
class SharedReflectiveDataset:
    """Reflection records shared by all components."""

    records: list[dict[str, Any]]


@dataclass(slots=True)
class ComponentReflectiveDataset:
    """Reflection records keyed by component name."""

    records_by_component: Mapping[str, list[dict[str, Any]]]


ReflectiveDataset = SharedReflectiveDataset | ComponentReflectiveDataset



class Adapter(Protocol, Generic[InputT, OutputT, MetadataT]):
    """Protocol describing the minimal surface required by the GEPA engine."""

    async def evaluate(
        self,
        batch: Sequence[Case[InputT, OutputT, MetadataT]],
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
    ) -> ReflectiveDataset:
        ...

    def get_components(self) -> dict[str, str]:
        """Return the adapter's current candidate component mapping."""
        ...


__all__ = [
    "Adapter",
    "ReflectiveDataset",
    "SharedReflectiveDataset",
    "ComponentReflectiveDataset",
]
