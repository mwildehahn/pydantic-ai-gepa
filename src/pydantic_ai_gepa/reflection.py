"""Reflection-related protocols and helpers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ReflectionSampler(Protocol):
    """Protocol for sampling reflection records before LLM calls."""

    def __call__(
        self,
        records: list[dict[str, Any]],
        max_records: int,
    ) -> list[dict[str, Any]]:
        """Return a subset of reflection records.

        Args:
            records: All available reflection records.
            max_records: Upper bound for records to return.

        Returns:
            Subset of ``records`` (up to ``max_records`` items).
        """
        ...


__all__ = ["ReflectionSampler"]
