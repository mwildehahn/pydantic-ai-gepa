"""Type definitions for GEPA adapter integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic_ai import usage as _usage
from pydantic_ai.messages import ModelMessage
from pydantic_evals import Case

# Type variable for the output type in RolloutOutput
OutputT = TypeVar("OutputT")


@runtime_checkable
class MetadataWithMessageHistory(Protocol):
    """Metadata protocol for cases that expose conversation context."""

    message_history: list[ModelMessage] | None


class Trajectory(Protocol):
    """Minimal interface for execution trajectories consumed by GEPA."""

    instructions: str | None
    metric_feedback: str | None
    final_output: Any | None

    def to_reflective_record(self) -> dict[str, Any]:
        ...


@dataclass
class MetricResult:
    """Standardized result returned by metric functions."""

    score: float
    feedback: str | None = None


@dataclass
class RolloutOutput(Generic[OutputT]):
    """Output from a single agent execution."""

    result: OutputT | None
    success: bool
    error_message: str | None = None
    error_kind: Literal["tool", "system"] | None = None
    usage: _usage.RunUsage | None = None

    @classmethod
    def from_success(
        cls, result: OutputT, *, usage: _usage.RunUsage | None = None
    ) -> "RolloutOutput[OutputT]":
        """Create from successful execution."""
        return cls(result=result, success=True, usage=usage)

    @classmethod
    def from_error(
        cls,
        error: Exception,
        *,
        kind: Literal["tool", "system"] | None = None,
        usage: _usage.RunUsage | None = None,
    ) -> "RolloutOutput[OutputT]":
        """Create from failed execution."""
        return cls(
            result=None,
            success=False,
            error_message=str(error),
            error_kind=kind,
            usage=usage,
        )


__all__ = [
    "Case",
    "MetadataWithMessageHistory",
    "Trajectory",
    "MetricResult",
    "RolloutOutput",
]
