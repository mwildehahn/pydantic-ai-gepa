"""Type definitions for GEPA adapter integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import warnings
from typing import Any, Generic, Protocol, Sequence, TypeVar

from pydantic import BaseModel
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    VideoUrl,
)

# Type variable for the input type
InputModelT = TypeVar("InputModelT", bound=BaseModel)

# Type variable for the output type in RolloutOutput
OutputT = TypeVar("OutputT")


@dataclass
class DataInstWithPrompt:
    """A single data instance for optimization.

    Each instance represents a single case from a pydantic-evals Dataset.
    """

    user_prompt: UserPromptPart
    message_history: list[ModelMessage] | None
    metadata: dict[str, Any]
    case_id: str  # Unique identifier for tracking


@dataclass(init=False)
class DataInstWithInput(Generic[InputModelT]):
    """A single data instance for optimization with a structured input model."""

    input: InputModelT
    message_history: list[ModelMessage] | None
    metadata: dict[str, Any]
    case_id: str  # Unique identifier for tracking

    def __init__(
        self,
        *,
        input: InputModelT | None = None,
        signature: InputModelT | None = None,
        message_history: list[ModelMessage] | None,
        metadata: dict[str, Any],
        case_id: str,
    ) -> None:
        if input is None and signature is None:
            raise TypeError("Either 'input' or legacy 'signature' must be provided.")
        if input is not None and signature is not None and input != signature:
            raise ValueError("Received both 'input' and legacy 'signature' with different values.")

        resolved = input if input is not None else signature
        assert resolved is not None

        if signature is not None and input is None:
            warnings.warn(
                "Passing 'signature=' to DataInstWithInput is deprecated; use 'input=' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.input = resolved
        self.message_history = message_history
        self.metadata = metadata
        self.case_id = case_id

    @property
    def signature(self) -> InputModelT:
        """Legacy accessor kept for backward compatibility."""
        warnings.warn(
            "DataInstWithInput.signature is deprecated; use .input instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.input

    @signature.setter
    def signature(self, value: InputModelT) -> None:
        warnings.warn(
            "Setting DataInstWithInput.signature is deprecated; assign to .input instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.input = value


DataInst = DataInstWithPrompt | DataInstWithInput[Any]


class Trajectory(Protocol):
    """Minimal interface for execution trajectories consumed by GEPA."""

    instructions: str | None
    metric_feedback: str | None
    final_output: Any | None

    def to_reflective_record(self) -> dict[str, Any]:
        ...
DataInstT = TypeVar("DataInstT", bound=DataInst)



@dataclass
class RolloutOutput(Generic[OutputT]):
    """Output from a single agent execution.

    Generic type parameter OutputT specifies the expected type of the result
    when the execution is successful.
    """

    result: OutputT | None
    success: bool
    error_message: str | None = None

    @classmethod
    def from_success(cls, result: OutputT) -> RolloutOutput[OutputT]:
        """Create from successful execution."""
        return cls(result=result, success=True)

    @classmethod
    def from_error(cls, error: Exception) -> RolloutOutput[OutputT]:
        """Create from failed execution."""
        return cls(result=None, success=False, error_message=str(error))
