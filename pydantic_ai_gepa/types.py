"""Type definitions for GEPA adapter integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart


@dataclass
class DataInst:
    """A single data instance for optimization.

    Each instance represents a single case from a pydantic-evals Dataset.
    """

    user_prompt: UserPromptPart
    message_history: list[ModelMessage] | None
    metadata: dict[str, Any]
    case_id: str  # Unique identifier for tracking


@dataclass
class Trajectory:
    """Execution trajectory capturing the agent run.

    This is kept minimal for v1 - just enough for reflection.
    """

    messages: list[ModelMessage]
    final_output: Any
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    data_inst: DataInst | None = None
    metric_feedback: str | None = None

    def _extract_user_content(self, part: UserPromptPart) -> str:
        """Extract text content from a UserPromptPart."""
        if isinstance(part.content, str):
            return part.content
        elif part.content:
            # For multi-modal content, just take the first text content
            for content_item in part.content:
                if isinstance(content_item, str):
                    return content_item
            return 'Multi-modal content'
        else:
            return 'No content'

    def _extract_user_message(self) -> str | None:
        """Extract the first user message from the trajectory."""
        for msg in self.messages:
            if isinstance(msg, ModelRequest):
                # Look for UserPromptPart in request parts
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        return self._extract_user_content(part)
        return None

    def _extract_assistant_message(self) -> str | None:
        """Extract the last assistant message from the trajectory."""
        for msg in reversed(self.messages):
            if isinstance(msg, ModelResponse):
                # Look for TextPart in response parts
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        return part.content
        return None

    def to_reflective_record(self) -> dict[str, Any]:
        """Convert to a compact record for reflection."""
        user_msg = self._extract_user_message()
        assistant_msg = self._extract_assistant_message()

        return {
            'user_prompt': user_msg or 'No user message',
            'assistant_response': assistant_msg or str(self.final_output),
            'error': self.error,
        }


@dataclass
class RolloutOutput:
    """Output from a single agent execution."""

    result: Any
    success: bool
    error_message: str | None = None

    @classmethod
    def from_success(cls, result: Any) -> RolloutOutput:
        """Create from successful execution."""
        return cls(result=result, success=True)

    @classmethod
    def from_error(cls, error: Exception) -> RolloutOutput:
        """Create from failed execution."""
        return cls(result=None, success=False, error_message=str(error))
