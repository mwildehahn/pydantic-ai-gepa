"""Custom exceptions used across pydantic-ai-gepa."""

from __future__ import annotations

class UsageBudgetExceeded(RuntimeError):
    """Raised when the GEPA optimization run exceeds its configured usage budget."""


__all__ = ["UsageBudgetExceeded"]

