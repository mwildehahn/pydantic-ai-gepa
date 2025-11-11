"""Shared logging helpers that default to structured logfire logging."""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol

module_logger = logging.getLogger("pydantic_ai_gepa")

try:  # pragma: no cover - logfire is a runtime dependency but keep defensive fallback
    import logfire as _logfire_module
except Exception:  # pragma: no cover - e.g., logfire missing in minimal environment
    _logfire_module = None


class StructuredLogger(Protocol):
    """Protocol for loggers supporting logfire-style structured methods."""

    def info(self, message: str, /, *args: Any, **kwargs: Any) -> Any:
        ...

    def debug(self, message: str, /, *args: Any, **kwargs: Any) -> Any:
        ...

    def warning(self, message: str, /, *args: Any, **kwargs: Any) -> Any:
        ...

    def error(self, message: str, /, *args: Any, **kwargs: Any) -> Any:
        ...


def get_structured_logger(preferred: StructuredLogger | None = None) -> StructuredLogger:
    """Return the preferred logger or fall back to logfire/stdlib logging."""

    if preferred is not None:
        return preferred
    if _logfire_module is not None:
        return _logfire_module  # type: ignore[return-value]
    return module_logger


def log_structured(
    logger: StructuredLogger,
    level: Literal["debug", "info", "warning", "error"],
    message: str,
    **data: Any,
) -> None:
    """Invoke ``logger.<level>`` passing structured kwargs when supported."""

    method = getattr(logger, level, None)
    if not callable(method):  # pragma: no cover - defensive guard
        return

    payload = data
    try:
        method(message, **payload)
    except TypeError:
        if payload:
            formatted = ", ".join(f"{key}={value!r}" for key, value in payload.items())
            message = f"{message} | {formatted}"
        method(message)


__all__ = [
    "StructuredLogger",
    "get_structured_logger",
    "log_structured",
    "module_logger",
]
