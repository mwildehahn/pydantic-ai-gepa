"""Tests for structured logging helpers in ``pydantic_ai_gepa.runner``."""

from __future__ import annotations

from pydantic_ai_gepa.logging_utils import log_structured


class _StructuredRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def info(self, message: str, /, **kwargs: object) -> None:  # pragma: no cover - exercised via helper
        self.calls.append(("info", message, dict(kwargs)))

    def debug(self, message: str, /, **kwargs: object) -> None:  # pragma: no cover - parity
        self.calls.append(("debug", message, dict(kwargs)))

    def warning(self, message: str, /, **kwargs: object) -> None:  # pragma: no cover - parity
        self.calls.append(("warning", message, dict(kwargs)))

    def error(self, message: str, /, **kwargs: object) -> None:  # pragma: no cover - exercised via helper
        self.calls.append(("error", message, dict(kwargs)))


class _PositionalOnlyLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, /, *args: object, **kwargs: object) -> None:  # pragma: no cover - fallback path
        if kwargs:
            raise TypeError("positional only")
        self.messages.append(message)

    def debug(self, message: str, /, *args: object, **kwargs: object) -> None:  # pragma: no cover - unused
        self.messages.append(message)

    def warning(self, message: str, /, *args: object, **kwargs: object) -> None:  # pragma: no cover - unused
        self.messages.append(message)

    def error(self, message: str, /, *args: object, **kwargs: object) -> None:  # pragma: no cover - unused
        self.messages.append(message)


def test_log_structured_with_kwargs() -> None:
    recorder = _StructuredRecorder()
    log_structured(recorder, "info", "Cache stats", hits=5, misses=1)
    assert recorder.calls == [
        ("info", "Cache stats", {"hits": 5, "misses": 1})
    ]


def test_log_structured_falls_back_to_message() -> None:
    logger = _PositionalOnlyLogger()
    log_structured(logger, "info", "Cache stats", hits=3)
    assert logger.messages == ["Cache stats | hits=3"]
