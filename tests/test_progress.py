"""Unit tests for the OptimizationProgress helper."""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai_gepa.progress import OptimizationProgress


class _FailingProgress:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("Progress should not be instantiated when disabled")


def test_optimization_progress_disabled_skips_rich(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure disabled progress bars avoid constructing Rich primitives."""
    monkeypatch.setattr("pydantic_ai_gepa.progress.Progress", _FailingProgress)
    with OptimizationProgress(total=5, description="Skip", enabled=False) as progress:
        progress.update(3)


def test_optimization_progress_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify updates clamp to the configured total and reset description."""
    updates: list[dict[str, Any]] = []

    class _StubProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            updates.clear()

        def __enter__(self) -> "_StubProgress":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def add_task(self, description: str, total: int) -> int:
            self.description = description
            self.total = total
            return 1

        def update(self, task_id: int, **kwargs: Any) -> None:
            updates.append({"task_id": task_id, **kwargs})

    monkeypatch.setattr("pydantic_ai_gepa.progress.Progress", _StubProgress)

    with OptimizationProgress(total=10, description="Test", enabled=True) as progress:
        progress.update(3)
        progress.update(12)

    assert [
        (entry["task_id"], entry["completed"], entry["description"])
        for entry in updates
    ] == [
        (1, 3, "Test"),
        (1, 10, "Test"),
    ]


def test_optimization_progress_includes_node_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure node context is embedded into the progress description."""
    descriptions: list[str] = []

    class _StubProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "_StubProgress":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def add_task(self, description: str, total: int) -> int:
            self.description = description
            self.total = total
            return 11

        def update(self, task_id: int, **kwargs: Any) -> None:
            descriptions.append(kwargs.get("description", ""))

    monkeypatch.setattr("pydantic_ai_gepa.progress.Progress", _StubProgress)

    with OptimizationProgress(
        total=5, description="Node run", enabled=True
    ) as progress:
        progress.update(
            2,
            current_node="EvaluateNode",
            previous_node="ReflectNode",
        )

    assert descriptions
    assert "prev: ReflectNode" in descriptions[-1]
    assert "curr: EvaluateNode" in descriptions[-1]


def test_optimization_progress_appends_best_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure best-score tracking is reflected in the description."""
    descriptions: list[str] = []

    class _StubProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "_StubProgress":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def add_task(self, description: str, total: int) -> int:
            return 42

        def update(self, task_id: int, **kwargs: Any) -> None:
            descriptions.append(kwargs.get("description", ""))

    monkeypatch.setattr("pydantic_ai_gepa.progress.Progress", _StubProgress)

    with OptimizationProgress(total=3, description="Score", enabled=True) as progress:
        progress.update(1, best_score=0.87654)

    assert descriptions
    assert descriptions[-1].endswith("best: 0.877")
