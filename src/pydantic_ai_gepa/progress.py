"""Utilities for displaying optimization progress in the terminal."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

DEFAULT_PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=None),
    TaskProgressColumn(),
    TextColumn("{task.completed}/{task.total} evals"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)


@dataclass
class OptimizationProgress:
    """Context manager that renders a Rich progress bar for optimization runs."""

    total: int
    description: str
    enabled: bool = False
    transient: bool = True

    _progress: Progress | None = field(init=False, default=None, repr=False)
    _task_id: TaskID | None = field(init=False, default=None, repr=False)
    _context: AbstractContextManager[Any] | None = field(
        init=False, default=None, repr=False
    )

    def __enter__(self) -> "OptimizationProgress":
        if not self.enabled:
            return self

        self._progress = Progress(
            *DEFAULT_PROGRESS_COLUMNS,
            transient=self.transient,
        )
        self._context = self._progress
        self._context.__enter__()
        self._task_id = self._progress.add_task(
            self.description,
            total=self.total,
        )
        return self

    def update(
        self,
        completed: int,
        *,
        current_node: str | None = None,
        previous_node: str | None = None,
        best_score: float | None = None,
    ) -> None:
        """Update the progress bar with the latest evaluation count and context."""
        if self._progress is None or self._task_id is None:
            return
        capped = completed if completed <= self.total else self.total
        description = self._format_description(
            current_node=current_node,
            previous_node=previous_node,
            best_score=best_score,
        )
        self._progress.update(
            self._task_id,
            completed=capped,
            description=description,
        )

    def _format_description(
        self,
        *,
        current_node: str | None,
        previous_node: str | None,
        best_score: float | None,
    ) -> str:
        if not current_node and not previous_node:
            return self._append_best_score(self.description, best_score)

        fragments: list[str] = []
        if previous_node:
            fragments.append(f"prev: {previous_node}")
        if current_node:
            fragments.append(f"curr: {current_node}")

        suffix = " | ".join(fragments)
        description = f"{self.description} ({suffix})"
        return self._append_best_score(description, best_score)

    def _append_best_score(self, description: str, best_score: float | None) -> str:
        if best_score is None:
            return description
        formatted = f"{best_score:.3f}" if best_score == best_score else "n/a"
        return f"{description} | best: {formatted}"

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._context is None:
            return False
        return bool(self._context.__exit__(exc_type, exc, tb))


__all__ = ["OptimizationProgress"]
