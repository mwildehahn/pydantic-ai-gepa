"""Component selection strategies."""

from __future__ import annotations

from typing import Protocol

from ..models import GepaState


class ComponentSelector(Protocol):
    """Protocol for selecting which components to update."""

    def select(self, state: GepaState, candidate_idx: int) -> list[str]:
        """Return a list of component names to update."""
        ...


class RoundRobinComponentSelector:
    """Cycle through each component of a candidate in order."""

    def __init__(self) -> None:
        self._pointers: dict[int, int] = {}

    def select(self, state: GepaState, candidate_idx: int) -> list[str]:
        if candidate_idx >= len(state.candidates) or candidate_idx < 0:
            raise IndexError(f"Candidate index {candidate_idx} out of range.")

        candidate = state.candidates[candidate_idx]
        component_names = list(candidate.components.keys())
        if not component_names:
            raise ValueError("Candidate has no components to select.")

        pointer = self._pointers.get(candidate_idx, 0) % len(component_names)
        selection = component_names[pointer]
        self._pointers[candidate_idx] = (pointer + 1) % len(component_names)
        return [selection]


class AllComponentSelector:
    """Return every component for simultaneous updates."""

    def select(self, state: GepaState, candidate_idx: int) -> list[str]:
        if candidate_idx >= len(state.candidates) or candidate_idx < 0:
            raise IndexError(f"Candidate index {candidate_idx} out of range.")
        component_names = list(state.candidates[candidate_idx].components.keys())
        if not component_names:
            raise ValueError("Candidate has no components to select.")
        return component_names
