"""Candidate selection strategies."""

from __future__ import annotations

from collections import Counter
from typing import Protocol
import random

from ..models import GepaState


class CandidateSelector(Protocol):
    """Protocol for deciding which candidate to improve next."""

    def select(self, state: GepaState) -> int:
        """Return the index of the candidate to improve."""
        ...


class ParetoCandidateSelector:
    """Sample candidates proportionally to their Pareto-front frequency."""

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select(self, state: GepaState) -> int:
        """Select a candidate weighted by Pareto-front frequency."""
        counts = Counter()
        for entry in state.pareto_front.values():
            for idx in entry.candidate_indices:
                if 0 <= idx < len(state.candidates):
                    counts[idx] += 1

        if counts:
            candidates = list(counts.keys())
            weights = list(counts.values())
            choice = self._rng.choices(candidates, weights=weights, k=1)[0]
            return choice

        if state.best_candidate_idx is not None:
            return state.best_candidate_idx

        if state.candidates:
            return 0

        raise ValueError("Cannot select candidate; state has no candidates.")


class CurrentBestCandidateSelector:
    """Always return the current best candidate index (fallback to seed)."""

    def select(self, state: GepaState) -> int:
        if state.best_candidate_idx is not None:
            return state.best_candidate_idx
        if state.candidates:
            return 0
        raise ValueError("Cannot select candidate; state has no candidates.")
