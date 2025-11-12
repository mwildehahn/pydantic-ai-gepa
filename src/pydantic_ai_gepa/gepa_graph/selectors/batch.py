"""Deterministic minibatch sampling."""

from __future__ import annotations

import random
from typing import Sequence

from ..models import GepaState
from ...types import DataInst


class BatchSampler:
    """Epoch-based deterministic batch sampler."""

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._shuffled_indices: list[int] = []
        self._position: int = 0
        self._dataset_size: int = 0
        self._usage_counts: dict[int, int] = {}

    def sample(
        self,
        training_set: Sequence[DataInst],
        state: GepaState,
        size: int,
    ) -> list[DataInst]:
        """Return a deterministic minibatch of training instances."""
        _ = state  # Reserved for future adaptive policies
        if size <= 0:
            raise ValueError("Batch size must be positive.")

        dataset_size = len(training_set)
        if dataset_size == 0:
            raise ValueError("Training set cannot be empty.")

        if dataset_size != self._dataset_size:
            self._dataset_size = dataset_size
            self._usage_counts = {idx: self._usage_counts.get(idx, 0) for idx in range(dataset_size)}
            self._shuffled_indices = []
            self._position = 0

        if not self._shuffled_indices or self._position + size > len(self._shuffled_indices):
            self._reshuffle(batch_size=size)

        batch_indices = self._shuffled_indices[self._position : self._position + size]
        self._position += size

        for idx in batch_indices:
            self._usage_counts[idx] = self._usage_counts.get(idx, 0) + 1

        return [training_set[idx] for idx in batch_indices]

    def _reshuffle(self, *, batch_size: int) -> None:
        if self._dataset_size == 0:
            raise ValueError("Cannot reshuffle empty dataset.")

        indices = list(range(self._dataset_size))
        self._rng.shuffle(indices)

        remainder = len(indices) % batch_size
        if remainder:
            needed = batch_size - remainder
            least_used = sorted(
                range(self._dataset_size),
                key=lambda idx: (self._usage_counts.get(idx, 0), idx),
            )
            padding = [least_used[i % len(least_used)] for i in range(needed)]
            indices.extend(padding)

        self._shuffled_indices = indices
        self._position = 0
