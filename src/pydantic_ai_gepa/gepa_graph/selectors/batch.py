"""Deterministic minibatch sampling."""

from __future__ import annotations

import random
from typing import Any

from ..datasets import ComparableHashable, DataLoader
from ..models import GepaState
from pydantic_evals import Case


class BatchSampler:
    """Epoch-based deterministic batch sampler."""

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._shuffled_ids: list[ComparableHashable] = []
        self._dataset_ids: list[ComparableHashable] = []
        self._position: int = 0
        self._dataset_size: int = 0
        self._usage_counts: dict[ComparableHashable, int] = {}

    async def sample(
        self,
        training_set: DataLoader[ComparableHashable, Case[Any, Any, Any]],
        state: GepaState,
        size: int,
    ) -> list[Case[Any, Any, Any]]:
        """Return a deterministic minibatch of training instances."""
        _ = state  # Reserved for future adaptive policies
        if size <= 0:
            raise ValueError("Batch size must be positive.")

        dataset_size = len(training_set)
        if dataset_size == 0:
            raise ValueError("Training set cannot be empty.")

        ids = list(await training_set.all_ids())
        if dataset_size != self._dataset_size or ids != self._dataset_ids:
            self._dataset_size = dataset_size
            self._dataset_ids = ids
            self._usage_counts = {data_id: self._usage_counts.get(data_id, 0) for data_id in ids}
            self._shuffled_ids = []
            self._position = 0

        if not self._shuffled_ids or self._position + size > len(self._shuffled_ids):
            self._reshuffle(batch_size=size)

        batch_ids = self._shuffled_ids[self._position : self._position + size]
        self._position += size

        for data_id in batch_ids:
            self._usage_counts[data_id] = self._usage_counts.get(data_id, 0) + 1

        return await training_set.fetch(batch_ids)

    def _reshuffle(self, *, batch_size: int) -> None:
        if not self._dataset_ids:
            raise ValueError("Cannot reshuffle empty dataset.")

        ids = list(self._dataset_ids)
        self._rng.shuffle(ids)

        remainder = len(ids) % batch_size
        if remainder:
            needed = batch_size - remainder
            least_used = sorted(
                self._dataset_ids,
                key=lambda data_id: (self._usage_counts.get(data_id, 0), data_id),
            )
            padding = [least_used[i % len(least_used)] for i in range(needed)]
            ids.extend(padding)

        self._shuffled_ids = ids
        self._position = 0
