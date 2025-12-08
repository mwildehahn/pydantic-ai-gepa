"""Merge proposal construction utilities."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from pydantic_evals import Case
from ..datasets import DataLoader, data_id_for_instance
from ..example_bank import InMemoryExampleBank
from ..models import CandidateMap, CandidateProgram, ComponentValue, GepaState

_SCORE_EPSILON = 1e-6


@dataclass(slots=True)
class MergeProposalBuilder:
    """Helper for constructing merge proposals from existing candidates."""

    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)
    _merge_history: set[tuple[int, int, str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._merge_history = set()

    def find_merge_pair(
        self,
        state: GepaState,
        dominators: Sequence[int],
    ) -> tuple[int, int] | None:
        """Randomly sample two distinct dominators."""
        if len(dominators) < 2:
            return None
        idx1, idx2 = self._rng.sample(list(dominators), 2)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        return idx1, idx2

    def find_common_ancestor(
        self,
        state: GepaState,
        idx1: int,
        idx2: int,
    ) -> int | None:
        """Return a valid common ancestor for the provided parents."""
        if idx1 == idx2:
            return None
        ancestors1 = self._collect_ancestors(state, idx1)
        ancestors2 = self._collect_ancestors(state, idx2)

        if idx1 in ancestors2 or idx2 in ancestors1:
            # Do not merge direct ancestor/descendant pairs.
            return None

        parent1 = state.candidates[idx1]
        parent2 = state.candidates[idx2]

        candidate_pool: list[tuple[int, float]] = []
        for ancestor_idx in ancestors1 & ancestors2:
            if ancestor_idx < 0 or ancestor_idx >= len(state.candidates):
                continue
            ancestor = state.candidates[ancestor_idx]
            if not self._has_desirable_predictor(ancestor, parent1, parent2):
                continue
            if not self._ancestor_score_ok(ancestor, parent1, parent2):
                continue
            weight = max(ancestor.avg_validation_score, 0.0) + _SCORE_EPSILON
            candidate_pool.append((ancestor_idx, weight))

        if not candidate_pool:
            return None

        indices, weights = zip(*candidate_pool)
        return self._rng.choices(indices, weights=weights, k=1)[0]

    def build_merged_candidate(
        self,
        state: GepaState,
        parent1_idx: int,
        parent2_idx: int,
        ancestor_idx: int,
    ) -> CandidateProgram:
        """Create a merged candidate following GEPA's crossover rules."""
        parent1 = state.candidates[parent1_idx]
        parent2 = state.candidates[parent2_idx]
        ancestor = state.candidates[ancestor_idx]

        if set(parent1.components) != set(parent2.components):
            raise ValueError(
                "Merge requires parents to share identical component sets."
            )

        parent1_score = parent1.avg_validation_score
        parent2_score = parent2.avg_validation_score

        merged_components: CandidateMap = {}
        for name in parent1.components:
            merged_components[name] = self._choose_component(
                component=name,
                ancestor=ancestor.components.get(name),
                parent1=parent1.components.get(name),
                parent2=parent2.components.get(name),
                parent1_score=parent1_score,
                parent2_score=parent2_score,
            )

        # Merge example banks from both parents
        example_bank = self._merge_example_banks(
            parent1.example_bank,
            parent2.example_bank,
            max_examples=state.config.example_bank.max_examples
            if state.config.example_bank
            else 50,
        )

        return CandidateProgram(
            idx=len(state.candidates),
            components=merged_components,
            parent_indices=[parent1_idx, parent2_idx],
            creation_type="merge",
            discovered_at_iteration=max(state.iteration, 0),
            discovered_at_evaluation=state.total_evaluations,
            example_bank=example_bank,
        )

    async def select_merge_subsample(
        self,
        state: GepaState,
        parent1_idx: int,
        parent2_idx: int,
    ) -> list[tuple[str, Case[Any, Any, Any]]]:
        """Return a stratified subsample of shared validation instances."""
        parent1 = state.candidates[parent1_idx]
        parent2 = state.candidates[parent2_idx]

        shared_ids = list(
            set(parent1.validation_scores) & set(parent2.validation_scores)
        )
        if not shared_ids:
            return []

        min_shared = max(1, state.config.min_shared_validation)
        if len(shared_ids) < min_shared:
            return []

        target = min(len(shared_ids), max(1, state.config.merge_subsample_size))
        buckets = self._build_score_buckets(parent1, parent2, shared_ids)

        selected_ids: list[str] = []
        per_bucket = max(1, math.ceil(target / 3))
        for bucket in buckets:
            if len(selected_ids) >= target:
                break
            to_take = min(per_bucket, target - len(selected_ids))
            selected_ids.extend(self._sample(bucket, to_take))

        if len(selected_ids) < target:
            remaining = [
                data_id for data_id in shared_ids if data_id not in selected_ids
            ]
            needed = target - len(selected_ids)
            selected_ids.extend(self._sample(remaining, needed))

        lookup = await self._build_validation_lookup(state.validation_set)
        subsample: list[tuple[str, Case[Any, Any, Any]]] = []
        for data_id in selected_ids:
            instance = lookup.get(data_id)
            if instance is not None:
                subsample.append((data_id, instance))
        return subsample

    def register_candidate(
        self,
        *,
        candidate: CandidateProgram,
        parent1_idx: int,
        parent2_idx: int,
    ) -> bool:
        """Record the merged candidate and return False if it was already seen."""
        descriptor = self._merge_descriptor(candidate, parent1_idx, parent2_idx)
        if descriptor in self._merge_history:
            return False
        self._merge_history.add(descriptor)
        return True

    def _merge_descriptor(
        self,
        candidate: CandidateProgram,
        parent1_idx: int,
        parent2_idx: int,
    ) -> tuple[int, int, str]:
        low, high = sorted((parent1_idx, parent2_idx))
        candidate_hash = self._components_hash(candidate.components.values())
        return low, high, candidate_hash

    def _components_hash(self, components: Iterable[ComponentValue]) -> str:
        hasher = hashlib.sha256()
        for component in sorted(components, key=lambda comp: comp.name):
            payload = f"{component.name}::{component.text}".encode("utf-8")
            hasher.update(payload)
        return hasher.hexdigest()

    async def _build_validation_lookup(
        self,
        validation_set: DataLoader[Any, Case[Any, Any, Any]] | None,
    ) -> dict[str, Case[Any, Any, Any]]:
        if validation_set is None or len(validation_set) == 0:
            return {}
        ids = list(await validation_set.all_ids())
        batch = await validation_set.fetch(ids)
        lookup: dict[str, Case[Any, Any, Any]] = {}
        for idx, instance in enumerate(batch):
            lookup[data_id_for_instance(instance, idx)] = instance
        return lookup

    def _sample(self, pool: Sequence[str], count: int) -> list[str]:
        if count <= 0 or not pool:
            return []
        if len(pool) <= count:
            # Deterministic order to keep reproducibility when taking entire pool.
            shuffled = list(pool)
            self._rng.shuffle(shuffled)
            return shuffled
        return self._rng.sample(list(pool), count)

    def _build_score_buckets(
        self,
        parent1: CandidateProgram,
        parent2: CandidateProgram,
        shared_ids: Sequence[str],
    ) -> tuple[list[str], list[str], list[str]]:
        bucket_p1: list[str] = []
        bucket_p2: list[str] = []
        bucket_tie: list[str] = []

        for data_id in shared_ids:
            score1 = parent1.validation_scores.get(data_id, 0.0)
            score2 = parent2.validation_scores.get(data_id, 0.0)
            if score1 > score2 + _SCORE_EPSILON:
                bucket_p1.append(data_id)
            elif score2 > score1 + _SCORE_EPSILON:
                bucket_p2.append(data_id)
            else:
                bucket_tie.append(data_id)

        return bucket_p1, bucket_p2, bucket_tie

    def _merge_example_banks(
        self,
        bank1: InMemoryExampleBank | None,
        bank2: InMemoryExampleBank | None,
        max_examples: int,
    ) -> InMemoryExampleBank | None:
        """Merge example banks from two parents.

        Combines examples from both banks, deduplicating by ID and
        respecting the max_examples limit.
        """
        if bank1 is None and bank2 is None:
            return None

        merged = InMemoryExampleBank()
        seen_ids: set[str] = set()

        # Add examples from both banks, preferring bank1 if there are duplicates
        for bank in [bank1, bank2]:
            if bank is None:
                continue
            for example in bank:
                if example.id not in seen_ids and len(merged) < max_examples:
                    merged.add(example)
                    seen_ids.add(example.id)

        return merged if len(merged) > 0 else None

    def _choose_component(
        self,
        *,
        component: str,
        ancestor: ComponentValue | None,
        parent1: ComponentValue | None,
        parent2: ComponentValue | None,
        parent1_score: float,
        parent2_score: float,
    ) -> ComponentValue:
        if parent1 is None or parent2 is None or ancestor is None:
            raise ValueError(
                f"Component '{component}' is missing from ancestor or parents."
            )

        text_ancestor = ancestor.text
        text1 = parent1.text
        text2 = parent2.text

        if text1 == text2:
            return parent1.model_copy()
        if text1 == text_ancestor:
            return parent2.model_copy()
        if text2 == text_ancestor:
            return parent1.model_copy()

        if parent1_score > parent2_score + _SCORE_EPSILON:
            return parent1.model_copy()
        if parent2_score > parent1_score + _SCORE_EPSILON:
            return parent2.model_copy()

        # Tie-breaker: randomly choose one of the parents.
        return self._rng.choice([parent1, parent2]).model_copy()

    def _ancestor_score_ok(
        self,
        ancestor: CandidateProgram,
        parent1: CandidateProgram,
        parent2: CandidateProgram,
    ) -> bool:
        ancestor_score = ancestor.avg_validation_score
        return (
            ancestor_score <= parent1.avg_validation_score + _SCORE_EPSILON
            and ancestor_score <= parent2.avg_validation_score + _SCORE_EPSILON
        )

    def _has_desirable_predictor(
        self,
        ancestor: CandidateProgram,
        parent1: CandidateProgram,
        parent2: CandidateProgram,
    ) -> bool:
        for name, ancestor_component in ancestor.components.items():
            parent1_component = parent1.components.get(name)
            parent2_component = parent2.components.get(name)
            if parent1_component is None or parent2_component is None:
                continue
            text_a = ancestor_component.text
            text_1 = parent1_component.text
            text_2 = parent2_component.text
            if text_a == text_1 and text_1 != text_2:
                return True
            if text_a == text_2 and text_1 != text_2:
                return True
        return False

    def _collect_ancestors(self, state: GepaState, idx: int) -> set[int]:
        seen: set[int] = set()
        stack: list[int] = list(state.candidates[idx].parent_indices)
        while stack:
            parent_idx = stack.pop()
            if parent_idx is None or parent_idx in seen:
                continue
            seen.add(parent_idx)
            if 0 <= parent_idx < len(state.candidates):
                stack.extend(state.candidates[parent_idx].parent_indices)
        return seen


__all__ = ["MergeProposalBuilder"]
