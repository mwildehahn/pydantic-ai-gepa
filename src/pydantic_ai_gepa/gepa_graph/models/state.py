"""State and configuration models for the GEPA graph implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ...types import DataInst
from .candidate import CandidateProgram
from .pareto import ParetoFrontEntry


class GenealogyRecord(BaseModel):
    """Tracks the ancestry information for a candidate."""

    candidate_idx: int
    parent_indices: list[int] = Field(default_factory=list)
    creation_type: Literal["seed", "reflection", "merge"]
    iteration: int

    model_config = ConfigDict()

    @field_validator("candidate_idx")
    @classmethod
    def _validate_candidate_idx(cls, value: int) -> int:
        if value < 0:
            raise ValueError("candidate_idx must be >= 0.")
        return value

    @field_validator("parent_indices")
    @classmethod
    def _validate_parent_indices(cls, values: list[int]) -> list[int]:
        if any(idx < 0 for idx in values):
            raise ValueError("parent_indices must be >= 0.")
        return values

    @field_validator("iteration")
    @classmethod
    def _validate_iteration(cls, value: int) -> int:
        if value < 0:
            raise ValueError("iteration must be >= 0.")
        return value


class GepaConfig(BaseModel):
    """Immutable configuration for GEPA optimization."""

    # Budget
    max_evaluations: int = 200
    max_iterations: int | None = None

    # Reflection
    minibatch_size: int = 3
    perfect_score: float = 1.0
    skip_perfect_score: bool = True

    # Component selection
    component_selector: Literal["round_robin", "all"] = "round_robin"
    candidate_selector: Literal["pareto", "current_best"] = "pareto"

    # Merge
    use_merge: bool = False
    merges_per_accept: int = 1
    max_total_merges: int = 5
    min_shared_validation: int = 3

    # Parallelism
    max_concurrent_evaluations: int = 10
    enable_parallel_evaluation: bool = True
    enable_parallel_minibatch: bool = True
    enable_parallel_reflection: bool = True

    # Evaluation policy
    validation_policy: Literal["full", "sparse"] = "full"

    # Reproducibility
    seed: int = 0

    model_config = ConfigDict(frozen=True)

    @field_validator("max_evaluations", "minibatch_size", "max_concurrent_evaluations")
    @classmethod
    def _validate_positive_int(cls, value: int, info: ValidationInfo) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0.")
        return value

    @field_validator("merges_per_accept", "max_total_merges", "min_shared_validation")
    @classmethod
    def _validate_non_negative_int(cls, value: int, info: ValidationInfo) -> int:
        if value < 0:
            raise ValueError(f"{info.field_name} must be >= 0.")
        return value

    @field_validator("max_iterations")
    @classmethod
    def _validate_optional_iteration(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_iterations must be > 0 when provided.")
        return value

    @field_validator("perfect_score")
    @classmethod
    def _validate_perfect_score(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("perfect_score must be > 0.")
        return value


class GepaState(BaseModel):
    """Shared mutable state that flows through the GEPA graph nodes."""

    iteration: int = -1
    candidates: list[CandidateProgram] = Field(default_factory=list)
    pareto_front: dict[str, ParetoFrontEntry] = Field(default_factory=dict)
    genealogy: list[GenealogyRecord] = Field(default_factory=list)

    last_accepted: bool = False
    merge_scheduled: int = 0
    stopped: bool = False
    stop_reason: str | None = None

    total_evaluations: int = 0
    full_validations: int = 0

    best_candidate_idx: int | None = None
    best_score: float | None = None

    config: GepaConfig

    training_set: Sequence[DataInst] = Field(exclude=True)
    validation_set: Sequence[DataInst] | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("training_set", mode="before")
    @classmethod
    def _coerce_training_set(cls, value: Sequence[DataInst] | None) -> list[DataInst]:
        if value is None:
            raise ValueError("training_set is required.")
        converted = list(value)
        if not converted:
            raise ValueError("training_set must contain at least one instance.")
        return converted

    @field_validator("validation_set", mode="before")
    @classmethod
    def _coerce_validation_set(
        cls, value: Sequence[DataInst] | None
    ) -> list[DataInst] | None:
        if value is None:
            return None
        converted = list(value)
        if not converted:
            return None
        return converted

    @model_validator(mode="after")
    def _default_validation_set(self) -> GepaState:
        if self.validation_set is None:
            self.validation_set = list(self.training_set)
        return self

    def add_candidate(
        self,
        candidate: CandidateProgram,
        *,
        auto_assign_idx: bool = True,
    ) -> CandidateProgram:
        """Append a candidate to the state and record its genealogy."""
        expected_idx = len(self.candidates)
        if auto_assign_idx and candidate.idx != expected_idx:
            candidate = candidate.model_copy(update={"idx": expected_idx})
        elif not auto_assign_idx and candidate.idx != expected_idx:
            raise ValueError(
                f"Candidate idx {candidate.idx} does not match expected {expected_idx}."
            )

        self.candidates.append(candidate)
        self.genealogy.append(
            GenealogyRecord(
                candidate_idx=candidate.idx,
                parent_indices=list(candidate.parent_indices),
                creation_type=candidate.creation_type,
                iteration=candidate.discovered_at_iteration,
            )
        )
        return candidate

    def get_best_candidate(self) -> CandidateProgram | None:
        """Return the current best candidate if available."""
        if self.best_candidate_idx is None:
            return None
        if self.best_candidate_idx >= len(self.candidates):
            return None
        return self.candidates[self.best_candidate_idx]

    def recompute_best_candidate(self) -> CandidateProgram | None:
        """Recalculate the best candidate based on validation scores."""
        best_idx = None
        best_score = float("-inf")
        best_coverage = -1

        for idx, candidate in enumerate(self.candidates):
            if not candidate.validation_scores:
                continue
            coverage = candidate.coverage
            avg = candidate.avg_validation_score
            if avg > best_score or (avg == best_score and coverage > best_coverage):
                best_score = avg
                best_idx = idx
                best_coverage = coverage

        self.best_candidate_idx = best_idx
        self.best_score = None if best_idx is None else best_score
        return self.get_best_candidate()

    def schedule_merge(self, count: int) -> None:
        """Schedule upcoming merge operations."""
        if count < 0:
            raise ValueError("count must be >= 0.")
        self.merge_scheduled += count

    def budget_remaining(self) -> int:
        """Return the number of evaluations remaining in the budget."""
        remaining = self.config.max_evaluations - self.total_evaluations
        return max(0, remaining)

    def mark_stopped(self, *, reason: str | None = None) -> None:
        """Mark the run as stopped with an optional reason."""
        self.stopped = True
        self.stop_reason = reason
