"""Evaluate node - runs full validation for the latest candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import logfire
from ...types import DataInst
from ..deps import GepaDeps
from ..evaluation import EvaluationResults
from ..models import CandidateProgram, ComponentValue, GepaState
from .base import GepaNode, GepaRunContext

if TYPE_CHECKING:
    from .continue_node import ContinueNode


@dataclass(slots=True)
class EvaluateNode(GepaNode):
    """Evaluate the most recent candidate on the validation set."""

    async def run(self, ctx: GepaRunContext) -> "ContinueNode":
        from .continue_node import (
            ContinueNode,
        )  # Local import to avoid circular dependency

        state = ctx.state
        previous_best_idx = state.best_candidate_idx
        previous_best_score = state.best_score
        candidate = self._current_candidate(state)
        validation_batch = self._get_validation_batch(state)

        with logfire.span(
            "evaluate candidate",
            candidate_idx=candidate.idx,
            validation_batch_size=len(validation_batch),
        ):
            results = await ctx.deps.evaluator.evaluate_batch(
                candidate=candidate,
                batch=validation_batch,
                adapter=ctx.deps.adapter,
                max_concurrent=state.config.max_concurrent_evaluations,
            )

        state.record_evaluation_errors(
            candidate_idx=candidate.idx,
            stage="validation",
            data_ids=results.data_ids,
            outputs=results.outputs,
        )
        validation_total, validation_avg = self._summarize_scores(results.scores)
        logfire.debug(
            "EvaluateNode validation results",
            candidate_idx=candidate.idx,
            validation_total=validation_total,
            validation_average=validation_avg,
            evaluation_count=len(results.scores),
        )

        self._apply_results(candidate, results)
        ctx.deps.pareto_manager.update_fronts(state, candidate.idx, results)
        state.recompute_best_candidate()
        new_best_idx = state.best_candidate_idx
        new_best_score = state.best_score
        if new_best_idx != previous_best_idx or new_best_score != previous_best_score:
            logfire.info(
                "EvaluateNode promoted best candidate",
                candidate_idx=new_best_idx,
                previous_best_idx=previous_best_idx,
                previous_best_score=previous_best_score,
                new_best_score=new_best_score,
            )
        else:
            logfire.debug(
                "EvaluateNode best candidate unchanged",
                candidate_idx=candidate.idx,
                best_candidate_idx=new_best_idx,
                best_score=new_best_score,
            )
        state.total_evaluations += len(results.data_ids)
        state.full_validations += 1
        self._hydrate_missing_components(candidate, ctx.deps)

        return ContinueNode()

    @staticmethod
    def _current_candidate(state: GepaState) -> CandidateProgram:
        if not state.candidates:
            raise ValueError("EvaluateNode requires at least one candidate in state.")
        return state.candidates[-1]

    @staticmethod
    def _get_validation_batch(state: GepaState) -> list[DataInst]:
        if not state.validation_set:
            raise ValueError(
                "GepaState.validation_set must be populated before evaluation."
            )
        return list(state.validation_set)

    @staticmethod
    def _apply_results(
        candidate: CandidateProgram, results: EvaluationResults[str]
    ) -> None:
        for data_id, score, output in results:
            candidate.record_validation(
                data_id=data_id,
                score=score,
                output=output,
            )

    @staticmethod
    def _summarize_scores(scores: Sequence[float]) -> tuple[float, float]:
        if not scores:
            return 0.0, 0.0
        total = float(sum(scores))
        return total, total / len(scores)

    def _hydrate_missing_components(
        self,
        candidate: CandidateProgram,
        deps: GepaDeps,
    ) -> None:
        components = deps.adapter.get_components()

        missing = {
            key: text
            for key, text in components.items()
            if key not in candidate.components
        }
        if not missing:
            return

        for name, text in missing.items():
            candidate.components[name] = ComponentValue(name=name, text=text)

        seed = deps.seed_candidate or {}
        updated_seed = dict(seed)
        for name, text in missing.items():
            updated_seed.setdefault(name, text)
        deps.seed_candidate = updated_seed


__all__ = ["EvaluateNode"]
