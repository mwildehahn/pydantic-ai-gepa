"""ReflectNode - implements the core reflective mutation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from pydantic_ai.models import KnownModelName, Model

from ...types import DataInst
from ..evaluation import EvaluationResults
from ..models import CandidateProgram, ComponentValue, GepaState
from ..deps import GepaDeps
from .base import GepaNode, GepaRunContext

_IMPROVEMENT_EPSILON = 1e-9


@dataclass(slots=True)
class ReflectNode(GepaNode):
    """Generate and evaluate reflective mutations for the current candidate."""

    async def run(self, ctx: GepaRunContext):
        from .continue_node import ContinueNode  # Local imports avoid cycles
        from .evaluate import EvaluateNode

        state = ctx.state
        deps = ctx.deps

        parent_idx, parent = self._select_parent(state, deps)
        minibatch = self._sample_minibatch(state, deps)

        parent_results = await self._evaluate_minibatch(
            deps=deps,
            state=state,
            candidate=parent,
            batch=minibatch,
            capture_traces=True,
        )
        self._record_minibatch(parent, parent_results)
        self._increment_budget(state, parent_results)

        if self._should_skip_perfect(parent_results.scores, state):
            state.last_accepted = False
            return ContinueNode()

        components = self._select_components(state, deps, parent_idx)
        reflective_dataset = self._build_reflective_dataset(
            deps=deps,
            eval_results=parent_results,
            components=components,
        )
        reflection_model = self._resolve_reflection_model(deps)
        proposed_texts = await self._propose_new_texts(
            deps=deps,
            parent=parent,
            reflective_dataset=reflective_dataset,
            components=components,
            model=reflection_model,
        )
        new_candidate = self._create_candidate(
            state=state,
            parent=parent,
            parent_idx=parent_idx,
            new_texts=proposed_texts,
        )

        new_results = await self._evaluate_minibatch(
            deps=deps,
            state=state,
            candidate=new_candidate,
            batch=minibatch,
            capture_traces=False,
        )
        self._record_minibatch(new_candidate, new_results)
        self._increment_budget(state, new_results)

        if self._is_strict_improvement(
            baseline_scores=parent_results.scores,
            new_scores=new_results.scores,
        ):
            state.add_candidate(new_candidate)
            state.last_accepted = True
            state.schedule_merge(state.config.merges_per_accept)
            return EvaluateNode()

        state.last_accepted = False
        return ContinueNode()

    def _select_parent(
        self,
        state: GepaState,
        deps: GepaDeps,
    ) -> tuple[int, CandidateProgram]:
        if not state.candidates:
            raise ValueError("ReflectNode requires at least one candidate in state.")
        idx = deps.candidate_selector.select(state)
        if idx < 0 or idx >= len(state.candidates):
            raise IndexError(f"Candidate selector returned invalid index {idx}.")
        return idx, state.candidates[idx]

    def _sample_minibatch(
        self,
        state: GepaState,
        deps: GepaDeps,
    ) -> list[DataInst]:
        minibatch = deps.batch_sampler.sample(
            state.training_set,
            state,
            state.config.minibatch_size,
        )
        if not minibatch:
            raise ValueError("Batch sampler returned an empty minibatch.")
        return minibatch

    async def _evaluate_minibatch(
        self,
        *,
        deps: GepaDeps,
        state: GepaState,
        candidate: CandidateProgram,
        batch: Sequence[DataInst],
        capture_traces: bool,
    ) -> EvaluationResults[str]:
        return await deps.evaluator.evaluate_batch(
            candidate=candidate,
            batch=batch,
            adapter=deps.adapter,
            max_concurrent=state.config.max_concurrent_evaluations,
            capture_traces=capture_traces,
        )

    @staticmethod
    def _record_minibatch(
        candidate: CandidateProgram,
        results: EvaluationResults[str],
    ) -> None:
        candidate.minibatch_scores = list(results.scores)

    @staticmethod
    def _increment_budget(
        state: GepaState,
        results: EvaluationResults[str],
    ) -> None:
        state.total_evaluations += len(results.data_ids)

    def _should_skip_perfect(
        self,
        scores: Sequence[float],
        state: GepaState,
    ) -> bool:
        if not scores:
            return False
        config = state.config
        if not config.skip_perfect_score:
            return False
        threshold = config.perfect_score
        return all(score >= threshold for score in scores)

    def _select_components(
        self,
        state: GepaState,
        deps: GepaDeps,
        parent_idx: int,
    ) -> list[str]:
        components = deps.component_selector.select(state, parent_idx)
        if not components:
            raise ValueError("Component selector returned no components to update.")
        return components

    def _build_reflective_dataset(
        self,
        *,
        deps: GepaDeps,
        eval_results: EvaluationResults[str],
        components: Sequence[str],
    ) -> dict[str, list[dict]]:
        return deps.reflective_dataset_builder.build_dataset(
            eval_results=eval_results,
            components=components,
        )

    async def _propose_new_texts(
        self,
        *,
        deps: GepaDeps,
        parent: CandidateProgram,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, object]]],
        components: Sequence[str],
        model: Model | KnownModelName | str,
    ) -> dict[str, str]:
        return await deps.proposal_generator.propose_texts(
            candidate=parent,
            reflective_data=reflective_dataset,
            components=components,
            model=model,
        )

    def _create_candidate(
        self,
        *,
        state: GepaState,
        parent: CandidateProgram,
        parent_idx: int,
        new_texts: Mapping[str, str],
    ) -> CandidateProgram:
        updated_components = {
            name: self._update_component(component, new_texts.get(name))
            for name, component in parent.components.items()
        }

        return CandidateProgram(
            idx=len(state.candidates),
            components=updated_components,
            parent_indices=[parent_idx],
            creation_type="reflection",
            discovered_at_iteration=max(state.iteration, 0),
            discovered_at_evaluation=state.total_evaluations,
        )

    @staticmethod
    def _update_component(
        component: ComponentValue,
        new_text: str | None,
    ) -> ComponentValue:
        """Return an updated ComponentValue with optional text changes."""
        if new_text is None or new_text == component.text:
            return component.model_copy()
        return component.model_copy(
            update={
                "text": new_text,
                "version": component.version + 1,
            }
        )

    @staticmethod
    def _is_strict_improvement(
        *,
        baseline_scores: Sequence[float],
        new_scores: Sequence[float],
    ) -> bool:
        if not baseline_scores or not new_scores:
            return False
        if len(baseline_scores) != len(new_scores):
            return False
        baseline_total = sum(baseline_scores)
        new_total = sum(new_scores)
        return new_total > baseline_total + _IMPROVEMENT_EPSILON

    @staticmethod
    def _resolve_reflection_model(deps: GepaDeps) -> Model | KnownModelName | str:
        if deps.reflection_model is not None:
            return deps.reflection_model
        adapter_model = getattr(deps.adapter, "reflection_model", None)
        if adapter_model is None:
            raise ValueError("ReflectNode requires a reflection model to propose updates.")
        return adapter_model


__all__ = ["ReflectNode"]
