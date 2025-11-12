"""ReflectNode - implements the core reflective mutation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import logfire

from pydantic_ai.models import KnownModelName, Model

from ...adapter import (
    ComponentReflectiveDataset,
    ReflectiveDataset,
    SharedReflectiveDataset,
)
from ...evaluation_models import EvaluationBatch
from ...types import DataInst
from ..evaluation import EvaluationResults
from ..models import CandidateProgram, ComponentValue, GepaConfig, GepaState
from ..deps import GepaDeps
from .base import GepaNode, GepaRunContext
from .continue_node import ContinueNode
from .evaluate import EvaluateNode

_IMPROVEMENT_EPSILON = 1e-9

@dataclass(slots=True)
class ReflectNode(GepaNode):
    """Generate and evaluate reflective mutations for the current candidate."""

    async def run(self, ctx: GepaRunContext) -> EvaluateNode | ContinueNode:
        state = ctx.state
        deps = ctx.deps

        parent_idx, parent = self._select_parent(state, deps)
        minibatch = self._sample_minibatch(state, deps)
        with logfire.span(
            "evaluate first minibatch",
            parent_idx=parent_idx,
            component_versions=self._component_versions(parent),
            minibatch_size=len(minibatch),
        ):
            parent_results = await self._evaluate_minibatch(
                deps=deps,
                state=state,
                candidate=parent,
                batch=minibatch,
                capture_traces=True,
            )

        state.record_evaluation_errors(
            candidate_idx=parent_idx,
            stage="reflection_parent",
            data_ids=parent_results.data_ids,
            outputs=parent_results.outputs,
        )
        self._record_minibatch(parent, parent_results)
        self._increment_budget(state, parent_results)
        parent_total, parent_avg = self._summarize_scores(parent_results.scores)

        logfire.debug(
            "ReflectNode parent minibatch results",
            parent_idx=parent_idx,
            minibatch_scores=list(parent_results.scores),
            minibatch_total=parent_total,
            minibatch_average=parent_avg,
        )

        if self._should_skip_perfect(parent_results.scores, state):
            logfire.info(
                "ReflectNode skipping reflection due to perfect minibatch",
                parent_idx=parent_idx,
                threshold=state.config.perfect_score,
                minibatch_total=parent_total,
            )
            state.last_accepted = False
            return ContinueNode()

        components = self._select_components(state, deps, parent_idx)
        logfire.debug(
            "ReflectNode selected components",
            components=components,
            parent_idx=parent_idx,
        )
        reflective_dataset = self._build_reflective_dataset(
            deps=deps,
            state=state,
            candidate=parent,
            eval_results=parent_results,
            components=components,
        )
        reflection_model = self._resolve_reflection_model(deps)

        with logfire.span(
            "propose new texts",
            parent_idx=parent_idx,
            components=components,
            model=reflection_model,
        ):
            proposed_texts = await self._propose_new_texts(
                deps=deps,
                state=state,
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
        logfire.debug(
            "ReflectNode proposed candidate",
            candidate_idx=new_candidate.idx,
            parent_idx=parent_idx,
            updated_components=sorted(
                name
                for name, value in new_candidate.components.items()
                if value.text != parent.components[name].text
            )
            or components,
        )

        with logfire.span(
            "evaluate new candidate",
            candidate_idx=new_candidate.idx,
            parent_idx=parent_idx,
        ):
            new_results = await self._evaluate_minibatch(
                deps=deps,
                state=state,
                candidate=new_candidate,
                batch=minibatch,
                capture_traces=False,
            )

        state.record_evaluation_errors(
            candidate_idx=new_candidate.idx,
            stage="reflection_candidate",
            data_ids=new_results.data_ids,
            outputs=new_results.outputs,
        )
        self._record_minibatch(new_candidate, new_results)
        self._increment_budget(state, new_results)
        new_total, new_avg = self._summarize_scores(new_results.scores)
        logfire.debug(
            "ReflectNode candidate minibatch results",
            candidate_idx=new_candidate.idx,
            parent_idx=parent_idx,
            minibatch_scores=list(new_results.scores),
            minibatch_total=new_total,
            minibatch_average=new_avg,
        )

        improved = self._is_strict_improvement(
            baseline_scores=parent_results.scores,
            new_scores=new_results.scores,
        )
        decision_payload = dict(
            parent_idx=parent_idx,
            candidate_idx=new_candidate.idx,
            baseline_total=parent_total,
            candidate_total=new_total,
            improvement=improved,
        )
        if improved:
            state.add_candidate(new_candidate)
            state.last_accepted = True
            state.schedule_merge(state.config.merges_per_accept)
            logfire.info(
                "ReflectNode accepted candidate",
                **decision_payload,
            )
            return EvaluateNode()

        state.last_accepted = False
        logfire.info(
            "ReflectNode rejected candidate",
            failure_reason="not_strict_improvement",
            **decision_payload,
        )
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
        state: GepaState,
        candidate: CandidateProgram,
        eval_results: EvaluationResults[str],
        components: Sequence[str],
    ) -> ReflectiveDataset:
        eval_batch = EvaluationBatch(
            outputs=list(eval_results.outputs),
            scores=list(eval_results.scores),
            trajectories=eval_results.trajectories,
        )
        dataset = deps.adapter.make_reflective_dataset(
            candidate=candidate.to_dict_str(),
            eval_batch=eval_batch,
            components_to_update=list(components),
        )
        return self._apply_reflection_sampler(dataset, state.config)

    async def _propose_new_texts(
        self,
        *,
        deps: GepaDeps,
        state: GepaState,
        parent: CandidateProgram,
        reflective_dataset: ReflectiveDataset,
        components: Sequence[str],
        model: Model | KnownModelName | str,
    ) -> dict[str, str]:
        return await deps.proposal_generator.propose_texts(
            candidate=parent,
            reflective_data=reflective_dataset,
            components=components,
            model=model,
            iteration=state.iteration,
            current_best_score=state.best_score,
            parent_score=parent.avg_validation_score
            if parent.validation_scores
            else None,
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
    def _summarize_scores(scores: Sequence[float]) -> tuple[float, float]:
        if not scores:
            return 0.0, 0.0
        total = float(sum(scores))
        return total, total / len(scores)

    @staticmethod
    def _resolve_reflection_model(deps: GepaDeps) -> Model | KnownModelName | str:
        if deps.reflection_model is None:
            raise ValueError(
                "ReflectNode requires `reflection_model` to be set on GepaConfig."
            )
        return deps.reflection_model

    @staticmethod
    def _apply_reflection_sampler(
        dataset: ReflectiveDataset,
        config: GepaConfig,
    ) -> ReflectiveDataset:
        sampler = config.reflection_sampler
        if sampler is None:
            return dataset

        max_records = config.reflection_sampler_max_records
        if isinstance(dataset, SharedReflectiveDataset):
            sampled = sampler(list(dataset.records), max_records=max_records)
            return SharedReflectiveDataset(records=sampled)

        sampled_map: dict[str, list[dict]] = {}
        for component, records in dataset.records_by_component.items():
            if not records:
                sampled_map[component] = []
                continue
            sampled_map[component] = sampler(list(records), max_records=max_records)
        return ComponentReflectiveDataset(records_by_component=sampled_map)

    @staticmethod
    def _component_versions(candidate: CandidateProgram) -> dict[str, int]:
        return {
            name: component.version for name, component in candidate.components.items()
        }


__all__ = ["ReflectNode"]
