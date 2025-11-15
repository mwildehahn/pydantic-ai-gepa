"""Reflect step - implements the core reflective mutation workflow."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import logfire
from pydantic_graph.beta import StepContext
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings

from ...adapter import (
    ComponentReflectiveDataset,
    ReflectiveDataset,
    SharedReflectiveDataset,
)
from ...evaluation_models import EvaluationBatch
from ...types import DataInst
from ..deps import GepaDeps
from ..evaluation import EvaluationResults
from ..models import CandidateProgram, ComponentValue, GepaState
from ..proposal.instruction import ProposalResult
from .continue_step import IterationAction

_IMPROVEMENT_EPSILON = 1e-9


async def reflect_step(ctx: StepContext[GepaState, GepaDeps, None]) -> IterationAction:
    """Generate and evaluate reflective mutations for the current candidate."""

    state = ctx.state
    deps = ctx.deps

    parent_idx, parent = _select_parent(state, deps)
    minibatch = await _sample_minibatch(state, deps)
    with logfire.span(
        "evaluate first minibatch",
        parent_idx=parent_idx,
        component_versions=_component_versions(parent),
        minibatch_size=len(minibatch),
    ):
        parent_results = await _evaluate_minibatch(
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
    _record_minibatch(parent, parent_results)
    _increment_budget(state, parent_results)
    parent_total, parent_avg = _summarize_scores(parent_results.scores)

    logfire.debug(
        "ReflectStep parent minibatch results",
        parent_idx=parent_idx,
        minibatch_scores=list(parent_results.scores),
        minibatch_total=parent_total,
        minibatch_average=parent_avg,
    )

    if not parent_results.has_trajectories():
        logfire.info(
            "ReflectStep skipping reflection due to missing trajectories",
            parent_idx=parent_idx,
        )
        state.last_accepted = False
        return "continue"

    if _should_skip_perfect(parent_results.scores, parent, state):
        logfire.info(
            "ReflectStep skipping reflection due to perfect minibatch",
            parent_idx=parent_idx,
            threshold=state.config.perfect_score,
            minibatch_total=parent_total,
        )
        state.last_accepted = False
        return "continue"

    components = _select_components(state, deps, parent_idx)
    logfire.debug(
        "ReflectStep selected components",
        components=components,
        parent_idx=parent_idx,
    )
    reflective_dataset = _build_reflective_dataset(
        deps=deps,
        state=state,
        candidate=parent,
        eval_results=parent_results,
        components=components,
    )
    reflection_model = _resolve_reflection_model(deps)

    with logfire.span(
        "propose new texts",
        parent_idx=parent_idx,
        components=components,
        model=reflection_model,
    ):
        proposal_result = await _propose_new_texts(
            deps=deps,
            state=state,
            parent=parent,
            reflective_dataset=reflective_dataset,
            components=components,
            model=reflection_model,
            model_settings=state.config.reflection_model_settings,
        )
        component_metadata = (
            proposal_result.component_metadata
            if state.config.track_component_hypotheses
            else None
        )
        reasoning = proposal_result.reasoning
        if reasoning is not None:
            logfire.info(
                "ReflectStep proposal reasoning",
                parent_idx=parent_idx,
                components=components,
                pattern=reasoning.pattern_discovery,
                hypothesis=reasoning.creative_hypothesis,
                approach=reasoning.experimental_approach,
                edge_insight=reasoning.edge_insight,
                success_checkpoint=reasoning.success_checkpoint,
                evolution_moves=reasoning.evolution_moves,
            )

    new_candidate = _create_candidate(
        state=state,
        parent=parent,
        parent_idx=parent_idx,
        new_texts=proposal_result.texts,
        metadata=component_metadata,
    )
    logfire.debug(
        "ReflectStep proposed candidate",
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
        new_results = await _evaluate_minibatch(
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
    _record_minibatch(new_candidate, new_results)
    _increment_budget(state, new_results)
    new_total, new_avg = _summarize_scores(new_results.scores)
    logfire.debug(
        "ReflectStep candidate minibatch results",
        candidate_idx=new_candidate.idx,
        parent_idx=parent_idx,
        minibatch_scores=list(new_results.scores),
        minibatch_total=new_total,
        minibatch_average=new_avg,
    )

    improved = _is_strict_improvement(
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
            "ReflectStep accepted candidate",
            **cast(dict[str, Any], decision_payload),
        )
        return "evaluate"

    state.last_accepted = False
    logfire.info(
        "ReflectStep rejected candidate",
        failure_reason="not_strict_improvement",
        **cast(dict[str, Any], decision_payload),
    )
    return "continue"


def _select_parent(
    state: GepaState,
    deps: GepaDeps,
) -> tuple[int, CandidateProgram]:
    if not state.candidates:
        raise ValueError("ReflectStep requires at least one candidate in state.")
    selector = deps.candidate_selector
    select_fn = getattr(selector, "select", None)
    if select_fn is None:
        select_fn = getattr(selector, "select_candidate")
    idx = select_fn(state)
    if idx is None:
        raise RuntimeError("Candidate selector must return an index.")
    parent = state.candidates[idx]
    return idx, parent


async def _sample_minibatch(
    state: GepaState,
    deps: GepaDeps,
) -> list[DataInst]:
    loader = state.training_set
    batch = await deps.batch_sampler.sample(
        training_set=loader,
        state=state,
        size=state.config.minibatch_size,
    )
    if len(batch) < 1:
        raise ValueError("BatchSampler returned an empty minibatch.")
    return batch


async def _evaluate_minibatch(
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
        capture_traces=capture_traces,
        max_concurrent=state.config.max_concurrent_evaluations,
    )


def _record_minibatch(
    candidate: CandidateProgram,
    results: EvaluationResults[str],
) -> None:
    candidate.minibatch_scores = list(results.scores)


def _increment_budget(
    state: GepaState,
    results: EvaluationResults[str],
) -> None:
    state.total_evaluations += len(results.data_ids)


def _summarize_scores(scores: Sequence[float]) -> tuple[float, float]:
    total = float(sum(scores))
    avg = total / len(scores) if scores else 0.0
    return total, avg


def _should_skip_perfect(
    scores: Sequence[float],
    candidate: CandidateProgram,
    state: GepaState,
) -> bool:
    if not state.config.skip_perfect_score:
        return False
    perfect = state.config.perfect_score
    total = float(sum(scores))
    minibatch_perfect = total >= perfect * len(scores)
    if not minibatch_perfect:
        return False
    if state.config.skip_perfect_requires_validation:
        return candidate.avg_validation_score >= perfect
    return True


def _select_components(
    state: GepaState,
    deps: GepaDeps,
    parent_idx: int,
) -> list[str]:
    selector = deps.component_selector
    select_fn = getattr(selector, "select", None)
    if select_fn is None:
        select_fn = getattr(selector, "select_components")
    return list(select_fn(state, parent_idx))


def _build_reflective_dataset(
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
        trajectories=list(eval_results.trajectories) if eval_results.trajectories is not None else None,
    )

    raw_dataset = deps.adapter.make_reflective_dataset(
        candidate=candidate.to_dict_str(),
        eval_batch=eval_batch,
        components_to_update=components,
    )
    if isinstance(raw_dataset, SharedReflectiveDataset):
        records_by_component = {component: list(raw_dataset.records) for component in components}
    else:
        records_by_component = {
            component: list(raw_dataset.records_by_component.get(component, []))
            for component in components
        }

    sampler = state.config.reflection_sampler
    if sampler is not None:
        max_records = state.config.reflection_sampler_max_records
        sampled = {
            component: sampler(records, max_records)
            for component, records in records_by_component.items()
        }
    else:
        sampled = records_by_component

    return ComponentReflectiveDataset(records_by_component=sampled)


def _resolve_reflection_model(
    deps: GepaDeps,
) -> Model | KnownModelName | str:
    model = deps.reflection_model
    if model is None:
        raise ValueError("reflection_model must be configured before running reflection.")
    return model


async def _propose_new_texts(
    *,
    deps: GepaDeps,
    state: GepaState,
    parent: CandidateProgram,
    reflective_dataset: ReflectiveDataset,
    components: Sequence[str],
    model: Model | KnownModelName | str,
    model_settings: ModelSettings | None = None,
) -> ProposalResult:
    proposal = deps.proposal_generator
    return await proposal.propose_texts(
        candidate=parent,
        reflective_data=reflective_dataset,
        components=components,
        model=model,
        iteration=state.iteration,
        current_best_score=state.best_score,
        parent_score=parent.avg_validation_score,
        model_settings=model_settings,
    )


def _create_candidate(
    *,
    state: GepaState,
    parent: CandidateProgram,
    parent_idx: int,
    new_texts: Mapping[str, str],
    metadata: Mapping[str, dict[str, Any]] | None = None,
) -> CandidateProgram:
    new_components: dict[str, ComponentValue] = {}
    for name, value in parent.components.items():
        new_components[name] = ComponentValue(
            name=name,
            text=value.text,
            version=value.version,
            metadata=None if value.metadata is None else dict(value.metadata),
        )
    for name, text in new_texts.items():
        existing = parent.components.get(name)
        base_version = existing.version if existing is not None else 0
        component_metadata = None
        if metadata and name in metadata:
            component_metadata = dict(metadata[name])
            if state.iteration >= 0:
                component_metadata.setdefault("iteration", state.iteration)
        new_components[name] = ComponentValue(
            name=name,
            text=text,
            version=base_version + 1,
            metadata=component_metadata,
        )
    return CandidateProgram(
        idx=len(state.candidates),
        components=new_components,
        creation_type="reflection",
        parent_indices=[parent_idx],
        discovered_at_iteration=state.iteration,
        discovered_at_evaluation=state.total_evaluations,
    )


def _is_strict_improvement(
    *,
    baseline_scores: Sequence[float],
    new_scores: Sequence[float],
) -> bool:
    improvement = sum(new_scores) - sum(baseline_scores)
    return improvement > _IMPROVEMENT_EPSILON


def _component_versions(candidate: CandidateProgram) -> Mapping[str, str]:
    return {
        name: component.text
        for name, component in candidate.components.items()
    }


__all__ = ["reflect_step"]
