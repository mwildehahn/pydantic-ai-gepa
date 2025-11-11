"""High-level API for GEPA optimization of pydantic-ai agents."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model

from .adapter import AgentAdapter, ReflectionSampler
from .cache import CacheManager
from .components import (
    apply_candidate_to_agent,
    apply_candidate_to_agent_and_signature,
    extract_seed_candidate_with_signature,
    normalize_component_text,
)
from .gepa_graph import create_deps, create_gepa_graph
from .gepa_graph.models import GepaConfig, GepaResult, GepaState
from .gepa_graph.nodes import StartNode
from .signature import InputSpec
from .types import DataInst, RolloutOutput

# Type variable for the DataInst type
DataInstT = TypeVar("DataInstT", bound=DataInst)
ComponentSelectorLiteral = Literal["round_robin", "all"]
CandidateSelectorLiteral = Literal["pareto", "current_best"]

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent
    from pydantic_ai.models import Model


module_logger = logging.getLogger(__name__)


class LoggerProtocol(Protocol):
    """Minimal logger interface used by the optimization runner."""

    def log(self, message: str) -> None:
        ...


def _normalize_candidate(
    candidate: dict[str, Any] | None,
) -> dict[str, str]:
    if not candidate:
        return {}
    return {key: normalize_component_text(value) for key, value in candidate.items()}


class GepaOptimizationResult(BaseModel):
    """Result from GEPA optimization."""

    best_candidate: dict[str, str]
    """The best candidate found during optimization."""

    best_score: float
    """The validation score of the best candidate."""

    original_candidate: dict[str, str]
    """The original candidate before optimization."""

    original_score: float | None
    """The validation score of the original candidate (if evaluated)."""

    num_iterations: int
    """Number of optimization iterations performed."""

    num_metric_calls: int
    """Total number of metric evaluations performed."""

    raw_result: GepaResult | None = Field(default=None, exclude=True, repr=False)
    """Underlying GEPA graph result (for advanced users)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @contextmanager
    def apply_best(self, agent: AbstractAgent[Any, Any]) -> Iterator[None]:
        """Apply the best candidate to an agent as a context manager.

        Args:
            agent: The agent to apply the best candidate to.

        Yields:
            None while the context is active.
        """
        with apply_candidate_to_agent(agent, self.best_candidate):
            yield

    def improvement_ratio(self) -> float | None:
        """Calculate the improvement ratio from original to best.

        Returns:
            The ratio of improvement, or None if original score is not available.
        """
        if self.original_score is not None and self.original_score > 0:
            return (self.best_score - self.original_score) / self.original_score
        return None

    @contextmanager
    def apply_best_to(
        self,
        *,
        agent: AbstractAgent[Any, Any],
        input_type: InputSpec[BaseModel] | None = None,
    ) -> Iterator[None]:
        """Apply the best candidate to an agent and optional signature.

        Args:
            agent: The agent to apply the best candidate to.
            input_type: Optional structured input specification to also apply the candidate to.

        Yields:
            None while the context is active.
        """
        with apply_candidate_to_agent_and_signature(
            self.best_candidate, agent=agent, input_type=input_type
        ):
            yield


async def optimize_agent(
    agent: AbstractAgent[Any, Any],
    trainset: Sequence[DataInstT],
    *,
    metric: Callable[[DataInstT, RolloutOutput[Any]], tuple[float, str | None]],
    valset: Sequence[DataInstT] | None = None,
    input_type: InputSpec[BaseModel] | None = None,
    seed_candidate: dict[str, str] | None = None,
    reflection_model: Model | KnownModelName | str | None = None,
    candidate_selection_strategy: str = "pareto",
    skip_perfect_score: bool = True,
    reflection_minibatch_size: int = 3,
    perfect_score: float = 1.0,
    # Component selection configuration
    module_selector: str = "round_robin",
    # Merge-based configuration
    use_merge: bool = False,
    max_merge_invocations: int = 5,
    # Budget
    max_metric_calls: int = 200,
    # Caching configuration
    enable_cache: bool = False,
    cache_dir: str | None = None,
    cache_verbose: bool = False,
    # Logging
    logger: LoggerProtocol | None = None,
    # Reproducibility
    seed: int = 0,
    raise_on_exception: bool = True,
    # Testing support
    deterministic_proposer: Any | None = None,
    # Reflection sampler
    reflection_sampler: ReflectionSampler | None = None,
) -> GepaOptimizationResult:
    """Optimizes a pydantic-ai agent (and optional signature inputs) using the GEPA graph backend.

    This is the main entry point for prompt optimization. It takes a pydantic-ai
    agent and a dataset, and returns optimized prompts.

    Args:
        agent: The pydantic-ai agent to optimize.
        trainset: Training dataset (pydantic-evals Dataset or list of DataInst).
        metric: Function that computes (score, feedback) for each instance.
                The feedback (second element of tuple) is optional but recommended.
                If provided, it will be used to guide the optimization process.
        valset: Optional validation dataset. If not provided, trainset is used.
        input_type: Optional structured input specification whose instructions and
            field descriptions should be optimized alongside the agent's prompts.

        reflection_model: Model to use for reflection (proposing new prompts).
                         Can be a Model instance or a string like 'openai:gpt-4o'.
        candidate_selection_strategy: Strategy for selecting candidates ('pareto' or 'current_best').
        skip_perfect_score: Whether to skip updating if perfect score achieved on minibatch.
        reflection_minibatch_size: Number of examples to use for reflection in each proposal.
        perfect_score: Score threshold treated as perfect when `skip_perfect_score` is enabled.

        # Component selection configuration
        module_selector: Component selection strategy; must be 'round_robin' or 'all'.

        # Merge-based configuration
        use_merge: Whether to use the merge strategy for combining candidates.
        max_merge_invocations: Maximum number of merge invocations to perform.

        # Budget
        max_metric_calls: Maximum number of metric evaluations (budget).

        # Caching configuration
        enable_cache: Whether to enable caching of metric results for resumable runs.
        cache_dir: Directory to store cache files. If None, uses '.gepa_cache' in current directory.
        cache_verbose: Whether to log cache hits and misses.

        # Logging
        logger: Logger instance for tracking progress.
        # Reproducibility
        seed: Random seed for reproducibility.
        raise_on_exception: Whether to raise exceptions or continue on errors.

        # Testing support
        deterministic_proposer: For testing - a deterministic proposal function.

        # Reflection sampler
        reflection_sampler: Optional sampler for reflection records. If provided,
                               it will be called to sample records when needed. If None,
                               all reflection records are kept without sampling.

    Returns:
        GepaOptimizationResult with the best candidate and metadata.
    """
    train_instances = list(trainset)
    val_instances = list(valset) if valset is not None else train_instances

    extracted_seed_candidate = _normalize_candidate(
        extract_seed_candidate_with_signature(agent=agent, input_type=input_type)
    )
    if seed_candidate is None:
        normalized_seed_candidate = extracted_seed_candidate
    else:
        normalized_seed_candidate = _normalize_candidate(seed_candidate)
        if sorted(extracted_seed_candidate.keys()) != sorted(
            normalized_seed_candidate.keys()
        ):
            raise ValueError(
                "Seed candidate keys do not match extracted seed candidate keys"
            )

    cache_manager = None
    if enable_cache:
        cache_manager = CacheManager(
            cache_dir=cache_dir,
            enabled=True,
            verbose=cache_verbose,
        )

    adapter = AgentAdapter(
        agent=agent,
        metric=metric,
        input_type=input_type,
        reflection_sampler=reflection_sampler,
        reflection_model=reflection_model,
        cache_manager=cache_manager,
    )

    config = _build_gepa_config(
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        skip_perfect_score=skip_perfect_score,
        perfect_score=perfect_score,
        module_selector=module_selector,
        seed_candidate=normalized_seed_candidate,
        candidate_selection_strategy=candidate_selection_strategy,
        use_merge=use_merge,
        max_merge_invocations=max_merge_invocations,
        seed=seed,
    )

    deps = create_deps(adapter, config)
    deps.seed_candidate = normalized_seed_candidate
    if deterministic_proposer is not None:
        deps.proposal_generator = deterministic_proposer

    graph = create_gepa_graph(adapter=adapter, config=config)
    state = GepaState(
        config=config,
        training_set=train_instances,
        validation_set=val_instances,
    )

    try:
        async with graph.iter(StartNode(), state=state, deps=deps) as run:
            async for _ in run:
                pass
        run_result = run.result
        if run_result is None:
            raise RuntimeError("GEPA graph run did not produce a result.")
        gepa_result = run_result.output
    except Exception as exc:
        if raise_on_exception:
            raise
        if logger:
            logger.log(f"Optimization failed: {exc}")
        else:
            module_logger.exception("Optimization failed", exc_info=exc)
        return _fallback_result(normalized_seed_candidate)

    best_candidate_model = gepa_result.best_candidate or state.get_best_candidate()
    if best_candidate_model is None:
        best_candidate_dict = normalized_seed_candidate
    else:
        best_candidate_dict = _normalize_candidate(best_candidate_model.to_dict_str())

    if gepa_result.original_candidate is not None:
        original_candidate_model = gepa_result.original_candidate
    elif state.candidates:
        original_candidate_model = state.candidates[0]
    else:
        original_candidate_model = None

    if original_candidate_model is None:
        original_candidate_dict = normalized_seed_candidate
    else:
        original_candidate_dict = _normalize_candidate(
            original_candidate_model.to_dict_str()
        )

    result = GepaOptimizationResult(
        best_candidate=best_candidate_dict,
        best_score=gepa_result.best_score or 0.0,
        original_candidate=original_candidate_dict,
        original_score=gepa_result.original_score,
        num_iterations=gepa_result.iterations,
        num_metric_calls=gepa_result.total_evaluations,
        raw_result=gepa_result,
    )

    if cache_manager and cache_verbose:
        stats = cache_manager.get_cache_stats()
        if logger:
            logger.log(f"Cache stats: {stats}")
        else:
            module_logger.info("Cache stats: %s", stats)

    return result


def _build_gepa_config(
    *,
    max_metric_calls: int,
    reflection_minibatch_size: int,
    skip_perfect_score: bool,
    perfect_score: float,
    module_selector: str,
    seed_candidate: dict[str, str],
    candidate_selection_strategy: str,
    use_merge: bool,
    max_merge_invocations: int,
    seed: int,
) -> GepaConfig:
    component_selector: ComponentSelectorLiteral = _resolve_component_selector(
        module_selector, len(seed_candidate)
    )
    candidate_selector: CandidateSelectorLiteral = _resolve_candidate_selector(
        candidate_selection_strategy
    )

    return GepaConfig(
        max_evaluations=max_metric_calls,
        minibatch_size=reflection_minibatch_size,
        perfect_score=float(perfect_score),
        skip_perfect_score=skip_perfect_score,
        component_selector=component_selector,
        candidate_selector=candidate_selector,
        use_merge=use_merge,
        max_total_merges=max_merge_invocations,
        seed=seed,
    )


def _resolve_component_selector(
    selector: str, component_count: int
) -> ComponentSelectorLiteral:
    if selector not in {"round_robin", "all"}:
        raise ValueError(
            "module_selector must be either 'round_robin' or 'all' for gepa_graph runs."
        )
    if selector == "round_robin" and component_count <= 1:
        return "all"
    return cast(ComponentSelectorLiteral, selector)


def _resolve_candidate_selector(strategy: str) -> CandidateSelectorLiteral:
    if strategy not in {"pareto", "current_best"}:
        raise ValueError(
            "candidate_selection_strategy must be 'pareto' or 'current_best'."
        )
    return cast(CandidateSelectorLiteral, strategy)


def _fallback_result(seed_candidate: dict[str, str]) -> GepaOptimizationResult:
    candidate_copy = dict(seed_candidate)
    return GepaOptimizationResult(
        best_candidate=candidate_copy,
        best_score=0.0,
        original_candidate=dict(seed_candidate),
        original_score=None,
        num_iterations=0,
        num_metric_calls=0,
        raw_result=None,
    )
