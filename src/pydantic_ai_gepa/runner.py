"""High-level API for GEPA optimization of pydantic-ai agents."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

import gepa.api
from gepa.core.result import GEPAResult
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.reflective_mutation.base import (
    LanguageModel,
    ReflectionComponentSelector,
)
from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model

from .adapter import PydanticAIGEPAAdapter, ReflectionSampler
from .cache import CacheManager
from .components import (
    apply_candidate_to_agent,
    apply_candidate_to_agent_and_signature,
    extract_seed_candidate_with_signature,
    normalize_component_text,
)
from .types import DataInst, RolloutOutput

# Type variable for the DataInst type
DataInstT = TypeVar("DataInstT", bound=DataInst)

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent
    from pydantic_ai.models import Model


def _normalize_candidate(
    candidate: dict[str, Any] | None,
) -> dict[str, str]:
    if not candidate:
        return {}
    return {
        key: normalize_component_text(value)
        for key, value in candidate.items()
    }


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

    raw_result: GEPAResult[RolloutOutput[Any]] | None = Field(
        default=None, exclude=True, repr=False
    )
    """The raw GEPA optimization result (for advanced users)."""

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
        input_model: type[BaseModel] | None = None,
    ) -> Iterator[None]:
        """Apply the best candidate to an agent and optional signature.

        Args:
            agent: The agent to apply the best candidate to.
            input_model: Optional structured input model class to also apply the candidate to.

        Yields:
            None while the context is active.
        """
        with apply_candidate_to_agent_and_signature(
            self.best_candidate, agent=agent, input_model=input_model
        ):
            yield


class DefaultLanguageModel:
    """Simple LanguageModel wrapper using a pydantic-ai Agent returning text."""

    def __init__(self, model: Any | None):
        self._agent = Agent(model, output_type=str)

    def __call__(self, prompt: str) -> str:
        result = self._agent.run_sync(prompt)
        return result.output


def optimize_agent_prompts(
    agent: AbstractAgent[Any, Any],
    trainset: Sequence[DataInstT],
    *,
    metric: Callable[[DataInstT, RolloutOutput[Any]], tuple[float, str | None]],
    valset: Sequence[DataInstT] | None = None,
    input_model: type[BaseModel] | None = None,
    seed_candidate: dict[str, str] | None = None,
    # Reflection-based configuration
    reflection_lm: LanguageModel | None = None,
    reflection_model: Model | KnownModelName | str | None = None,
    candidate_selection_strategy: str = "pareto",
    skip_perfect_score: bool = True,
    reflection_minibatch_size: int = 3,
    perfect_score: int = 1,
    # Component selection configuration
    module_selector: ReflectionComponentSelector | str = "round_robin",
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
    run_dir: str | None = None,
    use_wandb: bool = False,
    wandb_api_key: str | None = None,
    wandb_init_kwargs: dict[str, Any] | None = None,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    track_best_outputs: bool = False,
    display_progress_bar: bool = False,
    # Reproducibility
    seed: int = 0,
    raise_on_exception: bool = True,
    # Testing support
    deterministic_proposer: Any | None = None,
    # Reflection sampler
    reflection_sampler: ReflectionSampler | None = None,
) -> GepaOptimizationResult:
    """Optimize agent (and optional signature) prompts using GEPA.

    This is the main entry point for prompt optimization. It takes a pydantic-ai
    agent and a dataset, and returns optimized prompts.

    Args:
        agent: The pydantic-ai agent to optimize.
        trainset: Training dataset (pydantic-evals Dataset or list of DataInst).
        metric: Function that computes (score, feedback) for each instance.
                The feedback (second element of tuple) is optional but recommended.
                If provided, it will be used to guide the optimization process.
        valset: Optional validation dataset. If not provided, trainset is used.
        input_model: Optional structured input model class whose instructions and
            field descriptions should be optimized alongside the agent's prompts.

        # Reflection-based configuration
        reflection_lm: LanguageModel to use for reflection (proposing new prompts).
        reflection_model: Model to use for reflection (proposing new prompts).
                         Can be a Model instance or a string like 'openai:gpt-4o'.
        candidate_selection_strategy: Strategy for selecting candidates ('pareto' or 'current_best').
        skip_perfect_score: Whether to skip updating if perfect score achieved on minibatch.
        reflection_minibatch_size: Number of examples to use for reflection in each proposal.
        perfect_score: The perfect score value to achieve (integer).

        # Component selection configuration
        module_selector: Component selection strategy. Can be a ReflectionComponentSelector
                        instance or a string ('round_robin', 'all').

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
        run_dir: Directory to save results to.
        use_wandb: Whether to use Weights and Biases for logging.
        wandb_api_key: API key for Weights and Biases.
        wandb_init_kwargs: Additional kwargs for wandb initialization.
        use_mlflow: Whether to use MLflow for logging.
        mlflow_tracking_uri: Tracking URI for MLflow.
        mlflow_experiment_name: Experiment name for MLflow.
        track_best_outputs: Whether to track best outputs on validation set.
        display_progress_bar: Whether to display a progress bar.

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
    # Convert datasets if needed
    train_instances = list(trainset)

    if valset is not None:
        val_instances = list(valset)
    else:
        # Use trainset as valset
        val_instances = train_instances

    # Extract seed candidate from agent and optional signature
    extracted_seed_candidate = _normalize_candidate(
        extract_seed_candidate_with_signature(
            agent=agent,
            input_model=input_model,
        )
    )
    if seed_candidate is None:
        seed_candidate = extracted_seed_candidate
    else:
        seed_candidate = _normalize_candidate(seed_candidate)
        if sorted(extracted_seed_candidate.keys()) != sorted(seed_candidate.keys()):
            raise ValueError(
                "Seed candidate keys do not match extracted seed candidate keys"
            )

    # Create cache manager if caching is enabled
    cache_manager = None
    if enable_cache:
        cache_manager = CacheManager(
            cache_dir=cache_dir,
            enabled=True,
            verbose=cache_verbose,
        )

    # Create adapter
    adapter = PydanticAIGEPAAdapter(
        agent=agent,
        metric=metric,
        signature_class=input_model,
        reflection_sampler=reflection_sampler,
        reflection_model=reflection_model,
        cache_manager=cache_manager,
    )

    if reflection_lm is None:
        reflection_lm = DefaultLanguageModel(reflection_model)

    # Adjust module_selector based on number of components if needed
    # If only one component and module_selector is still default, use 'all'
    if module_selector == "round_robin" and len(seed_candidate) == 1:
        module_selector = "all"

    # Run optimization
    raw_result: GEPAResult[RolloutOutput[Any]] = gepa.api.optimize(
        adapter=adapter,
        seed_candidate=seed_candidate,
        trainset=train_instances,
        valset=val_instances,
        # Budget
        max_metric_calls=max_metric_calls,
        # Reflection-based configuration
        reflection_lm=reflection_lm,
        candidate_selection_strategy=candidate_selection_strategy,
        skip_perfect_score=skip_perfect_score,
        reflection_minibatch_size=reflection_minibatch_size,
        perfect_score=perfect_score,
        # Component selection configuration
        module_selector=module_selector,
        # Merge-based configuration
        use_merge=use_merge,
        max_merge_invocations=max_merge_invocations,
        # Logging
        logger=logger,
        run_dir=run_dir,
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        track_best_outputs=track_best_outputs,
        display_progress_bar=display_progress_bar,
        # Reproducibility
        seed=seed,
        raise_on_exception=raise_on_exception,
    )

    # Extract results
    best_candidate = raw_result.best_candidate
    normalized_best_candidate = _normalize_candidate(best_candidate)
    normalized_seed_candidate = _normalize_candidate(seed_candidate)
    best_score = (
        raw_result.val_aggregate_scores[raw_result.best_idx]
        if raw_result.val_aggregate_scores
        else 0.0
    )

    # Get original score if available (assuming the first candidate is the seed)
    original_score = None
    if raw_result.candidates and len(raw_result.candidates) > 0:
        # Check if the first candidate is the seed candidate
        if (
            _normalize_candidate(raw_result.candidates[0])
            == normalized_seed_candidate
        ):
            original_score = raw_result.val_aggregate_scores[0]
        else:
            # Search through all candidates for the seed
            for i, candidate in enumerate(raw_result.candidates):
                if (
                    _normalize_candidate(candidate)
                    == normalized_seed_candidate
                ):
                    original_score = raw_result.val_aggregate_scores[i]
                    break

    result = GepaOptimizationResult(
        best_candidate=normalized_best_candidate,
        best_score=best_score,
        original_candidate=normalized_seed_candidate,
        original_score=original_score,
        num_iterations=raw_result.num_full_val_evals or len(raw_result.candidates),
        num_metric_calls=raw_result.total_metric_calls or 0,
        raw_result=raw_result,
    )

    # Log cache stats if caching was enabled
    if cache_manager and cache_verbose:
        stats = cache_manager.get_cache_stats()
        if logger:
            logger.log(f"Cache stats: {stats}")

    return result
