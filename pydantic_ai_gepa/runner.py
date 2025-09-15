"""High-level API for GEPA optimization of pydantic-ai agents."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gepa.api
from gepa.core.result import GEPAResult
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.reflective_mutation.base import LanguageModel, ReflectionComponentSelector

from .adapter import PydanticAIGEPAAdapter
from .components import apply_candidate_to_agent, extract_seed_candidate, validate_components
from .types import DataInst, RolloutOutput

if TYPE_CHECKING:
    from pydantic_ai.agent import Agent
    from pydantic_ai.models import Model


@dataclass
class GepaOptimizationResult:
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

    raw_result: GEPAResult[RolloutOutput] | None = None
    """The raw GEPA optimization result (for advanced users)."""

    @contextmanager
    def apply_best(self, agent: Agent[Any, Any]) -> Iterator[None]:
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


def optimize_agent_prompts(
    agent: Agent[Any, Any],
    trainset: Sequence[DataInst],
    *,
    metric: Callable[[DataInst, RolloutOutput], tuple[float, str | None]],
    valset: Sequence[DataInst] | None = None,
    components: Sequence[str] = ('instructions',),
    # Reflection-based configuration
    reflection_model: Model | str | None = None,
    candidate_selection_strategy: str = 'pareto',
    skip_perfect_score: bool = True,
    reflection_minibatch_size: int = 3,
    perfect_score: int = 1,
    # Component selection configuration
    module_selector: ReflectionComponentSelector | str = 'round_robin',
    # Merge-based configuration
    use_merge: bool = False,
    max_merge_invocations: int = 5,
    # Budget
    max_metric_calls: int = 200,
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
) -> GepaOptimizationResult:
    """Optimize agent prompts using GEPA.

    This is the main entry point for prompt optimization. It takes a pydantic-ai
    agent and a dataset, and returns optimized prompts.

    Args:
        agent: The pydantic-ai agent to optimize.
        trainset: Training dataset (pydantic-evals Dataset or list of DataInst).
        metric: Function that computes (score, feedback) for each instance.
        valset: Optional validation dataset. If not provided, trainset is used.
        components: Component names to optimize (default: just 'instructions').

        # Reflection-based configuration
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

    # Validate components
    components = validate_components(agent, components)

    # Extract seed candidate - but filter to only requested components
    full_candidate = extract_seed_candidate(agent)
    seed_candidate = {k: v for k, v in full_candidate.items() if k in components}

    # Create adapter
    adapter = PydanticAIGEPAAdapter(
        agent=agent,
        metric=metric,
        deterministic_proposer=deterministic_proposer,
    )

    # Handle reflection model
    reflection_lm: LanguageModel | str | None = None
    if reflection_model is not None:
        if isinstance(reflection_model, str):
            # Parse model string (e.g., 'openai:gpt-4o')
            reflection_lm = reflection_model
        else:
            # It's a Model instance
            # We'd need to wrap it for GEPA's interface
            # For v1, we'll require string format
            raise ValueError(
                "Model instances not yet supported for reflection_model. Please use a string like 'openai:gpt-4o'"
            )

    # Adjust module_selector based on components if needed
    # If only one component and module_selector is still default, use 'all'
    if module_selector == 'round_robin' and len(components) == 1:
        module_selector = 'all'

    # Run optimization
    raw_result: GEPAResult[RolloutOutput] = gepa.api.optimize(  # type: ignore[misc]
        adapter=adapter,
        seed_candidate=seed_candidate,
        trainset=train_instances,
        valset=val_instances,
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
    best_score = raw_result.val_aggregate_scores[raw_result.best_idx] if raw_result.val_aggregate_scores else 0.0

    # Get original score if available (assuming the first candidate is the seed)
    original_score = None
    if raw_result.candidates and len(raw_result.candidates) > 0:
        # Check if the first candidate is the seed candidate
        if raw_result.candidates[0] == seed_candidate:
            original_score = raw_result.val_aggregate_scores[0]
        else:
            # Search through all candidates for the seed
            for i, candidate in enumerate(raw_result.candidates):
                if candidate == seed_candidate:
                    original_score = raw_result.val_aggregate_scores[i]
                    break

    return GepaOptimizationResult(
        best_candidate=best_candidate,
        best_score=best_score,
        original_candidate=seed_candidate,
        original_score=original_score,
        num_iterations=raw_result.num_full_val_evals or len(raw_result.candidates),
        num_metric_calls=raw_result.total_metric_calls or 0,
        raw_result=raw_result,
    )
