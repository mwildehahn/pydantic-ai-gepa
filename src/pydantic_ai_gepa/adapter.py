"""GEPA adapter for pydantic-ai agents with single signature optimization."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.models import KnownModelName, Model

from .components import apply_candidate_to_agent
from .reflection import propose_new_texts
from .signature import Signature, apply_candidate_to_signature
from .signature_agent import SignatureAgent
from .cache import CacheManager
from .types import DataInst, DataInstWithPrompt, RolloutOutput, Trajectory

# Type variable for the DataInst type
DataInstT = TypeVar("DataInstT", bound=DataInst)

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent
    from pydantic_ai.messages import ModelMessage


class ReflectionSampler(Protocol):
    """Protocol for sampling reflection records."""

    def __call__(
        self, records: list[dict[str, Any]], max_records: int
    ) -> list[dict[str, Any]]:
        """Sample records for reflection.

        Args:
            records: All reflection records available.
            max_records: Maximum number of records to return.

        Returns:
            Sampled list of records (up to max_records).
        """
        ...


class PydanticAIGEPAAdapter(
    Generic[DataInstT], GEPAAdapter[DataInstT, Trajectory, RolloutOutput[Any]]
):
    """GEPA adapter for optimizing a single pydantic-ai agent with an optional signature.

    This adapter connects pydantic-ai agents to the GEPA optimization engine,
    enabling prompt optimization through evaluation and reflection. It focuses on
    optimizing a single agent's instructions and system prompts, optionally with
    a single Signature class for structured input formatting.
    """

    def __init__(
        self,
        agent: AbstractAgent[Any, Any],
        metric: Callable[[DataInstT, RolloutOutput[Any]], tuple[float, str | None]],
        *,
        signature_class: type[Signature] | None = None,
        reflection_sampler: ReflectionSampler | None = None,
        reflection_model: Model | KnownModelName | str | None = None,
        cache_manager: CacheManager | None = None,
    ):
        """Initialize the adapter.

        Args:
            agent: The pydantic-ai agent to optimize.
            metric: A function that computes (score, feedback) for a data instance
                   and its output. Higher scores are better. The feedback string
                   (second element) is optional but recommended for better optimization.
            signature_class: Optional single Signature class whose instructions and field
                            descriptions will be optimized alongside the agent's prompts.
            reflection_sampler: Optional sampler for reflection records. If provided,
                               it will be called to sample records when needed. If None,
                               all reflection records are kept without sampling.
            reflection_model: The model to use for reflection.
            cache_manager: The cache manager to use for caching.
        """
        self.agent = agent
        self.metric = metric
        self.signature_class = signature_class
        self.reflection_sampler = reflection_sampler
        self.reflection_model = reflection_model
        self.cache_manager = cache_manager

    def evaluate(
        self,
        batch: list[DataInstT],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput[Any]]:
        """Evaluate the candidate on a batch of data instances.

        Args:
            batch: List of data instances to evaluate.
            candidate: Candidate mapping component names to text.
            capture_traces: Whether to capture trajectories for reflection.

        Returns:
            EvaluationBatch with outputs, scores, and optionally trajectories.
        """
        outputs: list[RolloutOutput[Any]] = []
        scores: list[float] = []
        trajectories: list[Trajectory] | None = [] if capture_traces else None

        # Apply the candidate to the agent and optionally the signature
        with self._apply_candidate(candidate):
            for data_inst in batch:
                result = self.process_data_instance(
                    data_inst,
                    capture_traces,
                    candidate,
                )

                outputs.append(result["output"])
                scores.append(result["score"])

                if trajectories is not None and "trajectory" in result:
                    trajectories.append(result["trajectory"])

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def _apply_candidate(self, candidate: dict[str, str]):
        """Context manager to apply candidate to both agent and signature.

        Args:
            candidate: The candidate mapping component names to text.

        Returns:
            Context manager that applies the candidate.
        """
        from contextlib import ExitStack

        stack = ExitStack()

        # Apply to agent
        stack.enter_context(apply_candidate_to_agent(self.agent, candidate))

        # Apply to signature if provided
        if self.signature_class:
            stack.enter_context(
                apply_candidate_to_signature(self.signature_class, candidate)
            )

        return stack

    def process_data_instance(
        self,
        data_inst: DataInstT,
        capture_traces: bool = False,
        candidate: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Process a single data instance and return results.

        Args:
            data_inst: The data instance to process.
            capture_traces: Whether to capture trajectory information.

        Returns:
            Dictionary containing 'output', 'score', and optionally 'trajectory'.
        """
        try:
            # Check cache first for agent run (if we have a current candidate)
            if self.cache_manager and candidate:
                cached_agent_result = self.cache_manager.get_cached_agent_run(
                    data_inst,
                    candidate,
                    capture_traces,
                )

                if cached_agent_result is not None:
                    trajectory, output = cached_agent_result
                else:
                    # Run the agent and cache the result
                    if capture_traces:
                        trajectory, output = self._run_with_trace(data_inst)
                    else:
                        output = self._run_simple(data_inst)
                        trajectory = None

                    # Cache the agent run result
                    self.cache_manager.cache_agent_run(
                        data_inst,
                        candidate,
                        trajectory,
                        output,
                        capture_traces,
                    )
            else:
                # No caching, run normally
                if capture_traces:
                    trajectory, output = self._run_with_trace(data_inst)
                else:
                    output = self._run_simple(data_inst)
                    trajectory = None

            # Compute score using the metric and capture optional feedback
            # Use caching if available and we have a current candidate
            if self.cache_manager and candidate:
                # Check cache first
                cached_result = self.cache_manager.get_cached_metric_result(
                    data_inst,
                    output,
                    candidate,
                )

                if cached_result is not None:
                    score, metric_feedback = cached_result
                else:
                    # Call metric and cache result
                    score, metric_feedback = self.metric(data_inst, output)
                    self.cache_manager.cache_metric_result(
                        data_inst,
                        output,
                        candidate,
                        score,
                        metric_feedback,
                    )
            else:
                # No caching, call metric directly
                score, metric_feedback = self.metric(data_inst, output)

            # Attach metric-provided feedback to the trajectory if captured
            if trajectory is not None:
                trajectory.metric_feedback = metric_feedback

            result: dict[str, Any] = {
                "output": output,
                "score": score,
            }
            if trajectory is not None:
                result["trajectory"] = trajectory

            return result

        except Exception as e:
            # Handle errors gracefully
            output = RolloutOutput.from_error(e)
            trajectory = (
                Trajectory(messages=[], final_output=None, error=str(e))
                if capture_traces
                else None
            )

            error_result: dict[str, Any] = {
                "output": output,
                "score": 0.0,  # Failed execution gets score 0
            }
            if trajectory is not None:
                error_result["trajectory"] = trajectory

            return error_result

    def _run_with_trace(
        self, instance: DataInstT
    ) -> tuple[Trajectory, RolloutOutput[Any]]:
        """Run the agent and capture the trajectory.

        Args:
            instance: The data instance to run.

        Returns:
            Tuple of (trajectory, output).
        """
        messages: list[ModelMessage] = []

        try:
            if isinstance(instance, DataInstWithPrompt):
                # Run the agent and capture messages
                result = self.agent.run_sync(
                    instance.user_prompt.content,
                    message_history=instance.message_history,
                )
            else:
                assert isinstance(self.agent, SignatureAgent)
                result = self.agent.run_signature_sync(
                    instance.signature,
                    message_history=instance.message_history,
                )

            messages = result.new_messages()
            final_output = result.output
            target_agent = self.agent
            if isinstance(target_agent, WrapperAgent):
                target_agent = target_agent.wrapped

            system_prompt = None
            if hasattr(target_agent.model, "_map_messages"):
                model_messages = asyncio.run(target_agent.model._map_messages(messages))  # type: ignore
                system_prompt = "\n".join(
                    [m["content"] for m in model_messages if m["role"] == "system"]
                )

            trajectory = Trajectory(
                messages=messages,
                system_prompt=system_prompt,
                final_output=final_output,
                error=None,
                usage=asdict(result.usage()),  # Convert RunUsage to dict
                data_inst=instance,
            )
            output = RolloutOutput.from_success(final_output)

            return trajectory, output
        except Exception as e:
            trajectory = Trajectory(messages=messages, final_output=None, error=str(e))
            output = RolloutOutput.from_error(e)
            return trajectory, output

    def _run_simple(self, instance: DataInstT) -> RolloutOutput[Any]:
        """Run the agent without capturing traces.

        Args:
            instance: The data instance to run.

        Returns:
            The rollout output.
        """
        try:
            if isinstance(instance, DataInstWithPrompt):
                result = self.agent.run_sync(
                    instance.user_prompt.content,
                    message_history=instance.message_history,
                )
            else:
                assert isinstance(self.agent, SignatureAgent)
                result = self.agent.run_signature_sync(
                    instance.signature,
                    message_history=instance.message_history,
                )

            return RolloutOutput.from_success(result.output)
        except Exception as e:
            return RolloutOutput.from_error(e)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput[Any]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build a reflective dataset for instruction refinement.

        Args:
            candidate: The candidate that was evaluated.
            eval_batch: The evaluation results with trajectories.
            components_to_update: Component names to update.

        Returns:
            Mapping from component name to list of reflection records.
        """
        if not eval_batch.trajectories:
            # No trajectories available, return empty dataset
            return {comp: [] for comp in components_to_update}

        # Build reflection records from trajectories
        reflection_records: list[dict[str, Any]] = []
        for trajectory, output, score in zip(
            eval_batch.trajectories,
            eval_batch.outputs,
            eval_batch.scores,
        ):
            record: dict[str, Any] = trajectory.to_reflective_record()

            # Add score and success information
            record["score"] = score
            record["success"] = output.success
            if output.error_message:
                record["error_message"] = output.error_message

            if trajectory.system_prompt:
                record["system_prompt"] = trajectory.system_prompt

            # Use metric feedback if available, otherwise use a simple fallback
            feedback_text = trajectory.metric_feedback

            if not feedback_text:
                # Simple fallback when metric doesn't provide feedback
                if score >= 0.8:
                    feedback_text = "Good response"
                elif score >= 0.5:
                    feedback_text = "Adequate response, could be improved"
                else:
                    feedback_text = f"Poor response (score: {score:.2f})"
                    if output.error_message:
                        feedback_text += f" - Error: {output.error_message}"

            record["feedback"] = feedback_text
            reflection_records.append(record)

        # Apply sampling if a sampler is configured
        if self.reflection_sampler and reflection_records:
            # Let the sampler determine its own max_records internally
            # For backward compatibility, we can use a reasonable default
            reflection_records = self.reflection_sampler(
                reflection_records, max_records=10
            )

        # For pydantic-ai, all components work together, so they all need
        # the same reflection data to understand the full context
        return {comp: reflection_records for comp in components_to_update}

    def propose_new_texts(  # type: ignore[override]
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        return propose_new_texts(
            candidate,
            reflective_dataset,
            components_to_update,
            self.reflection_model,
        )
