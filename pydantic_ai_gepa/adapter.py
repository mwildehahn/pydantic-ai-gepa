"""GEPA adapter for pydantic-ai agents."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter, ProposalFn

from .components import apply_candidate_to_agent_and_signatures
from .signature import Signature
from .types import DataInst, RolloutOutput, Trajectory

if TYPE_CHECKING:
    from pydantic_ai.agent import Agent
    from pydantic_ai.messages import ModelMessage


class PydanticAIGEPAAdapter(GEPAAdapter[DataInst, Trajectory, RolloutOutput]):
    """GEPA adapter for pydantic-ai agents.

    This adapter connects pydantic-ai agents to the GEPA optimization engine,
    enabling prompt optimization through evaluation and reflection.
    """

    def __init__(
        self,
        agent: Agent[Any, Any],
        metric: Callable[[DataInst, RolloutOutput], tuple[float, str | None]],
        *,
        signatures: Sequence[type[Signature]] | None = None,
        deterministic_proposer: ProposalFn | None = None,
    ):
        """Initialize the adapter.

        Args:
            agent: The pydantic-ai agent to optimize.
            metric: A function that computes (score, feedback) for a data instance
                   and its output. Higher scores are better.
            signatures: Optional list of Signature classes whose prompts will be optimized.
            deterministic_proposer: Optional deterministic proposer for testing.
                                   If provided, this will be used as propose_new_texts.
        """
        self.agent = agent
        self.metric = metric
        self.signatures = signatures or []
        self.propose_new_texts = deterministic_proposer

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """Evaluate the candidate on a batch of data instances.

        Args:
            batch: List of data instances to evaluate.
            candidate: Candidate mapping component names to text.
            capture_traces: Whether to capture trajectories for reflection.

        Returns:
            EvaluationBatch with outputs, scores, and optionally trajectories.
        """
        outputs: list[RolloutOutput] = []
        scores: list[float] = []
        trajectories: list[Trajectory] | None = [] if capture_traces else None

        # Apply the candidate to the agent and signatures
        with apply_candidate_to_agent_and_signatures(candidate, agent=self.agent, signatures=self.signatures):
            for data_inst in batch:
                result = self.process_data_instance(data_inst, capture_traces)

                outputs.append(result['output'])
                scores.append(result['score'])

                if trajectories is not None and 'trajectory' in result:
                    trajectories.append(result['trajectory'])

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def process_data_instance(self, data_inst: DataInst, capture_traces: bool = False) -> dict[str, Any]:
        """Process a single data instance and return results.

        Args:
            data_inst: The data instance to process.
            capture_traces: Whether to capture trajectory information.

        Returns:
            Dictionary containing 'output', 'score', and optionally 'trajectory'.
        """
        try:
            # Run the agent
            if capture_traces:
                trajectory, output = self._run_with_trace(data_inst)
            else:
                output = self._run_simple(data_inst)
                trajectory = None

            # Compute score using the metric
            score, _ = self.metric(data_inst, output)

            result: dict[str, Any] = {
                'output': output,
                'score': score,
            }
            if trajectory is not None:
                result['trajectory'] = trajectory

            return result

        except Exception as e:
            # Handle errors gracefully
            output = RolloutOutput.from_error(e)
            trajectory = Trajectory(messages=[], final_output=None, error=str(e)) if capture_traces else None

            error_result: dict[str, Any] = {
                'output': output,
                'score': 0.0,  # Failed execution gets score 0
            }
            if trajectory is not None:
                error_result['trajectory'] = trajectory

            return error_result

    def _run_with_trace(self, instance: DataInst) -> tuple[Trajectory, RolloutOutput]:
        """Run the agent and capture the trajectory.

        Args:
            instance: The data instance to run.

        Returns:
            Tuple of (trajectory, output).
        """
        messages: list[ModelMessage] = []

        try:
            # Run the agent and capture messages
            result = self.agent.run_sync(
                instance.user_prompt.content,
                message_history=instance.message_history,
            )
            messages = result.new_messages()
            final_output = result.output

            trajectory = Trajectory(
                messages=messages,
                final_output=final_output,
                error=None,
                usage=asdict(result.usage()),  # Convert RunUsage to dict
            )
            output = RolloutOutput.from_success(final_output)

            return trajectory, output
        except Exception as e:
            trajectory = Trajectory(messages=messages, final_output=None, error=str(e))
            output = RolloutOutput.from_error(e)
            return trajectory, output

    def _run_simple(self, instance: DataInst) -> RolloutOutput:
        """Run the agent without capturing traces.

        Args:
            instance: The data instance to run.

        Returns:
            The rollout output.
        """
        try:
            result = self.agent.run_sync(
                instance.user_prompt.content,
                message_history=instance.message_history,
            )

            return RolloutOutput.from_success(result.output)
        except Exception as e:
            return RolloutOutput.from_error(e)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
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
        for trajectory, output, score in zip(eval_batch.trajectories, eval_batch.outputs, eval_batch.scores):
            record: dict[str, Any] = trajectory.to_reflective_record()

            # Add score and success information
            record['score'] = score
            record['success'] = output.success
            if output.error_message:
                record['error_message'] = output.error_message

            # Generate feedback based on score
            if score >= 0.8:
                feedback = 'Good response'
            elif score >= 0.5:
                feedback = 'Adequate response, could be improved'
            else:
                feedback = f'Poor response (score: {score:.2f})'
                if output.error_message:
                    feedback += f' - Error: {output.error_message}'

            record['feedback'] = feedback
            reflection_records.append(record)

        # Sample records if too many (keep it manageable for reflection)
        max_records = 10
        if len(reflection_records) > max_records:
            # Use deterministic sampling based on scores
            # Include both good and bad examples
            sorted_records: list[dict[str, Any]] = sorted(reflection_records, key=lambda r: r['score'])
            sampled: list[dict[str, Any]] = []
            # Take some low-scoring examples
            sampled.extend(sorted_records[: max_records // 3])
            # Take some high-scoring examples
            sampled.extend(sorted_records[-(max_records // 3) :])
            # Fill the rest randomly but deterministically
            remaining = max_records - len(sampled)
            if remaining > 0:
                middle: list[dict[str, Any]] = sorted_records[max_records // 3 : -(max_records // 3)]
                random.Random(42).shuffle(middle)
                sampled.extend(middle[:remaining])
            reflection_records = sampled

        # For v1, we provide the same dataset to all components
        # In v2, we could do component-specific attribution
        return {comp: reflection_records for comp in components_to_update}
