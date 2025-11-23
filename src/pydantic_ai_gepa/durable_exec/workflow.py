"""Temporal Workflow for GEPA optimization."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, cast

from temporalio import workflow
from pydantic_ai.durable_exec.temporal import TemporalAgent

# Import activities
from pydantic_ai_gepa.durable_exec.activities import (
    load_dataset_ids,
    load_dataset_batch,
    calculate_metric,
)
from pydantic_ai_gepa.durable_exec.utils import ref_to_object

from pydantic_ai_gepa.gepa_graph.models import CandidateMap, GepaConfig


@workflow.defn
class GepaOptimizationWorkflow:
    @workflow.run
    async def run(
        self,
        agent_ref: str,
        dataset_ref: str,
        metric_ref: str,
        config: GepaConfig,
        input_type_ref: str | None = None,
        seed_candidate: CandidateMap | None = None,
    ) -> CandidateMap:
        # 1. Hydrate the agent inside the workflow to get the TemporalAgent wrapper
        # This ensures we generate the same activity stubs as the Worker
        agent_instance = ref_to_object(agent_ref)
        # Note: We assume the agent name matches what the Worker registered.
        temporal_agent = TemporalAgent(agent_instance)

        # 2. Initialize State
        # In a real implementation, we'd use a full DurableGepaState.
        # For this skeleton, we track simple best_candidate.
        best_candidate: CandidateMap | None = None
        best_score = -1.0

        # Load dataset IDs once
        case_ids = await workflow.execute_activity(
            load_dataset_ids,
            args=[dataset_ref],
            start_to_close_timeout=timedelta(minutes=5),
        )

        # Main Optimization Loop
        for iteration in range(config.max_iterations or 10):
            workflow.logger.info(f"Starting iteration {iteration}")

            # Evaluation Loop
            # We process in batches to avoid huge histories, but here we iterate one by one
            # or small batches for simplicity.

            # Create a batch of indices
            indices = list(range(len(case_ids)))  # Simple full pass

            # We can't load all cases into workflow memory if large.
            # Load in chunks.
            chunk_size = 10
            total_score = 0.0

            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                cases = await workflow.execute_activity(
                    load_dataset_batch,
                    args=[dataset_ref, chunk_indices],
                    start_to_close_timeout=timedelta(minutes=1),
                )

                for case in cases:
                    # Run the agent (Durable Activity Call)
                    # We need to apply the candidate prompts.
                    # temporal_agent.override() context manager usage:

                    # TODO: Construct overrides from current_candidate_map
                    # This depends on how CandidateMap maps to agent components

                    # output = await temporal_agent.run(case.input)
                    # For now, assume agent runs as-is (Seed evaluation)

                    # Note: agent.run returns AgentRunResult
                    try:
                        # pyright doesn't know Case has `.input`; cast to Any for now.
                        result = await temporal_agent.run(cast(Any, case).input)
                    except Exception as e:
                        workflow.logger.error(f"Agent run failed: {e}")
                        continue

                    # Calculate Metric (Activity Call)
                    # result is serializable (Pydantic model)
                    metric_res = await workflow.execute_activity(
                        calculate_metric,
                        args=[metric_ref, case, result],
                        start_to_close_timeout=timedelta(minutes=1),
                    )

                    total_score += metric_res.score

            avg_score = total_score / len(indices) if indices else 0
            workflow.logger.info(f"Iteration {iteration} Score: {avg_score}")

            if avg_score > best_score:
                best_score = avg_score
                # best_candidate = current_candidate

            # Reflection Step
            # new_candidates = await workflow.execute_activity(reflect_candidate, ...)

            # Update population...

        return best_candidate or {}
