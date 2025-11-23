"""Client-side runner for Temporal GEPA optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import uuid

from pydantic_ai.agent import AbstractAgent
from pydantic_evals import Case

from pydantic_ai_gepa.gepa_graph.models import CandidateMap, GepaConfig
from pydantic_ai_gepa.gepa_graph.datasets import DatasetInput
from pydantic_ai_gepa.input_type import InputSpec
from pydantic_ai_gepa.durable_exec.utils import ensure_ref
from pydantic_ai_gepa.types import MetricResult, RolloutOutput


@dataclass
class TemporalRunner:
    """Configuration for running GEPA optimization on Temporal."""

    client: Any | None = None
    """Existing Temporal Client instance. If None, attempts to connect to localhost:7233."""

    task_queue: str = "gepa-optimization"
    """The task queue to submit the workflow to."""


async def run_temporal_optimization(
    *,
    agent: AbstractAgent[Any, Any],
    trainset: DatasetInput,
    metric: Callable[[Case[Any, Any, Any], RolloutOutput[Any]], MetricResult],
    config: GepaConfig,
    input_type: InputSpec[Any] | None = None,
    seed_candidate: CandidateMap | None = None,
    temporal_client: Any | None = None,
    task_queue: str = "gepa-optimization",
) -> CandidateMap:
    """Execute the GEPA optimization workflow on Temporal.

    All passed objects must be importable at worker side; they are converted to
    string refs before the workflow starts.
    """

    try:
        from temporalio.client import Client
    except ImportError:
        raise ImportError(
            "temporalio is required for Temporal execution. Install it with 'pip install temporalio'."
        )

    # Convert to importable refs early to fail fast if something isn't importable.
    agent_ref = ensure_ref(agent)
    trainset_ref = ensure_ref(trainset)
    metric_ref = ensure_ref(metric)
    input_type_ref = ensure_ref(input_type) if input_type else None

    if temporal_client is None:
        temporal_client = await Client.connect("localhost:7233")

    from .workflow import GepaOptimizationWorkflow

    workflow_id = f"gepa-opt-{uuid.uuid4()}"

    handle = await temporal_client.start_workflow(
        GepaOptimizationWorkflow.run,
        args=[
            agent_ref,
            trainset_ref,
            metric_ref,
            config,
            input_type_ref,
            seed_candidate,
        ],
        id=workflow_id,
        task_queue=task_queue,
    )

    print(f"Started Temporal Workflow: {workflow_id}")
    return await handle.result()
