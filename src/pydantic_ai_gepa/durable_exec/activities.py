"""Temporal activities for GEPA optimization."""

from __future__ import annotations

from typing import Any

from temporalio import activity

from pydantic_ai_gepa.durable_exec.utils import ref_to_object
from pydantic_ai_gepa.gepa_graph.models import (
    CandidateMap,
    CandidateProgram,
    GepaConfig,
    EvaluationErrorEvent,
)
from pydantic_ai_gepa.types import Case, MetricResult, RolloutOutput


@activity.defn
async def load_dataset_ids(dataset_ref: str) -> list[str]:
    """Load all available IDs from the dataset."""
    dataset = ref_to_object(dataset_ref)
    # Determine how to extract IDs
    # If it's a sequence of Cases, use their names or indices
    if isinstance(dataset, (list, tuple)):
        return [
            getattr(item, "name", None) or f"case-{i}" for i, item in enumerate(dataset)
        ]
    # If it matches the DataLoader protocol
    if hasattr(dataset, "all_ids"):
        ids = await dataset.all_ids()
        return [str(id) for id in ids]

    raise ValueError(f"Unsupported dataset type: {type(dataset)}")


@activity.defn
async def load_dataset_batch(
    dataset_ref: str, indices: list[int]
) -> list[Case[Any, Any, Any]]:
    """Load a specific batch of cases by index."""
    dataset = ref_to_object(dataset_ref)

    # Sequence handling
    if isinstance(dataset, (list, tuple)):
        return [dataset[i] for i in indices]

    # DataLoader handling (fetch by ID if possible, but indices are requested here)
    # This assumes the workflow tracks indices 0..N matching the dataset order
    # This might be brittle for DataLoaders that aren't ordered sequences.
    # For now, assume sequence-like behavior.
    if hasattr(dataset, "fetch"):
        # If it's a DataLoader, we might need IDs, not indices.
        # But load_dataset_ids returns IDs.
        # Let's change the signature to accept IDs if we want to support arbitrary Loaders.
        # But for simplicity, let's stick to "Dataset is a Sequence" for this MVP.
        raise NotImplementedError(
            "DataLoader support not fully implemented in simple batch loader"
        )

    raise ValueError(f"Unsupported dataset type: {type(dataset)}")


@activity.defn
async def calculate_metric(
    metric_ref: str, case: Case[Any, Any, Any], output: RolloutOutput[Any]
) -> MetricResult:
    """Execute the metric function for a single case result."""
    metric_fn = ref_to_object(metric_ref)

    # Metric might be sync or async
    result = metric_fn(case, output)
    if hasattr(result, "__await__"):
        result = await result

    return result


@activity.defn
async def reflect_candidate(
    candidate: CandidateProgram,
    errors: list[EvaluationErrorEvent],
    config: GepaConfig,
    input_type_ref: str | None = None,
) -> list[CandidateMap]:
    """Generate new candidate proposals via reflection."""

    # Reuse the core GEPA logic
    # Note: generate_reflection_proposals expects a 'deps' object with a reflection model.
    # We need to instantiate the reflection model here or pass it in config.
    # GepaConfig has 'reflection_model' as a string or Model object.

    # If reflection_model is a string (e.g. "openai:gpt-4"), pydantic-ai handles it.
    # If it's an object, it must have survived serialization (unlikely for Temporal)
    # or be recreated.

    # For this activity, we assume config.reflection_model is a string or a simpler config
    # that generate_reflection_proposals can handle, OR we need to resolve it.

    # TODO: Adapter/Deps creation for reflection
    # This is complex because 'generate_reflection_proposals' is tied to the graph Deps.
    # We might need to inline the logic or refactor 'generate_reflection_proposals'
    # to be more standalone.

    # For MVP, placeholder:
    return []
