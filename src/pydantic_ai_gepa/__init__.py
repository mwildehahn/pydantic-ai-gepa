"""GEPA optimization integration for pydantic-ai."""

from __future__ import annotations

from .adapter import Adapter, AdapterTrajectory, AgentAdapter
from .reflection import ReflectionSampler
from .cache import CacheManager, create_cached_metric
from .inspection import (
    InspectingModel,
    InspectionAborted,
    InspectionSnapshot,
)
from .runner import GepaOptimizationResult, optimize_agent
from .signature import (
    BoundInputSpec,
    InputSpec,
    SignatureSuffix,
    apply_candidate_to_input_model,
    build_input_spec,
    generate_system_instructions,
    generate_user_content,
    get_gepa_components,
)
from .signature_agent import SignatureAgent
from .types import DataInst, MetricResult, OutputT, RolloutOutput, Trajectory

__all__ = [
    "optimize_agent",
    "GepaOptimizationResult",
    "Adapter",
    "AgentAdapter",
    "ReflectionSampler",
    "CacheManager",
    "create_cached_metric",
    "DataInst",
    "AdapterTrajectory",
    "Trajectory",
    "RolloutOutput",
    "MetricResult",
    "OutputT",
    "BoundInputSpec",
    "InputSpec",
    "generate_system_instructions",
    "generate_user_content",
    "get_gepa_components",
    "apply_candidate_to_input_model",
    "build_input_spec",
    "SignatureSuffix",
    "SignatureAgent",
    "InspectingModel",
    "InspectionAborted",
    "InspectionSnapshot",
]

__version__ = "0.1.0"
