"""GEPA optimization integration for pydantic-ai."""

from __future__ import annotations

from .adapter import Adapter
from .adapters.agent_adapter import (
    AgentAdapter,
    AgentAdapterTrajectory,
    SignatureAgentAdapter,
    create_adapter,
)
from .reflection import ReflectionSampler
from .cache import CacheManager, create_cached_metric
from .inspection import (
    InspectingModel,
    InspectionAborted,
    InspectionSnapshot,
)
from .exceptions import UsageBudgetExceeded
from .runner import GepaOptimizationResult, optimize_agent
from .input_type import (
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
from .types import (
    Case,
    ExampleBankConfig,
    MetadataWithMessageHistory,
    MetricResult,
    OutputT,
    ReflectionConfig,
    RolloutOutput,
    Trajectory,
)

__all__ = [
    "optimize_agent",
    "GepaOptimizationResult",
    "Adapter",
    "AgentAdapter",
    "SignatureAgentAdapter",
    "ReflectionSampler",
    "CacheManager",
    "create_cached_metric",
    "Case",
    "create_adapter",
    "AgentAdapterTrajectory",
    "Trajectory",
    "RolloutOutput",
    "MetricResult",
    "MetadataWithMessageHistory",
    "ExampleBankConfig",
    "ReflectionConfig",
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
    "UsageBudgetExceeded",
]

__version__ = "0.1.0"
