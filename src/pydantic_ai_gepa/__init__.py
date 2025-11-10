"""GEPA optimization integration for pydantic-ai."""

from __future__ import annotations

from .adapter import AgentAdapter, ReflectionSampler
from .cache import CacheManager, create_cached_metric
from .openai_inspection import (
    InspectingOpenAIModel,
    OpenAIInspectionAborted,
    OpenAIInspectionSnapshot,
)
from .runner import GepaOptimizationResult, optimize_agent_prompts
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
from .types import DataInst, RolloutOutput, Trajectory, OutputT

__all__ = [
    "optimize_agent_prompts",
    "GepaOptimizationResult",
    "AgentAdapter",
    "ReflectionSampler",
    "CacheManager",
    "create_cached_metric",
    "DataInst",
    "Trajectory",
    "RolloutOutput",
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
    "InspectingOpenAIModel",
    "OpenAIInspectionAborted",
    "OpenAIInspectionSnapshot",
]

__version__ = "0.1.0"
