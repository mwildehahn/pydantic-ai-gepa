"""GEPA optimization integration for pydantic-ai."""

from __future__ import annotations

from .adapter import PydanticAIGEPAAdapter, ReflectionSampler
from .cache import CacheManager, create_cached_metric
from .runner import GepaOptimizationResult, optimize_agent_prompts
from .signature import (
    SignatureSuffix,
    apply_candidate_to_input_model,
    generate_system_instructions,
    generate_user_content,
    get_gepa_components,
)
from .signature_agent import SignatureAgent
from .types import DataInst, RolloutOutput, Trajectory, OutputT

__all__ = [
    "optimize_agent_prompts",
    "GepaOptimizationResult",
    "PydanticAIGEPAAdapter",
    "ReflectionSampler",
    "CacheManager",
    "create_cached_metric",
    "DataInst",
    "Trajectory",
    "RolloutOutput",
    "OutputT",
    "generate_system_instructions",
    "generate_user_content",
    "get_gepa_components",
    "apply_candidate_to_input_model",
    "SignatureSuffix",
    "SignatureAgent",
]

__version__ = "0.1.0"
