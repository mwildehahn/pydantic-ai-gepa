"""GEPA optimization integration for pydantic-ai."""

from __future__ import annotations

from .adapter import PydanticAIGEPAAdapter, ReflectionSampler
from .runner import GepaOptimizationResult, optimize_agent_prompts
from .signature import Signature
from .signature_agent import SignatureAgent
from .types import DataInst, RolloutOutput, Trajectory

__all__ = [
    'optimize_agent_prompts',
    'GepaOptimizationResult',
    'PydanticAIGEPAAdapter',
    'ReflectionSampler',
    'DataInst',
    'Trajectory',
    'RolloutOutput',
    'Signature',
    'SignatureAgent',
]

__version__ = '0.1.0'
