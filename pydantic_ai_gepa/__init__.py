"""GEPA optimization integration for pydantic-ai."""

from __future__ import annotations

from .adapter import PydanticAIGEPAAdapter
from .runner import GepaOptimizationResult, optimize_agent_prompts
from .types import DataInst, RolloutOutput, Trajectory

__all__ = [
    'optimize_agent_prompts',
    'GepaOptimizationResult',
    'PydanticAIGEPAAdapter',
    'DataInst',
    'Trajectory',
    'RolloutOutput',
]

__version__ = '0.1.0'
