"""Shared helpers for GEPA graph nodes."""

from __future__ import annotations

from typing import TypeAlias

from pydantic_graph import BaseNode, End, GraphRunContext

from ..models import GepaResult, GepaState
from ..deps import GepaDeps

GepaRunContext: TypeAlias = GraphRunContext[GepaState, GepaDeps]


class GepaNode(BaseNode[GepaState, GepaDeps, None]):
    """Base class for GEPA graph nodes."""


__all__ = ["End", "GepaNode", "GepaRunContext", "GepaResult"]
