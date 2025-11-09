"""Placeholder ReflectNode implementation.

The full reflection workflow lands in Step 6 of the refactor plan.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import GepaNode, GepaRunContext


@dataclass(slots=True)
class ReflectNode(GepaNode):
    """Stub node that will be replaced with the full reflection algorithm."""

    async def run(self, ctx: GepaRunContext):
        raise NotImplementedError("ReflectNode will be implemented in Step 6.")


__all__ = ["ReflectNode"]
