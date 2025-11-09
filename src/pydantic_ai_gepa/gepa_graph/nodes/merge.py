"""Placeholder MergeNode implementation.

The full merge workflow lands in Step 8 of the refactor plan.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import GepaNode, GepaRunContext


@dataclass(slots=True)
class MergeNode(GepaNode):
    """Stub node that will be replaced with the full merge algorithm."""

    async def run(self, ctx: GepaRunContext):
        raise NotImplementedError("MergeNode will be implemented in Step 8.")


__all__ = ["MergeNode"]
