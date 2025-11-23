"""Utilities for resolving object references for durable execution."""

from __future__ import annotations

import importlib
from typing import Any


def object_to_ref(obj: Any) -> str:
    """Convert a top-level object to an importable string reference.

    Format: 'module.path:qualname'
    """
    if not hasattr(obj, "__module__") or not hasattr(obj, "__qualname__"):
        raise ValueError(f"Object {obj!r} must have __module__ and __qualname__")

    if obj.__module__ == "__main__":
        raise ValueError(
            f"Object {obj!r} is defined in __main__ and cannot be imported by a worker. "
            "Please move it to a dedicated module."
        )

    return f"{obj.__module__}:{obj.__qualname__}"


def ref_to_object(ref: str) -> Any:
    """Resolve an importable string reference back to an object."""
    try:
        module_path, qualname = ref.split(":")
    except ValueError as e:
        raise ValueError(
            f"Invalid reference format '{ref}'. Expected 'module:qualname'"
        ) from e

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}'") from e

    try:
        obj = module
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    except AttributeError as e:
        raise AttributeError(
            f"Could not find '{qualname}' in module '{module_path}'"
        ) from e


def ensure_ref(obj_or_ref: Any | str) -> str:
    """Ensure the input is a string reference, converting if necessary."""
    if isinstance(obj_or_ref, str):
        return obj_or_ref
    return object_to_ref(obj_or_ref)
