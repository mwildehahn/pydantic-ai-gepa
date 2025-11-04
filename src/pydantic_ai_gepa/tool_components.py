"""Utilities for exposing and applying tool components to GEPA."""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from threading import Thread
from typing import Any, Iterable

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

ToolCandidate = dict[str, str]


def _run_coro_sync(coro: asyncio.Future[Any] | asyncio.coroutines.Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine in sync context, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = Thread(target=_runner)
    thread.start()
    thread.join()

    if error:
        raise error[0]
    return result[0] if result else None


def _unwrap_agent(agent: AbstractAgent[Any, Any]) -> AbstractAgent[Any, Any]:
    """Return the innermost agent (unwrap WrapperAgent layers)."""
    current = agent
    while isinstance(current, WrapperAgent):
        current = current.wrapped
    return current


def _build_run_context(agent: AbstractAgent[Any, Any]) -> RunContext[Any]:
    """Create a minimal RunContext for tool preparation."""
    model = getattr(agent, "model", None)
    if not isinstance(model, Model):
        model_instance: Model = TestModel()
    else:
        model_instance = model

    return RunContext(deps=None, model=model_instance, usage=RunUsage())


def _description_key(tool_name: str) -> str:
    return f"tool:{tool_name}:description"


def _format_path(path: tuple[str, ...]) -> str:
    formatted: list[str] = []
    for segment in path:
        if segment == "[]":
            if not formatted:
                formatted.append("[]")
            else:
                formatted[-1] = f"{formatted[-1]}[]"
        else:
            formatted.append(segment)
    return ".".join(formatted)


def _parameter_key(tool_name: str, path: tuple[str, ...]) -> str:
    return f"tool:{tool_name}:param:{_format_path(path)}"


def _iter_schema_descriptions(schema: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str]]:
    """Yield (path, description) pairs for a tool parameter schema."""
    if not isinstance(schema, dict):
        return

    description = schema.get("description")
    if isinstance(description, str) and description.strip() and path:
        yield path, description

    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties") or {}
        if isinstance(properties, dict):
            for name, subschema in properties.items():
                if isinstance(subschema, dict):
                    yield from _iter_schema_descriptions(subschema, path + (name,))
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            yield from _iter_schema_descriptions(items, path + ("[]",))


def _set_schema_description(schema: dict[str, Any], path: tuple[str, ...], value: str) -> bool:
    """Set a description on a copied schema, returning True if modified."""
    target: dict[str, Any] | None = schema
    for segment in path:
        if target is None:
            return False
        if segment == "[]":
            next_target = target.get("items")
            if not isinstance(next_target, dict):
                return False
            target = next_target
        else:
            properties = target.get("properties")
            if not isinstance(properties, dict):
                return False
            next_target = properties.get(segment)
            if not isinstance(next_target, dict):
                return False
            target = next_target

    if target is None:
        return False

    current = target.get("description")
    if current == value:
        return False
    target["description"] = value
    return True


@dataclass
class ToolComponentInfo:
    """Component metadata for a single tool."""

    description_key: str | None
    parameter_paths: dict[str, tuple[str, ...]]


def _extract_components(tool_defs: Iterable[ToolDefinition]) -> tuple[dict[str, str], dict[str, ToolComponentInfo]]:
    """Extract component seeds and metadata from tool definitions."""
    seed_components: dict[str, str] = {}
    component_info: dict[str, ToolComponentInfo] = {}

    for tool_def in tool_defs:
        description_key: str | None = None
        parameter_paths: dict[str, tuple[str, ...]] = {}

        if isinstance(tool_def.description, str) and tool_def.description.strip():
            description_key = _description_key(tool_def.name)
            seed_components[description_key] = tool_def.description

        for path, desc in _iter_schema_descriptions(tool_def.parameters_json_schema):
            key = _parameter_key(tool_def.name, path)
            seed_components[key] = desc
            parameter_paths[key] = path

        component_info[tool_def.name] = ToolComponentInfo(
            description_key=description_key,
            parameter_paths=parameter_paths,
        )

    return seed_components, component_info


class ToolOptimizationManager:
    """Manage tool component extraction and candidate application."""

    def __init__(self, agent: AbstractAgent[Any, Any]) -> None:
        self._base_agent = _unwrap_agent(agent)
        self._base_prepare = getattr(self._base_agent, "_prepare_tools", None)
        self._candidate_var: contextvars.ContextVar[ToolCandidate | None] = contextvars.ContextVar(
            "gepa_tool_candidate", default=None
        )
        self._tool_components: dict[str, ToolComponentInfo] = {}
        self._seed_components: dict[str, str] | None = None

        # Install wrapper only once.
        if getattr(self._base_agent, "_gepa_tool_prepare_wrapper", None) is None:
            setattr(self._base_agent, "_gepa_tool_prepare_wrapper", self._prepare_wrapper)
            self._base_agent._prepare_tools = self._prepare_wrapper  # type: ignore[assignment]

    def get_seed_components(self) -> dict[str, str]:
        """Return cached seed components, collecting them if necessary."""
        if self._seed_components is not None:
            return dict(self._seed_components)

        try:
            tool_defs = _run_coro_sync(self._fetch_tool_definitions()) or []
        except Exception:
            return {}

        if not tool_defs:
            return {}

        seed_components, component_info = _extract_components(tool_defs)
        self._seed_components = dict(seed_components)
        self._tool_components = component_info
        return dict(seed_components)

    def get_component_keys(self) -> list[str]:
        """Return all known component keys."""
        components = self.get_seed_components()
        return list(components.keys())

    @contextmanager
    def candidate_context(self, candidate: dict[str, str] | None) -> Iterable[None]:
        """Context manager to apply a candidate during tool preparation."""
        filtered = self._filter_candidate(candidate)
        token = self._candidate_var.set(filtered)
        try:
            yield
        finally:
            self._candidate_var.reset(token)

    async def _fetch_tool_definitions(self) -> list[ToolDefinition]:
        toolset = self._base_agent._get_toolset()  # type: ignore[attr-defined]
        ctx = _build_run_context(self._base_agent)

        async with toolset:
            tools_dict = await toolset.get_tools(ctx)
        return [tool.tool_def for tool in tools_dict.values()]

    def _filter_candidate(self, candidate: dict[str, str] | None) -> ToolCandidate | None:
        if not candidate:
            return None
        filtered = {key: value for key, value in candidate.items() if key.startswith("tool:")}
        return filtered or None

    async def _prepare_wrapper(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition] | None:
        prepared = tool_defs
        if self._base_prepare:
            prepared_result = await self._base_prepare(ctx, tool_defs)
            if prepared_result is None:
                return None
            prepared = prepared_result

        seed_components, component_info = _extract_components(prepared)
        if self._seed_components is None:
            self._seed_components = dict(seed_components)
        else:
            for key, value in seed_components.items():
                self._seed_components.setdefault(key, value)
        self._tool_components = component_info

        candidate = self._candidate_var.get()
        if not candidate:
            return prepared

        modified: list[ToolDefinition] = []
        changed = False
        for tool_def in prepared:
            new_def = self._apply_candidate_to_tool(tool_def, candidate)
            if new_def is not tool_def:
                changed = True
            modified.append(new_def)

        return modified if changed else prepared

    def _apply_candidate_to_tool(
        self, tool_def: ToolDefinition, candidate: ToolCandidate
    ) -> ToolDefinition:
        info = self._tool_components.get(tool_def.name)
        if not info:
            return tool_def

        updates: dict[str, Any] = {}

        if info.description_key:
            raw_value = candidate.get(info.description_key)
            if raw_value is not None:
                new_description = str(raw_value)
                if tool_def.description != new_description:
                    updates["description"] = new_description

        schema_copy: dict[str, Any] | None = None
        schema_changed = False

        for key, path in info.parameter_paths.items():
            if key not in candidate:
                continue
            raw_value = candidate[key]
            if raw_value is None:
                continue
            new_description = str(raw_value)
            if schema_copy is None:
                schema_copy = deepcopy(tool_def.parameters_json_schema)
            if _set_schema_description(schema_copy, path, new_description):
                schema_changed = True

        if schema_changed and schema_copy is not None:
            updates["parameters_json_schema"] = schema_copy

        if updates:
            return replace(tool_def, **updates)
        return tool_def


def get_tool_optimizer(agent: AbstractAgent[Any, Any]) -> ToolOptimizationManager | None:
    """Return the installed tool optimization manager for an agent, if any."""
    base_agent = _unwrap_agent(agent)
    manager = getattr(base_agent, "_gepa_tool_optimizer", None)
    if isinstance(manager, ToolOptimizationManager):
        return manager
    return None


def get_or_create_tool_optimizer(agent: AbstractAgent[Any, Any]) -> ToolOptimizationManager:
    """Retrieve or attach a tool optimization manager to an agent."""
    base_agent = _unwrap_agent(agent)
    manager = getattr(base_agent, "_gepa_tool_optimizer", None)
    if isinstance(manager, ToolOptimizationManager):
        return manager

    manager = ToolOptimizationManager(base_agent)
    setattr(base_agent, "_gepa_tool_optimizer", manager)
    return manager

