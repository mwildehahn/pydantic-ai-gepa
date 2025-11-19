"""Tests for tool component extraction and application."""

from __future__ import annotations

from typing import Any, cast

from pydantic_ai.agent import AbstractAgent
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_gepa.tool_components import (
    ToolComponentCatalog,
    ToolOptimizationManager,
)


def _make_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="format_text",
        description="Format content for downstream processing.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Raw text to format.",
                },
                "style": {
                    "type": "string",
                    "description": "Formatting instructions to apply.",
                },
            },
            "required": ["text", "style"],
            "additionalProperties": False,
        },
    )


class _DummyAgent:
    """Minimal agent stub for ToolOptimizationManager tests."""

    def __init__(self) -> None:
        self._prepare_tools = None
        self.model = None


def test_catalog_ingest_records_seed_components() -> None:
    catalog = ToolComponentCatalog()
    tool_def = _make_tool_definition()

    catalog.ingest([tool_def])
    seeds = catalog.seed_snapshot()

    assert (
        seeds["tool:format_text:description"]
        == "Format content for downstream processing."
    )
    assert seeds["tool:format_text:param:text"] == "Raw text to format."
    assert seeds["tool:format_text:param:style"] == "Formatting instructions to apply."
    assert catalog.metadata_for("format_text") is not None


def test_apply_candidate_updates_tool_definition() -> None:
    manager = ToolOptimizationManager(cast(AbstractAgent[Any, Any], _DummyAgent()))
    tool_def = _make_tool_definition()
    manager._catalog.ingest([tool_def])  # type: ignore[attr-defined]

    candidate = {
        "tool:format_text:description": "Polish the incoming copy.",
        "tool:format_text:param:text": "Draft prose needing polish.",
    }
    updated = manager._apply_candidate_to_tool(tool_def, candidate)  # type: ignore[attr-defined]

    assert updated.description == "Polish the incoming copy."
    params = updated.parameters_json_schema["properties"]
    assert params["text"]["description"] == "Draft prose needing polish."
    # Keys that were not overridden remain intact.
    assert params["style"]["description"] == "Formatting instructions to apply."


def test_record_model_request_tracks_builtin_tools() -> None:
    manager = ToolOptimizationManager(cast(AbstractAgent[Any, Any], _DummyAgent()))
    builtin = WebSearchTool()

    manager.record_model_request(
        function_tools=[_make_tool_definition()], builtin_tools=[builtin]
    )

    latest = manager.latest_builtin_tools()
    assert len(latest) == 1
    assert latest[0].kind == "web_search"
