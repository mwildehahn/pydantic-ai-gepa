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


class TestSelectiveToolOptimization:
    """Tests for selective tool optimization via allowed_tools."""

    def test_filter_candidate_with_allowed_tools_filters_by_name(self) -> None:
        """When allowed_tools is set, only those tools are included."""
        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools={"format_text"},
        )

        candidate = {
            "tool:format_text:description": "Allowed tool description",
            "tool:other_tool:description": "Should be filtered out",
            "tool:format_text:param:text": "Allowed param",
            "tool:other_tool:param:foo": "Should be filtered out",
            "instructions": "Not a tool key",
        }

        filtered = manager._filter_candidate(candidate)

        assert filtered is not None
        assert "tool:format_text:description" in filtered
        assert "tool:format_text:param:text" in filtered
        assert "tool:other_tool:description" not in filtered
        assert "tool:other_tool:param:foo" not in filtered
        assert "instructions" not in filtered

    def test_filter_candidate_without_allowed_tools_includes_all(self) -> None:
        """When allowed_tools is None, all tool keys are included."""
        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools=None,
        )

        candidate = {
            "tool:format_text:description": "First tool",
            "tool:other_tool:description": "Second tool",
            "instructions": "Not a tool key",
        }

        filtered = manager._filter_candidate(candidate)

        assert filtered is not None
        assert "tool:format_text:description" in filtered
        assert "tool:other_tool:description" in filtered
        assert "instructions" not in filtered

    def test_allow_tool_adds_to_existing_set(self) -> None:
        """allow_tool() adds a tool name to the allowed set."""
        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools={"existing_tool"},
        )

        manager.allow_tool("new_tool")

        assert manager._allowed_tools == {"existing_tool", "new_tool"}

    def test_allow_tool_creates_set_when_none(self) -> None:
        """allow_tool() creates a new set when allowed_tools was None."""
        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools=None,
        )

        manager.allow_tool("search_examples")

        assert manager._allowed_tools == {"search_examples"}

    def test_apply_candidate_respects_allowed_tools(self) -> None:
        """Candidate application only affects allowed tools."""
        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools={"format_text"},
        )

        # Create two tools
        format_tool = _make_tool_definition()
        other_tool = ToolDefinition(
            name="other_tool",
            description="Original other description",
            parameters_json_schema={"type": "object", "properties": {}},
        )

        manager._catalog.ingest([format_tool, other_tool])

        # Candidate tries to update both
        candidate = {
            "tool:format_text:description": "Updated format description",
            "tool:other_tool:description": "Updated other description",
        }

        # Filter the candidate (simulating what happens in candidate_context)
        filtered = manager._filter_candidate(candidate)
        assert filtered is not None

        # Apply to format_text - should update
        updated_format = manager._apply_candidate_to_tool(format_tool, filtered)
        assert updated_format.description == "Updated format description"

        # Apply to other_tool - should NOT update (not in filtered candidate)
        updated_other = manager._apply_candidate_to_tool(other_tool, filtered)
        assert updated_other.description == "Original other description"


def test_get_or_create_adds_allowed_tools_to_existing_manager() -> None:
    """get_or_create_tool_optimizer adds to existing manager's allowed tools."""
    from pydantic_ai_gepa.tool_components import (
        get_or_create_tool_optimizer,
    )

    agent = cast(AbstractAgent[Any, Any], _DummyAgent())

    # Create initial manager with one allowed tool
    manager1 = get_or_create_tool_optimizer(agent, allowed_tools={"tool_a"})
    assert manager1._allowed_tools == {"tool_a"}

    # Call again with additional tools - should add to existing
    manager2 = get_or_create_tool_optimizer(agent, allowed_tools={"tool_b", "tool_c"})

    assert manager1 is manager2  # Same instance
    assert manager2._allowed_tools == {"tool_a", "tool_b", "tool_c"}


class TestSearchExamplesToolOptimization:
    """Tests for search_examples tool component optimization."""

    def test_search_examples_components_registered_via_record_model_request(
        self,
    ) -> None:
        """Tool components are available after record_model_request."""
        from pydantic_ai_gepa.gepa_graph.proposal.student_tools import (
            create_example_search_tool,
        )
        from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank
        from pydantic_ai_gepa.types import ExampleBankConfig

        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools={"search_examples"},
        )

        # Create the example bank and toolset
        config = ExampleBankConfig(search_tool_instruction="Find similar examples")
        bank = InMemoryExampleBank(config=config)
        toolset = create_example_search_tool(
            bank=bank,
            instruction=bank.search_tool_instruction,
            k=3,
        )

        # Extract ToolDefinition and register it
        tool = toolset.tools["search_examples"]
        tool_def = ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.function_schema.json_schema,
        )
        manager.record_model_request(function_tools=[tool_def])

        # Verify components are available
        seed_components = manager.get_seed_components()
        assert "tool:search_examples:description" in seed_components
        assert (
            seed_components["tool:search_examples:description"]
            == "Find similar examples"
        )
        assert "tool:search_examples:param:query" in seed_components

    def test_search_examples_description_updated_from_candidate(self) -> None:
        """Optimized description from candidate is applied to tool."""
        from pydantic_ai_gepa.gepa_graph.proposal.student_tools import (
            create_example_search_tool,
        )
        from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank
        from pydantic_ai_gepa.types import ExampleBankConfig

        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools={"search_examples"},
        )

        # Create and register the tool
        config = ExampleBankConfig(search_tool_instruction="Original description")
        bank = InMemoryExampleBank(config=config)
        toolset = create_example_search_tool(
            bank=bank,
            instruction=bank.search_tool_instruction,
            k=3,
        )
        tool = toolset.tools["search_examples"]
        tool_def = ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.function_schema.json_schema,
        )
        manager.record_model_request(function_tools=[tool_def])

        # Candidate with optimized description
        candidate = {
            "tool:search_examples:description": "Optimized: Search for relevant examples to guide your response",
            "tool:search_examples:param:query": "Describe the type of example you need",
        }

        # Filter and apply the candidate
        filtered = manager._filter_candidate(candidate)
        assert filtered is not None

        updated_tool = manager._apply_candidate_to_tool(tool_def, filtered)

        # Verify the description was updated
        assert (
            updated_tool.description
            == "Optimized: Search for relevant examples to guide your response"
        )
        # Verify param description was updated
        assert (
            updated_tool.parameters_json_schema["properties"]["query"]["description"]
            == "Describe the type of example you need"
        )
