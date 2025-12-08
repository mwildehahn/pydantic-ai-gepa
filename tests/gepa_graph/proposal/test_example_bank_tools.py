"""Tests for the example bank tools used by the reflection agent."""

from __future__ import annotations

from typing import Any, Callable

from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank, BankedExample
from pydantic_ai_gepa.gepa_graph.proposal.example_bank_tools import (
    create_example_bank_tools,
)


def _get_tool_fn(toolset: Any, name: str) -> Callable[..., Any]:
    """Extract the raw function from a tool, typed loosely for test convenience."""
    return toolset.tools[name].function  # type: ignore[no-any-return]


class TestExampleBankTools:
    def test_creates_toolset(self) -> None:
        """Verify that create_example_bank_tools returns a valid toolset."""
        bank = InMemoryExampleBank()
        toolset = create_example_bank_tools(bank)

        # Toolset should have registered tools
        assert len(toolset.tools) == 4
        tool_names = set(toolset.tools.keys())
        assert tool_names == {
            "add_example",
            "remove_example",
            "list_examples",
            "test_retrieval",
        }

    def test_add_example_modifies_bank(self) -> None:
        """Test that add_example tool modifies the bank correctly."""
        bank = InMemoryExampleBank()
        toolset = create_example_bank_tools(bank)

        # Get the underlying function and call it
        add_fn = _get_tool_fn(toolset, "add_example")
        result = add_fn(
            title="Test Example", keywords=["test", "example"], content="Test content"
        )

        assert "Added example 'Test Example'" in result
        assert len(bank) == 1
        assert bank.search("test")[0].title == "Test Example"

    def test_remove_example_modifies_bank(self) -> None:
        """Test that remove_example tool modifies the bank correctly."""
        bank = InMemoryExampleBank()
        ex = BankedExample(id="test-id", title="Test", keywords=[], content="")
        bank.add(ex)

        toolset = create_example_bank_tools(bank)
        remove_fn = _get_tool_fn(toolset, "remove_example")
        result = remove_fn(example_id="test-id")

        assert "Removed example test-id" in result
        assert len(bank) == 0

    def test_remove_example_not_found(self) -> None:
        """Test remove_example when ID doesn't exist."""
        bank = InMemoryExampleBank()
        toolset = create_example_bank_tools(bank)

        remove_fn = _get_tool_fn(toolset, "remove_example")
        result = remove_fn(example_id="nonexistent")
        assert "not found" in result

    def test_list_examples_empty(self) -> None:
        """Test list_examples on empty bank."""
        bank = InMemoryExampleBank()
        toolset = create_example_bank_tools(bank)

        list_fn = _get_tool_fn(toolset, "list_examples")
        result = list_fn()
        assert "empty" in result.lower()

    def test_list_examples_with_content(self) -> None:
        """Test list_examples shows all examples."""
        bank = InMemoryExampleBank()
        bank.add(
            BankedExample(
                id="ex1", title="First", keywords=["a", "b"], content="Content 1"
            )
        )
        bank.add(
            BankedExample(id="ex2", title="Second", keywords=["c"], content="Content 2")
        )

        toolset = create_example_bank_tools(bank)
        list_fn = _get_tool_fn(toolset, "list_examples")
        result = list_fn()

        assert "[ex1] First" in result
        assert "[ex2] Second" in result
        assert "a, b" in result

    def test_test_retrieval(self) -> None:
        """Test test_retrieval shows what would be retrieved."""
        bank = InMemoryExampleBank()
        bank.add(
            BankedExample(
                title="JSON Parsing", keywords=["json", "parsing"], content="..."
            )
        )
        bank.add(
            BankedExample(
                title="Error Handling", keywords=["error", "exception"], content="..."
            )
        )

        toolset = create_example_bank_tools(bank)
        test_fn = _get_tool_fn(toolset, "test_retrieval")

        result = test_fn(query="json")
        assert "JSON Parsing" in result

        result = test_fn(query="error")
        assert "Error Handling" in result

    def test_test_retrieval_no_results(self) -> None:
        """Test test_retrieval on empty bank."""
        bank = InMemoryExampleBank()
        toolset = create_example_bank_tools(bank)

        test_fn = _get_tool_fn(toolset, "test_retrieval")
        result = test_fn(query="anything")
        assert "No examples" in result
