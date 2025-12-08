"""Tools for the student agent to use during execution."""

from __future__ import annotations

from pydantic_ai import FunctionToolset

from ..example_bank import InMemoryExampleBank


def create_example_search_tool(
    bank: InMemoryExampleBank,
    instruction: str,
    k: int = 3,
) -> FunctionToolset:
    """Create the example search tool for the student agent.

    This tool allows the student agent to search for relevant few-shot
    examples during execution.

    Args:
        bank: The example bank to search.
        instruction: Description of when to use this tool.
        k: Number of examples to retrieve.
    """
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool(description=instruction)
    def search_examples(query: str) -> str:
        """Search for relevant examples to guide your response.

        Args:
            query: What kind of example are you looking for?
        """
        results = bank.search(query, k=k)
        if not results:
            return "No relevant examples found."

        formatted = []
        for ex in results:
            formatted.append(f"### {ex.title}\n{ex.content}")
        return "\n\n---\n\n".join(formatted)

    return toolset


__all__ = ["create_example_search_tool"]
