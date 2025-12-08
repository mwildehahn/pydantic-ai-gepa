"""Tools for the reflection agent to manage the example bank."""

from __future__ import annotations

from pydantic_ai import FunctionToolset

from ..example_bank import BankedExample, InMemoryExampleBank


def create_example_bank_tools(bank: InMemoryExampleBank) -> FunctionToolset:
    """Create tools for the reflection agent to manage examples.

    These tools allow the reflection agent to iteratively curate the example
    bank by adding, removing, and testing examples.
    """
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool
    def add_example(title: str, keywords: list[str], content: str) -> str:
        """Add a few-shot example to help the student agent handle similar cases.

        Use this when you identify a pattern in the failures that could be
        addressed by showing the student a concrete example of correct behavior.

        Args:
            title: Short descriptive name for the example (e.g., "Handle nested JSON").
            keywords: Semantic keywords for retrieval - what user queries should find
                this example? (e.g., ["json", "nested", "parsing"]).
            content: The full example content. This can be any format that helps:
                input/output pairs, reasoning chains, do/don't comparisons, etc.
        """
        example = BankedExample(title=title, keywords=keywords, content=content)
        bank.add(example)
        return f"Added example '{title}' (id: {example.id})"

    @toolset.tool
    def remove_example(example_id: str) -> str:
        """Remove an example that isn't helping or is causing issues.

        Use this when an example is:
        - Redundant with another example
        - Causing confusion or incorrect behavior
        - No longer relevant after prompt changes
        """
        if bank.remove(example_id):
            return f"Removed example {example_id}"
        return f"Example {example_id} not found"

    @toolset.tool
    def list_examples() -> str:
        """View all current examples in the bank.

        Use this to understand what examples already exist before adding
        new ones, or to identify examples that should be removed.
        """
        if len(bank) == 0:
            return "Example bank is empty."
        lines = ["Current examples in bank:"]
        for ex in bank:
            kw_str = ", ".join(ex.keywords) if ex.keywords else "(no keywords)"
            lines.append(f"- [{ex.id}] {ex.title}")
            lines.append(f"  Keywords: {kw_str}")
            # Show truncated content preview
            preview = ex.content[:100] + "..." if len(ex.content) > 100 else ex.content
            preview = preview.replace("\n", " ")
            lines.append(f"  Preview: {preview}")
        return "\n".join(lines)

    @toolset.tool
    def test_retrieval(query: str) -> str:
        """Test what examples would be retrieved for a given query.

        Use this to verify that your keywords are working correctly and
        that the right examples would surface for expected user inputs.
        """
        results = bank.search(query, k=3)
        if not results:
            return f"No examples would be retrieved for query: '{query}'"
        lines = [f"Query '{query}' would retrieve:"]
        for i, ex in enumerate(results, 1):
            lines.append(f"{i}. {ex.title} (id: {ex.id})")
        return "\n".join(lines)

    return toolset


__all__ = ["create_example_bank_tools"]
