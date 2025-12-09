"""Tools for the reflection agent to manage the example bank."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import FunctionToolset

from ..example_bank import BankedExample, InMemoryExampleBank


class ExampleInput(BaseModel):
    """Input schema for adding an example to the bank."""

    title: str = Field(
        description="Short descriptive name (e.g., 'Handle nested JSON')"
    )
    keywords: list[str] = Field(
        description="Semantic keywords for retrieval - what user queries should find this example? (e.g., ['json', 'nested', 'parsing'])"
    )
    content: str = Field(
        description="The full example content. Can be any format: input/output pairs, reasoning chains, do/don't comparisons, etc."
    )


def create_example_bank_tools(bank: InMemoryExampleBank) -> FunctionToolset:
    """Create tools for the reflection agent to manage examples.

    These tools allow the reflection agent to iteratively curate the example
    bank by adding, removing, and testing examples.
    """
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool
    def add_examples(examples: list[ExampleInput]) -> str:
        """Add multiple few-shot examples to help the student agent handle similar cases.

        Use this when you identify patterns in the failures that could be addressed
        by showing the student concrete examples of correct behavior. You can add
        many examples at once to build a comprehensive reference library.
        """
        added: list[str] = []
        for ex in examples:
            example = BankedExample(
                title=ex.title,
                keywords=ex.keywords,
                content=ex.content,
            )
            bank.add(example)
            added.append(f"'{ex.title}' (id: {example.id})")
        if not added:
            return "No examples provided."
        return f"Added {len(added)} examples: {', '.join(added)}"

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
        """View all current examples in the bank (titles and keywords only).

        Use this to understand what examples already exist before adding
        new ones, or to identify examples that should be removed.
        Use read_example() to see the full content of a specific example.
        """
        if len(bank) == 0:
            return "Example bank is empty."
        lines = ["Current examples in bank:"]
        for ex in bank:
            kw_str = ", ".join(ex.keywords) if ex.keywords else "(no keywords)"
            lines.append(f"- [{ex.id}] {ex.title}")
            lines.append(f"  Keywords: {kw_str}")
        return "\n".join(lines)

    @toolset.tool
    def read_examples(example_ids: list[str]) -> str:
        """Read the full content of one or more examples.

        Use this after list_examples() to see the complete content of
        examples you want to review, modify, or use as reference.
        """
        results: list[str] = []
        for example_id in example_ids:
            example = bank.get(example_id)
            if example is None:
                results.append(f"[{example_id}] Not found")
            else:
                lines = [
                    f"[{example.id}] {example.title}",
                    f"Keywords: {', '.join(example.keywords) if example.keywords else '(none)'}",
                    "",
                    example.content,
                ]
                results.append("\n".join(lines))
        if not results:
            return "No example IDs provided."
        return "\n\n---\n\n".join(results)

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
