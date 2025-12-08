"""Example bank for few-shot retrieval during agent execution."""

from __future__ import annotations

import uuid
from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


class BankedExample(BaseModel):
    """A few-shot example stored in the example bank."""

    id: str = Field(default_factory=_generate_id)
    """Unique identifier for the example (auto-generated if not provided)."""

    title: str
    """Short descriptive name for the example."""

    keywords: list[str]
    """Semantic keywords for retrieval (model-generated)."""

    content: str
    """The example content (free-form, can be anything useful)."""


@runtime_checkable
class ExampleBank(Protocol):
    """Protocol for storing and retrieving few-shot examples."""

    def add(self, example: BankedExample) -> None:
        """Add an example to the bank."""
        ...

    def add_many(self, examples: Sequence[BankedExample]) -> None:
        """Add multiple examples to the bank."""
        ...

    def remove(self, example_id: str) -> bool:
        """Remove an example by ID.

        Returns:
            True if the example was found and removed, False otherwise.
        """
        ...

    def remove_many(self, example_ids: Sequence[str]) -> int:
        """Remove multiple examples by ID.

        Returns:
            The number of examples that were actually removed.
        """
        ...

    def get(self, example_id: str) -> BankedExample | None:
        """Get an example by ID.

        Returns:
            The example if found, None otherwise.
        """
        ...

    def search(self, query: str, k: int = 3) -> list[BankedExample]:
        """Search for relevant examples.

        Args:
            query: The search query (typically the user's input).
            k: Maximum number of examples to return.

        Returns:
            List of matching examples, ordered by relevance (best first).
        """
        ...

    def clear(self) -> None:
        """Remove all examples from the bank."""
        ...

    def __len__(self) -> int:
        """Return the number of examples in the bank."""
        ...

    def __iter__(self):
        """Iterate over all examples in the bank."""
        ...


class InMemoryExampleBank:
    """In-memory example bank using keyword matching.

    Uses simple TF-IDF-style scoring over keywords and title terms.
    Suitable for small example sets (<1000 examples).
    """

    def __init__(self) -> None:
        self._examples: list[BankedExample] = []
        self._by_id: dict[str, BankedExample] = {}
        self._idf: dict[str, float] = {}

    def add(self, example: BankedExample) -> None:
        """Add an example to the bank."""
        self._examples.append(example)
        self._by_id[example.id] = example
        self._rebuild_idf()

    def add_many(self, examples: Sequence[BankedExample]) -> None:
        """Add multiple examples to the bank."""
        self._examples.extend(examples)
        for ex in examples:
            self._by_id[ex.id] = ex
        self._rebuild_idf()

    def remove(self, example_id: str) -> bool:
        """Remove an example by ID."""
        if example_id not in self._by_id:
            return False
        example = self._by_id.pop(example_id)
        self._examples.remove(example)
        self._rebuild_idf()
        return True

    def remove_many(self, example_ids: Sequence[str]) -> int:
        """Remove multiple examples by ID."""
        removed = 0
        ids_to_remove = set(example_ids)
        # Filter in one pass to avoid repeated list.remove() calls
        new_examples = []
        for ex in self._examples:
            if ex.id in ids_to_remove:
                self._by_id.pop(ex.id, None)
                removed += 1
            else:
                new_examples.append(ex)
        self._examples = new_examples
        if removed > 0:
            self._rebuild_idf()
        return removed

    def get(self, example_id: str) -> BankedExample | None:
        """Get an example by ID."""
        return self._by_id.get(example_id)

    def search(self, query: str, k: int = 3) -> list[BankedExample]:
        """Search for relevant examples using keyword matching.

        Scores examples based on TF-IDF-weighted term overlap between
        the query and each example's title + keywords.
        """
        if not self._examples:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return self._examples[:k]

        scored = []
        for example in self._examples:
            score = self._score(query_terms, example)
            scored.append((score, example))

        # Sort by score descending, return top k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:k]]

    def clear(self) -> None:
        """Remove all examples from the bank."""
        self._examples.clear()
        self._idf.clear()

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self):
        return iter(self._examples)

    def copy(self) -> InMemoryExampleBank:
        """Create a shallow copy of the bank (for candidate forking)."""
        new_bank = InMemoryExampleBank()
        new_bank._examples = list(self._examples)
        new_bank._by_id = dict(self._by_id)
        new_bank._idf = dict(self._idf)
        return new_bank

    def _rebuild_idf(self) -> None:
        """Rebuild IDF scores for all terms."""
        import math
        from collections import Counter

        if not self._examples:
            self._idf = {}
            return

        doc_freq: Counter[str] = Counter()
        for example in self._examples:
            terms = self._get_example_terms(example)
            for term in terms:
                doc_freq[term] += 1

        n = len(self._examples)
        self._idf = {term: math.log(n / df) for term, df in doc_freq.items()}

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text into lowercase terms."""
        # Simple whitespace tokenization + lowercase
        # Could be extended with stemming, stop word removal, etc.
        return {term.lower().strip(".,!?;:\"'()[]{}") for term in text.split() if term}

    def _get_example_terms(self, example: BankedExample) -> set[str]:
        """Get all searchable terms from an example."""
        terms = self._tokenize(example.title)
        terms.update(kw.lower() for kw in example.keywords)
        return terms

    def _score(self, query_terms: set[str], example: BankedExample) -> float:
        """Score an example against query terms using TF-IDF."""
        example_terms = self._get_example_terms(example)
        score = 0.0
        for term in query_terms:
            if term in example_terms:
                score += self._idf.get(term, 1.0)
        return score


__all__ = [
    "BankedExample",
    "ExampleBank",
    "InMemoryExampleBank",
]
