"""Tests for the example bank."""

from __future__ import annotations

from pydantic_ai_gepa.gepa_graph.example_bank import (
    BankedExample,
    ExampleBank,
    InMemoryExampleBank,
)


def _make_example(
    title: str, keywords: list[str], content: str = "content", id: str | None = None
) -> BankedExample:
    if id is not None:
        return BankedExample(id=id, title=title, keywords=keywords, content=content)
    return BankedExample(title=title, keywords=keywords, content=content)


class TestBankedExample:
    def test_creation(self) -> None:
        ex = BankedExample(
            title="Handle nested JSON",
            keywords=["json", "nested", "parsing"],
            content="When parsing nested JSON, use recursive descent...",
        )
        assert ex.title == "Handle nested JSON"
        assert ex.keywords == ["json", "nested", "parsing"]
        assert "recursive" in ex.content

    def test_auto_generates_id(self) -> None:
        ex1 = BankedExample(title="Ex1", keywords=[], content="")
        ex2 = BankedExample(title="Ex2", keywords=[], content="")
        assert ex1.id  # Not empty
        assert ex2.id  # Not empty
        assert ex1.id != ex2.id  # Unique

    def test_custom_id(self) -> None:
        ex = BankedExample(id="custom-123", title="Ex", keywords=[], content="")
        assert ex.id == "custom-123"


class TestInMemoryExampleBank:
    def test_implements_protocol(self) -> None:
        bank = InMemoryExampleBank()
        assert isinstance(bank, ExampleBank)

    def test_add_and_len(self) -> None:
        bank = InMemoryExampleBank()
        assert len(bank) == 0

        bank.add(_make_example("test", ["keyword"]))
        assert len(bank) == 1

        bank.add(_make_example("test2", ["keyword2"]))
        assert len(bank) == 2

    def test_add_many(self) -> None:
        bank = InMemoryExampleBank()
        examples = [
            _make_example("ex1", ["a"]),
            _make_example("ex2", ["b"]),
            _make_example("ex3", ["c"]),
        ]
        bank.add_many(examples)
        assert len(bank) == 3

    def test_clear(self) -> None:
        bank = InMemoryExampleBank()
        bank.add_many([_make_example("ex1", ["a"]), _make_example("ex2", ["b"])])
        assert len(bank) == 2

        bank.clear()
        assert len(bank) == 0

    def test_iter(self) -> None:
        bank = InMemoryExampleBank()
        examples = [
            _make_example("ex1", ["a"]),
            _make_example("ex2", ["b"]),
        ]
        bank.add_many(examples)

        iterated = list(bank)
        assert iterated == examples

    def test_search_empty_bank(self) -> None:
        bank = InMemoryExampleBank()
        results = bank.search("anything", k=3)
        assert results == []

    def test_search_by_keyword(self) -> None:
        bank = InMemoryExampleBank()
        bank.add_many(
            [
                _make_example("JSON parsing", ["json", "parsing"]),
                _make_example("Error handling", ["error", "exception"]),
                _make_example("API design", ["rest", "api"]),
            ]
        )

        results = bank.search("json", k=2)
        assert len(results) >= 1
        assert results[0].title == "JSON parsing"

    def test_search_by_title_term(self) -> None:
        bank = InMemoryExampleBank()
        bank.add_many(
            [
                _make_example("JSON parsing", ["parsing"]),
                _make_example("Error handling", ["exception"]),
            ]
        )

        results = bank.search("error", k=2)
        assert results[0].title == "Error handling"

    def test_search_respects_k(self) -> None:
        bank = InMemoryExampleBank()
        bank.add_many([_make_example(f"ex{i}", ["common"]) for i in range(10)])

        results = bank.search("common", k=3)
        assert len(results) == 3

    def test_search_ranks_by_relevance(self) -> None:
        bank = InMemoryExampleBank()
        bank.add_many(
            [
                _make_example("General", ["json"]),
                _make_example("Specific", ["json", "nested", "parsing"]),
            ]
        )

        # Query matches more keywords in "Specific"
        results = bank.search("json nested parsing", k=2)
        assert results[0].title == "Specific"

    def test_search_case_insensitive(self) -> None:
        bank = InMemoryExampleBank()
        bank.add(_make_example("JSON Parsing", ["JSON", "PARSING"]))

        results = bank.search("json parsing", k=1)
        assert len(results) == 1
        assert results[0].title == "JSON Parsing"

    def test_copy(self) -> None:
        bank = InMemoryExampleBank()
        bank.add_many(
            [
                _make_example("ex1", ["a"]),
                _make_example("ex2", ["b"]),
            ]
        )

        copied = bank.copy()

        # Same content
        assert len(copied) == 2
        assert list(copied) == list(bank)

        # Independent - modifying copy doesn't affect original
        copied.add(_make_example("ex3", ["c"]))
        assert len(copied) == 3
        assert len(bank) == 2

    def test_idf_scoring(self) -> None:
        """Test that rare keywords get higher scores than common ones."""
        bank = InMemoryExampleBank()
        # "common" appears in all examples, "rare" only in one
        bank.add_many(
            [
                _make_example("Has rare", ["common", "rare"]),
                _make_example("Common only 1", ["common"]),
                _make_example("Common only 2", ["common"]),
            ]
        )

        # Searching for "rare" should rank "Has rare" first
        results = bank.search("rare", k=3)
        assert results[0].title == "Has rare"

        # Searching for "common" gives all results (less discriminative)
        results = bank.search("common", k=3)
        assert len(results) == 3

    def test_get_by_id(self) -> None:
        bank = InMemoryExampleBank()
        ex = _make_example("Test", ["a"], id="test-id")
        bank.add(ex)

        assert bank.get("test-id") == ex
        assert bank.get("nonexistent") is None

    def test_remove(self) -> None:
        bank = InMemoryExampleBank()
        ex1 = _make_example("Ex1", ["a"], id="id-1")
        ex2 = _make_example("Ex2", ["b"], id="id-2")
        bank.add_many([ex1, ex2])

        assert len(bank) == 2
        assert bank.remove("id-1") is True
        assert len(bank) == 1
        assert bank.get("id-1") is None
        assert bank.get("id-2") == ex2

        # Removing non-existent returns False
        assert bank.remove("nonexistent") is False
        assert len(bank) == 1

    def test_remove_many(self) -> None:
        bank = InMemoryExampleBank()
        examples = [_make_example(f"Ex{i}", ["kw"], id=f"id-{i}") for i in range(5)]
        bank.add_many(examples)

        assert len(bank) == 5
        removed = bank.remove_many(["id-0", "id-2", "id-4"])
        assert removed == 3
        assert len(bank) == 2
        assert bank.get("id-1") is not None
        assert bank.get("id-3") is not None
        assert bank.get("id-0") is None

    def test_remove_many_with_nonexistent(self) -> None:
        bank = InMemoryExampleBank()
        bank.add(_make_example("Ex", ["kw"], id="exists"))

        # Mix of existing and non-existing IDs
        removed = bank.remove_many(["exists", "nope", "also-nope"])
        assert removed == 1
        assert len(bank) == 0

    def test_copy_preserves_id_index(self) -> None:
        bank = InMemoryExampleBank()
        ex = _make_example("Test", ["a"], id="test-id")
        bank.add(ex)

        copied = bank.copy()
        assert copied.get("test-id") == ex

        # Removing from copy doesn't affect original
        copied.remove("test-id")
        assert copied.get("test-id") is None
        assert bank.get("test-id") == ex
