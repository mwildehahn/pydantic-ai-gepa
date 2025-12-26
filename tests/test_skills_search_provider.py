from __future__ import annotations

import pytest

from pydantic_ai_gepa.skills import SkillsFS
from pydantic_ai_gepa.skills.search import (
    InMemorySkillsSearchProvider,
    LocalSkillsSearchProvider,
)


@pytest.mark.asyncio
async def test_local_skills_search_provider_searches_skill_md_and_files() -> None:
    fs = SkillsFS()
    fs.write_text(
        "index/tasks/SKILL.md",
        "---\nname: index-tasks\ndescription: Create and update tasks\n---\n# Tasks\n\nBody\n",
    )
    fs.write_text("index/tasks/references/REF.md", "reference text")

    provider = LocalSkillsSearchProvider()
    results = await provider.search(query="reference", top_k=10, fs=fs, candidate=None)
    assert results
    assert any(r.file_path == "references/REF.md" for r in results)


@pytest.mark.asyncio
async def test_inmemory_search_provider_requires_reindexing() -> None:
    fs = SkillsFS()
    fs.write_text(
        "spaces/SKILL.md",
        "---\nname: spaces\ndescription: old\n---\n# Spaces\n\nOld body\n",
    )

    provider = InMemorySkillsSearchProvider()
    empty = await provider.search(query="spaces", top_k=5, fs=fs, candidate=None)
    assert empty == []

    await provider.reindex_skill(fs=fs, skill_path="spaces", candidate=None)
    results = await provider.search(query="spaces", top_k=5, fs=fs, candidate=None)
    assert results
    assert results[0].skill_path == "spaces"


@pytest.mark.asyncio
async def test_inmemory_search_provider_reindex_overwrites_old_content() -> None:
    fs = SkillsFS()
    fs.write_text(
        "spaces/SKILL.md",
        "---\nname: spaces\ndescription: old\n---\n# Spaces\n\nOld body\n",
    )
    fs.write_text("spaces/examples/001.md", "old example")

    provider = InMemorySkillsSearchProvider()
    await provider.reindex_skill(fs=fs, skill_path="spaces", candidate=None)

    # Update content on disk and reindex.
    fs.write_text(
        "spaces/SKILL.md",
        "---\nname: spaces\ndescription: new\n---\n# Spaces\n\nNew body\n",
    )
    fs.write_text("spaces/examples/001.md", "new example")
    await provider.reindex_skill(fs=fs, skill_path="spaces", candidate=None)

    results = await provider.search(
        query="new example", top_k=10, fs=fs, candidate=None
    )
    assert results
    assert any(r.file_path == "examples/001.md" for r in results)
