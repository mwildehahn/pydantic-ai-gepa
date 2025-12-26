"""Search provider interfaces and in-memory implementations for Agent Skills.

This module intentionally avoids any external search/index dependencies.
Consumers can pass a custom provider (e.g. TurboPuffer) without the core
library importing those clients.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from ..gepa_graph.models import CandidateMap
from .fs import OverlayFS, SkillsFS
from .models import SkillSearchResult
from .skill_md import parse_skill_md


class SkillsSearchProvider(Protocol):
    async def search(
        self,
        *,
        query: str,
        top_k: int,
        fs: SkillsFS | OverlayFS,
        candidate: CandidateMap | None,
    ) -> list[SkillSearchResult]: ...

    async def reindex_skill(
        self,
        *,
        fs: SkillsFS | OverlayFS,
        skill_path: str,
        candidate: CandidateMap | None = None,
    ) -> None: ...

    async def reindex_skills(
        self,
        *,
        fs: SkillsFS | OverlayFS,
        skill_paths: Sequence[str],
        candidate: CandidateMap | None = None,
    ) -> None: ...


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate_skills_overlay_key(candidate: CandidateMap | None) -> str | None:
    """Return a stable overlay key for the skills-relevant parts of a candidate."""
    if not candidate:
        return None
    parts: list[str] = []
    for name, value in sorted(candidate.items(), key=lambda item: item[0]):
        if name.startswith("skill:"):
            parts.append(f"{name}\n{value.text}\n")
    if not parts:
        return None
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def changed_skill_paths(candidate: CandidateMap | None) -> set[str]:
    """Return skill paths that have skill components present in the candidate."""
    if not candidate:
        return set()
    paths: set[str] = set()
    for key in candidate.keys():
        if not key.startswith("skill:"):
            continue
        # skill:{skill_path}:frontmatter:description | skill:{skill_path}:body | skill:{skill_path}:file:...
        parts = key.split(":")
        if len(parts) < 3:
            continue
        skill_path = parts[1]
        if skill_path:
            paths.add(skill_path)
    return paths


def _iter_searchable_skill_files(
    fs: SkillsFS | OverlayFS,
    *,
    skill_path: str,
    subdirs: tuple[str, ...] = ("examples", "references"),
) -> list[tuple[str, str]]:
    """Return [(full_path, relative_path)] for searchable files under a skill."""
    prefixes = [f"{skill_path}/{subdir}/" for subdir in subdirs if subdir.strip()]
    items: list[tuple[str, str]] = []
    for path, _ in fs.iter_files():
        if any(path.startswith(prefix) for prefix in prefixes):
            items.append((path, path[len(skill_path) + 1 :]))
    return items


class LocalSkillsSearchProvider:
    """Cheap keyword search over SKILL.md and selected files.

    This provider does not require any external infrastructure.
    """

    async def search(
        self,
        *,
        query: str,
        top_k: int,
        fs: SkillsFS | OverlayFS,
        candidate: CandidateMap | None,
    ) -> list[SkillSearchResult]:
        return local_search_skills_sync(query=query, top_k=top_k, fs=fs)

    async def reindex_skill(
        self,
        *,
        fs: SkillsFS | OverlayFS,
        skill_path: str,
        candidate: CandidateMap | None = None,
    ) -> None:
        return None

    async def reindex_skills(
        self,
        *,
        fs: SkillsFS | OverlayFS,
        skill_paths: Sequence[str],
        candidate: CandidateMap | None = None,
    ) -> None:
        return None


def _split_text(text: str, *, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    """Chunk text into roughly max_chars chunks with overlap."""
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= max_chars:
        return [stripped]

    chunks: list[str] = []
    start = 0
    while start < len(stripped):
        end = min(len(stripped), start + max_chars)
        chunk = stripped[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(stripped):
            break
        start = max(0, end - overlap)
    return chunks


@dataclass(frozen=True)
class IndexedSkillChunk:
    id: str
    skill_path: str
    file_path: str
    doc_type: str
    text: str


class InMemorySkillsSearchProvider:
    """In-memory indexed search over skills content.

    Call `reindex_skills()` to (re)build the searchable corpus. Search does not
    require filesystem traversal once indexed.
    """

    def __init__(
        self,
        *,
        searchable_subdirs: tuple[str, ...] = ("examples", "references"),
        max_chars: int = 1200,
        overlap: int = 200,
    ) -> None:
        self._searchable_subdirs = searchable_subdirs
        self._max_chars = max_chars
        self._overlap = overlap
        self._chunks: dict[str, IndexedSkillChunk] = {}

    async def search(
        self,
        *,
        query: str,
        top_k: int,
        fs: SkillsFS | OverlayFS,
        candidate: CandidateMap | None,
    ) -> list[SkillSearchResult]:
        q = query.strip().lower()
        if not q:
            return []
        tokens = [t for t in re.split(r"\W+", q) if t]
        if not tokens:
            return []

        scored: list[tuple[float, IndexedSkillChunk]] = []
        for chunk in self._chunks.values():
            haystack = chunk.text.lower()
            score = float(sum(haystack.count(tok) for tok in tokens))
            if score <= 0:
                continue
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[SkillSearchResult] = []
        for score, chunk in scored[: max(1, top_k)]:
            results.append(
                SkillSearchResult(
                    skill_path=chunk.skill_path,
                    file_path=chunk.file_path,
                    doc_type=chunk.doc_type,
                    snippet=_snippet_for_tokens(chunk.text.lower(), tokens),
                    relevance_score=score,
                )
            )
        return results

    async def reindex_skill(
        self,
        *,
        fs: SkillsFS | OverlayFS,
        skill_path: str,
        candidate: CandidateMap | None = None,
    ) -> None:
        await self.reindex_skills(fs=fs, skill_paths=[skill_path], candidate=candidate)

    async def reindex_skills(
        self,
        *,
        fs: SkillsFS | OverlayFS,
        skill_paths: Sequence[str],
        candidate: CandidateMap | None = None,
    ) -> None:
        for skill_path in sorted(set(skill_paths)):
            self._drop_skill(skill_path)
            self._index_skill(fs, skill_path)

    def _drop_skill(self, skill_path: str) -> None:
        prefix = f"{skill_path}:"
        to_delete = [key for key in self._chunks.keys() if key.startswith(prefix)]
        for key in to_delete:
            self._chunks.pop(key, None)

    def _index_skill(self, fs: SkillsFS | OverlayFS, skill_path: str) -> None:
        try:
            raw = fs.read_text(f"{skill_path}/SKILL.md")
            skill_md = parse_skill_md(raw)
        except Exception:
            return

        # Index SKILL.md (description + body) in chunks.
        combined = f"{skill_md.frontmatter.description}\n{skill_md.body}".strip()
        for idx, chunk in enumerate(
            _split_text(combined, max_chars=self._max_chars, overlap=self._overlap)
        ):
            chunk_id = f"{skill_path}:SKILL.md:{idx}"
            self._chunks[chunk_id] = IndexedSkillChunk(
                id=chunk_id,
                skill_path=skill_path,
                file_path="SKILL.md",
                doc_type="skill_md",
                text=chunk,
            )

        # Index searchable files.
        for full_path, rel_path in _iter_searchable_skill_files(
            fs, skill_path=skill_path, subdirs=self._searchable_subdirs
        ):
            try:
                text = fs.read_text(full_path)
            except Exception:
                continue
            for idx, chunk in enumerate(
                _split_text(text, max_chars=self._max_chars, overlap=self._overlap)
            ):
                chunk_id = f"{skill_path}:{rel_path}:{idx}"
                self._chunks[chunk_id] = IndexedSkillChunk(
                    id=chunk_id,
                    skill_path=skill_path,
                    file_path=rel_path,
                    doc_type="skill_file",
                    text=chunk,
                )


def local_search_skills_sync(
    *,
    query: str,
    top_k: int,
    fs: SkillsFS | OverlayFS,
    searchable_subdirs: tuple[str, ...] = ("examples", "references"),
) -> list[SkillSearchResult]:
    """Keyword search across SKILL.md and selected skill files.

    This is intentionally lightweight and designed to work without external infra.
    """
    q = query.strip().lower()
    if not q:
        return []
    tokens = [t for t in re.split(r"\W+", q) if t]
    if not tokens:
        return []

    all_files = [path for path, _ in fs.iter_files()]

    results: list[SkillSearchResult] = []
    for skill_dir in fs.iter_skill_dirs():
        if not skill_dir:
            continue
        try:
            raw = fs.read_text(f"{skill_dir}/SKILL.md")
            skill_md = parse_skill_md(raw)
        except Exception:
            continue

        haystack = (skill_md.frontmatter.description + "\n" + skill_md.body).lower()
        score = float(sum(haystack.count(tok) for tok in tokens))
        if score > 0:
            snippet = _snippet_for_tokens(haystack, tokens)
            results.append(
                SkillSearchResult(
                    skill_path=skill_dir,
                    file_path="SKILL.md",
                    doc_type="skill_md",
                    snippet=snippet,
                    relevance_score=score,
                )
            )

        prefixes = [
            f"{skill_dir}/{subdir}/" for subdir in searchable_subdirs if subdir.strip()
        ]
        for path in all_files:
            if not any(path.startswith(prefix) for prefix in prefixes):
                continue
            rel = path[len(skill_dir) + 1 :]
            try:
                text = fs.read_text(path)
            except Exception:
                continue
            file_haystack = text.lower()
            file_score = float(sum(file_haystack.count(tok) for tok in tokens))
            if file_score <= 0:
                continue
            results.append(
                SkillSearchResult(
                    skill_path=skill_dir,
                    file_path=rel,
                    doc_type="skill_file",
                    snippet=_snippet_for_tokens(file_haystack, tokens),
                    relevance_score=file_score,
                )
            )

    results.sort(key=lambda r: r.relevance_score or 0.0, reverse=True)
    return results[: max(1, top_k)]


def _snippet_for_tokens(text: str, tokens: list[str]) -> str | None:
    for tok in tokens:
        idx = text.find(tok)
        if idx >= 0:
            start = max(0, idx - 60)
            end = min(len(text), idx + 180)
            return text[start:end].strip()
    return None


# Backwards-compatible aliases.
SkillsSearchBackend = SkillsSearchProvider
LocalSkillsSearchBackend = LocalSkillsSearchProvider


__all__ = [
    "SkillsSearchProvider",
    "LocalSkillsSearchProvider",
    "InMemorySkillsSearchProvider",
    "SkillsSearchBackend",
    "LocalSkillsSearchBackend",
    "candidate_skills_overlay_key",
    "changed_skill_paths",
    "local_search_skills_sync",
    "_hash_text",
    "_iter_searchable_skill_files",
]
