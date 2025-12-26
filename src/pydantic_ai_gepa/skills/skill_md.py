"""Parsing and rendering for SKILL.md files (Agent Skills spec + extensions)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


_SKILL_MD_FRONTMATTER_RE = re.compile(r"^---\s*$")


class SkillFrontmatter(BaseModel):
    """Spec-aligned frontmatter with support for extension fields via `extras`."""

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] | None = None
    allowed_tools: str | None = Field(default=None, alias="allowed-tools")

    extras: dict[str, Any] = Field(default_factory=dict, exclude=True)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


@dataclass(slots=True, frozen=True)
class SkillMd:
    frontmatter: SkillFrontmatter
    body: str


def parse_skill_md(text: str) -> SkillMd:
    """Parse a SKILL.md file into frontmatter and body."""
    lines = text.splitlines()
    if not lines or not _SKILL_MD_FRONTMATTER_RE.match(lines[0]):
        raise ValueError("SKILL.md must start with YAML frontmatter ('---')")

    end_index: int | None = None
    for idx in range(1, len(lines)):
        if _SKILL_MD_FRONTMATTER_RE.match(lines[idx]):
            end_index = idx
            break
    if end_index is None:
        raise ValueError("SKILL.md frontmatter block is not closed ('---')")

    yaml_block = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :])

    data = yaml.safe_load(yaml_block) or {}
    if not isinstance(data, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping")

    known_keys = {
        "name",
        "description",
        "license",
        "compatibility",
        "metadata",
        "allowed-tools",
        "allowed_tools",
    }
    known: dict[str, Any] = {k: v for k, v in data.items() if k in known_keys}
    extras: dict[str, Any] = {k: v for k, v in data.items() if k not in known_keys}

    frontmatter = SkillFrontmatter.model_validate(known)
    if extras:
        frontmatter.extras = dict(extras)

    return SkillMd(frontmatter=frontmatter, body=body)


def render_skill_md(skill: SkillMd) -> str:
    """Render a SkillMd back into SKILL.md text."""
    frontmatter_data = skill.frontmatter.model_dump(
        by_alias=True,
        exclude_none=True,
        exclude={"extras"},
    )
    if skill.frontmatter.extras:
        frontmatter_data.update(skill.frontmatter.extras)

    yaml_text = yaml.safe_dump(
        frontmatter_data,
        sort_keys=False,
        allow_unicode=True,
    ).strip()

    body = skill.body
    if body and not body.startswith("\n"):
        body = "\n" + body

    return f"---\n{yaml_text}\n---{body}\n"
