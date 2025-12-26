from __future__ import annotations

from pydantic_ai_gepa.gepa_graph.models import ComponentValue
from pydantic_ai_gepa.skill_components import (
    apply_candidate_to_skills,
    extract_skill_components,
    skill_body_key,
    skill_description_key,
    skill_file_key,
)
from pydantic_ai_gepa.skills import SkillsFS, parse_skill_md


def test_extract_skill_components() -> None:
    fs = SkillsFS()
    fs.write_text(
        "index/tasks/SKILL.md",
        "---\nname: index-tasks\ndescription: tasks desc\n---\n# Tasks\n\nBody\n",
    )
    fs.write_text("index/tasks/examples/001.md", "example 001")
    comps = extract_skill_components(fs)
    assert comps[skill_description_key("index/tasks")].text == "tasks desc"
    assert "Body" in comps[skill_body_key("index/tasks")].text
    assert comps[skill_file_key("index/tasks", "examples/001.md")].text == "example 001"


def test_apply_candidate_to_skills_overlays_skill_md() -> None:
    fs = SkillsFS()
    fs.write_text(
        "spaces/SKILL.md",
        "---\nname: spaces\ndescription: old desc\n---\n# Spaces\n\nOld body\n",
    )
    candidate = {
        skill_description_key("spaces"): ComponentValue(
            name=skill_description_key("spaces"), text="new desc"
        ),
        skill_body_key("spaces"): ComponentValue(
            name=skill_body_key("spaces"), text="# Spaces\n\nNew body\n"
        ),
    }

    with apply_candidate_to_skills(fs, candidate) as overlay_fs:
        raw = overlay_fs.read_text("spaces/SKILL.md")
        parsed = parse_skill_md(raw)
        assert parsed.frontmatter.description == "new desc"
        assert "New body" in parsed.body


def test_apply_candidate_to_skills_overlays_example_files() -> None:
    fs = SkillsFS()
    fs.write_text(
        "spaces/SKILL.md",
        "---\nname: spaces\ndescription: old desc\n---\n# Spaces\n\nOld body\n",
    )
    fs.write_text("spaces/examples/001.md", "old example")
    candidate = {
        skill_file_key("spaces", "examples/001.md"): ComponentValue(
            name=skill_file_key("spaces", "examples/001.md"), text="new example"
        ),
    }

    with apply_candidate_to_skills(fs, candidate) as overlay_fs:
        assert overlay_fs.read_text("spaces/examples/001.md") == "new example"
        assert fs.read_text("spaces/examples/001.md") == "old example"
