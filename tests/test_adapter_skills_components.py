from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapter
from pydantic_ai_gepa.skill_components import skill_body_key, skill_description_key
from pydantic_ai_gepa.skills import SkillsFS
from pydantic_ai_gepa.types import MetricResult, RolloutOutput


def test_agent_adapter_includes_skill_components_in_seed() -> None:
    skills_fs = SkillsFS()
    skills_fs.write_text(
        "index/tasks/SKILL.md",
        "---\nname: index-tasks\ndescription: tasks desc\n---\n# Tasks\n\nBody\n",
    )

    agent = Agent(TestModel(custom_output_text="ok"), instructions="Base")

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        return MetricResult(score=1.0)

    adapter = AgentAdapter(agent=agent, metric=metric, skills_fs=skills_fs)
    components = adapter.get_components()

    # Skill components are not included in the seed by default; they are activated
    # dynamically by the reflection agent when needed.
    assert skill_description_key("index/tasks") not in components
    assert skill_body_key("index/tasks") not in components
