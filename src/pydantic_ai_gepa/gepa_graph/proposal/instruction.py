"""LLM-based proposal generation utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model

from ..models import CandidateProgram

DEFAULT_AGENT_INSTRUCTIONS = """Analyze agent performance data and propose improved prompt components.

Your task is to:
1. Review the reflection dataset showing how the agent performed with current prompts.
2. Read all assistant responses and the corresponding feedback.
3. Identify patterns in successes and failures.
4. Capture niche or domain-specific knowledge from the examples and weave it into the prompts so future runs don't miss it.
5. Preserve generalizable strategies that consistently work well.
6. Rewrite only the components listed in `components_to_update`, keeping other components unchanged.
7. Add few-shot examples when they help clarify the task.

Focus on making prompts clearer, more specific, and better aligned with successful outcomes.
Extract domain knowledge from the examples to enhance the instructions.

Always respond using the structured schema `updated_components: list[{component_name, optimized_value}]`
and include an entry for each component you were asked to update."""


class ComponentUpdate(BaseModel):
    """Structured representation of a component update."""

    component_name: str = Field(
        description="Name of the component that was updated.",
    )
    optimized_value: str = Field(
        description="Fully rewritten text for the component.",
    )


class InstructionProposalOutput(BaseModel):
    """Agent output schema for instruction proposals."""

    updated_components: list[ComponentUpdate] = Field(
        default_factory=list,
        description="Updates for each requested component.",
    )


class InstructionProposalGenerator:
    """Generate improved component texts via a structured agent call."""

    def __init__(self, instructions: str | None = None) -> None:
        self._agent = Agent(
            instructions=instructions or DEFAULT_AGENT_INSTRUCTIONS,
            output_type=InstructionProposalOutput,
        )

    async def propose_texts(
        self,
        *,
        candidate: CandidateProgram,
        reflective_data: Mapping[str, Sequence[Mapping[str, Any]]],
        components: Sequence[str],
        model: Model | KnownModelName | str | None,
    ) -> dict[str, str]:
        """Propose new texts for each component via the structured agent."""
        if not components:
            return {}
        if model is None:
            raise ValueError(
                "A reflection model must be provided to generate proposals."
            )

        untouched: dict[str, str] = {}
        actionable: list[str] = []
        for component in components:
            if component not in candidate.components:
                raise KeyError(f"Component '{component}' not found in candidate.")
            records = list(reflective_data.get(component, ()))
            if records:
                actionable.append(component)
            else:
                untouched[component] = candidate.components[component].text

        if not actionable:
            return untouched

        prompt = self._build_user_prompt(
            candidate=candidate,
            reflective_data=reflective_data,
            components=actionable,
        )
        try:
            result = await self._agent.run(prompt, model=model)
        except Exception:
            # Fall back to the existing component texts when the agent fails.
            return {
                **untouched,
                **{component: candidate.components[component].text for component in actionable},
            }

        updates = {
            update.component_name: update.optimized_value
            for update in result.output.updated_components
            if update.component_name
        }

        updated: dict[str, str] = dict(untouched)
        for component in actionable:
            updated[component] = updates.get(
                component, candidate.components[component].text
            )
        return updated

    def _build_user_prompt(
        self,
        *,
        candidate: CandidateProgram,
        reflective_data: Mapping[str, Sequence[Mapping[str, Any]]],
        components: Sequence[str],
    ) -> str:
        header_lines = [
            "components_to_update:",
            *(f"- `{component}`" for component in components),
            "",
            "Component details and reflective evidence:",
        ]

        sections: list[str] = []
        for component in components:
            component_value = candidate.components[component]
            dataset_markdown = self._format_dataset(
                reflective_data.get(component, ()),
            )
            section = (
                f"### Component `{component}`\n"
                f"Current text:\n```\n{component_value.text.strip()}\n```\n\n"
                f"Reflective dataset:\n{dataset_markdown}"
            )
            sections.append(section)

        return "\n".join(header_lines + ["\n\n".join(sections)])

    def _format_dataset(self, records: Sequence[Mapping[str, Any]]) -> str:
        if not records:
            return "No reflective examples were provided."

        sections: list[str] = []
        for idx, record in enumerate(records, start=1):
            lines = [f"### Example {idx}"]
            for key, value in record.items():
                if value is None or value == "":
                    continue
                label = key.replace("_", " ").title()
                lines.extend(self._format_record(label, value, indent=0))
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def _format_record(
        self,
        label: str,
        value: Any,
        *,
        indent: int,
    ) -> list[str]:
        prefix = "  " * indent
        if self._is_scalar(value):
            return [f"{prefix}- **{label}:** {self._format_scalar(value)}"]

        if isinstance(value, Mapping):
            lines = [f"{prefix}- **{label}:**"]
            for key, inner in value.items():
                sub_label = key.replace("_", " ").title()
                lines.extend(self._format_record(sub_label, inner, indent=indent + 1))
            return lines

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            lines = [f"{prefix}- **{label}:**"]
            for idx, item in enumerate(value, start=1):
                item_label = f"Item {idx}"
                lines.extend(self._format_record(item_label, item, indent=indent + 1))
            return lines

        return [f"{prefix}- **{label}:** {self._format_scalar(value)}"]

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        return isinstance(value, (str, int, float, bool)) or value is None

    @staticmethod
    def _format_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        return str(value).strip()


__all__ = [
    "InstructionProposalGenerator",
    "DEFAULT_AGENT_INSTRUCTIONS",
    "InstructionProposalOutput",
    "ComponentUpdate",
]
