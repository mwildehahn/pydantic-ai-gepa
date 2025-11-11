"""LLM-based proposal generation utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model

from ...adapter import (
    ComponentReflectiveDataset,
    ReflectiveDataset,
    SharedReflectiveDataset,
)
from ..models import CandidateProgram

DEFAULT_AGENT_INSTRUCTIONS = """You are optimizing prompt components for a student agent based on production performance.

Your task is to:
1. Review the full student agent configuration to understand the context
2. Analyze production evidence showing successes and failures
3. Identify patterns across components and evidence
4. Consider cross-component dependencies and alignment issues
5. Capture domain-specific knowledge from examples
6. Preserve successful patterns; fix what doesn't work
7. Rewrite only the listed components as a coordinated update
8. Add few-shot examples when they clarify the task

Focus on making prompts clearer, more specific, and better aligned with successful outcomes.
When updating multiple components, ensure they work together cohesively.

Always respond using the structured schema with an entry for each component you were asked to update."""


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
        reflective_data: ReflectiveDataset,
        components: Sequence[str],
        model: Model | KnownModelName | str,
    ) -> dict[str, str]:
        """Propose new texts for each component via the structured agent."""
        if not components:
            return {}

        untouched: dict[str, str] = {}
        actionable: list[str] = []
        for component in components:
            if component not in candidate.components:
                raise KeyError(f"Component '{component}' not found in candidate.")

            records = list(self._records_for_component(reflective_data, component))
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
                **{
                    component: candidate.components[component].text
                    for component in actionable
                },
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
        reflective_data: ReflectiveDataset,
        components: Sequence[str],
    ) -> str:
        lines = [
            "# Role: Component Optimizer for Student Agent",
            "",
            "You are optimizing prompt components for a student agent based on its production performance.",
            "",
            "## Context",
            "- A student agent has been running with the configuration shown below",
            "- We've collected traces from real production runs",
            "- Your job is to improve specific components so the student agent performs better",
            "",
            "---",
            "",
            "## Full student agent configuration",
            "",
            "This is the complete configuration the student agent was running with:",
            "",
        ]

        # Show non-tool components in the candidate (tools are shown via JSON Schema below)
        for component_name, component_value in candidate.components.items():
            if component_name.startswith("tool:"):
                continue  # Skip tool components, they're shown in JSON Schema
            lines.append(f"**`{component_name}` given to student:**")
            lines.append("```")
            lines.append(component_value.text.strip())
            lines.append("```")
            lines.append("")

        # Collect and show tools if present in evidence
        tools = self._collect_tools(reflective_data)
        if tools:
            lines.append("**Tools available to student (JSON Schema):**")
            lines.append("```json")
            lines.append(json.dumps(tools, indent=2))
            lines.append("```")
            lines.append("")

        lines.extend(
            [
                "---",
                "",
                "## Production traces from student agent runs",
                "",
                "Each trace contains:",
                "- `messages`: Full conversation history with system prompts, user inputs, assistant responses, tool calls, and tool returns",
                "- `tools`: Tool definitions that were available (if any)",
                "- `score`: Performance score (0.0-1.0, higher is better)",
                "- `success`: Whether the run completed successfully",
                "- `feedback`: Evaluator feedback on this specific run",
                "",
            ]
        )

        # Get records based on SharedReflectiveDataset or ComponentReflectiveDataset
        if isinstance(reflective_data, SharedReflectiveDataset):
            lines.append(
                "**Use these traces to optimize the components listed below:**"
            )
            lines.append("")
            cleaned_records = [
                {k: v for k, v in record.items() if k not in ("tools", "instructions")}
                for record in reflective_data.records
            ]
            lines.extend(
                self._format_trace_sections(
                    cleaned_records,
                    heading_level="###",
                    label="Trace",
                )
            )
        else:
            lines.append(
                "_Component-specific evidence is shown with each component below._"
            )
            lines.append("")

        lines.extend(
            [
                "",
                "### Analysis guidance",
                "- What failure patterns repeat across runs?",
                "- Are components misaligned (e.g., instructions referencing tools that don't exist)?",
                "- Which successful patterns should be preserved or extended?",
                "- What domain knowledge should be codified in the prompts?",
                "",
                "---",
                "",
                "## Components to update",
                "",
                "Rewrite these components as a coordinated update based on the evidence above:",
                "",
            ]
        )

        # Show each component to update
        for component in components:
            component_value = candidate.components[component]
            lines.append(f"### Component: `{component}`")
            lines.append("Current value:")
            lines.append("```")
            lines.append(component_value.text.strip())
            lines.append("```")
            lines.append("")

            # For ComponentReflectiveDataset, show component-specific traces
            if isinstance(reflective_data, ComponentReflectiveDataset):
                component_records = reflective_data.records_by_component.get(
                    component, ()
                )
                if component_records:
                    lines.append("Evidence for this component:")
                    lines.append("")
                    cleaned_records = [
                        {
                            k: v
                            for k, v in record.items()
                            if k not in ("tools", "instructions")
                        }
                        for record in component_records
                    ]
                    lines.extend(
                        self._format_trace_sections(
                            cleaned_records,
                            heading_level="####",
                            label="Example",
                        )
                    )
                    lines.append("")

        return "\n".join(lines)

    def _collect_tools(
        self, reflective_data: ReflectiveDataset
    ) -> list[dict[str, Any]]:
        """Collect unique tools from all records in the reflective dataset."""
        tools_map: dict[str, dict[str, Any]] = {}

        records: Sequence[Mapping[str, Any]]
        if isinstance(reflective_data, SharedReflectiveDataset):
            records = reflective_data.records
        else:
            # Flatten all component records
            all_records: list[Mapping[str, Any]] = []
            for component_records in reflective_data.records_by_component.values():
                all_records.extend(component_records)
            records = all_records

        for record in records:
            if "tools" in record and record["tools"]:
                for tool in record["tools"]:
                    if isinstance(tool, dict) and "function" in tool:
                        tool_name = tool["function"].get("name")
                        if tool_name and tool_name not in tools_map:
                            tools_map[tool_name] = tool

        return list(tools_map.values())

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
    def _records_for_component(
        dataset: ReflectiveDataset,
        component: str,
    ) -> Sequence[Mapping[str, Any]]:
        if isinstance(dataset, SharedReflectiveDataset):
            return dataset.records
        return dataset.records_by_component.get(component, ())

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

    def _format_trace_sections(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        heading_level: str,
        label: str,
    ) -> list[str]:
        """Render reflective records in a readable structure."""
        if not records:
            return ["No reflective examples were provided.", ""]

        lines: list[str] = []
        for idx, record in enumerate(records, start=1):
            heading = f"{heading_level} {label} {idx}"
            user_prompt = record.get("user_prompt")
            if isinstance(user_prompt, str) and user_prompt.strip():
                heading += f": {user_prompt.strip()}"
            lines.append(heading)

            summary_lines: list[str] = []
            summary_fields = (
                "score",
                "success",
                "feedback",
                "assistant_response",
                "error",
            )
            for field in summary_fields:
                value = record.get(field)
                if value not in (None, ""):
                    formatted_label = field.replace("_", " ").title()
                    summary_lines.append(
                        f"- **{formatted_label}:** {self._format_scalar(value)}"
                    )

            if summary_lines:
                lines.extend(summary_lines)
                lines.append("")

            messages = record.get("messages")
            primary_instructions = self._extract_primary_instructions(messages)
            if messages:
                sanitized_messages = self._strip_duplicate_instructions(
                    messages,
                    primary_instructions,
                )
                lines.append("- **Messages:**")
                lines.append("```json")
                lines.append(json.dumps(sanitized_messages, indent=2))
                lines.append("```")

            run_usage = record.get("run_usage")
            if run_usage:
                lines.append("- **Usage:**")
                lines.append("```json")
                lines.append(json.dumps(run_usage, indent=2))
                lines.append("```")

            lines.append("")

        return lines

    @staticmethod
    def _extract_primary_instructions(messages: Any) -> str | None:
        if not isinstance(messages, Sequence):
            return None
        for message in messages:
            if isinstance(message, Mapping):
                instructions = message.get("instructions")
                if isinstance(instructions, str) and instructions.strip():
                    return instructions.strip()
        return None

    @staticmethod
    def _strip_duplicate_instructions(
        messages: Any,
        canonical: str | None,
    ) -> Any:
        if not isinstance(messages, list):
            return messages

        sanitized: list[Any] = []
        for message in messages:
            if not isinstance(message, dict):
                sanitized.append(message)
                continue

            message_copy = dict(message)
            instructions = message_copy.get("instructions")
            if (
                isinstance(instructions, str)
                and canonical
                and instructions.strip() == canonical
            ):
                message_copy.pop("instructions", None)

            sanitized.append(message_copy)
        return sanitized


__all__ = [
    "InstructionProposalGenerator",
    "DEFAULT_AGENT_INSTRUCTIONS",
    "InstructionProposalOutput",
    "ComponentUpdate",
]
