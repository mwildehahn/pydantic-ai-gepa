"""LLM-based proposal generation utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model

from pydantic_ai_gepa.inspection import InspectionAborted

from ...adapter import (
    ComponentReflectiveDataset,
    ReflectiveDataset,
    SharedReflectiveDataset,
)
from ..models import CandidateProgram

DEFAULT_AGENT_INSTRUCTIONS = """You are optimizing prompt components for a student agent based on production trajectory analysis.

## Your Task

Analyze the provided execution traces to understand:
1. What patterns led to success vs. failure
2. What domain knowledge is revealed by the correct solutions
3. How the student agent is currently behaving
4. What specific guidance would help the student improve

Then provide updated component texts that encode this knowledge clearly and effectively.

## Critical Understanding: Teaching Less Capable Models

When the student model is less capable (small, cheap, or "dumb"), you MUST compensate with:

**1. Extensive Few-Shot Examples**
- Include multiple complete examples showing full reasoning traces
- Show multi-turn interactions, not just single inputs/outputs
- Demonstrate exactly how to break down complex problems
- Include edge cases and common pitfalls
- Show the pattern: think → plan → execute → verify

**2. Explicit Step-by-Step Guidance**
- Break complex tasks into numbered steps
- State prerequisites and dependencies clearly
- Provide decision trees for common scenarios
- Make implicit knowledge explicit

**3. Domain Knowledge Extraction**
- When traces show successful solutions, extract the underlying approach
- Codify problem-solving patterns into the instructions
- Document common gotchas revealed by failures
- Include heuristics for when to use which approach

**4. Constraint Management**
- If traces show tool limit failures, teach efficiency
- If traces show repeated errors, provide guardrails
- If traces show missing context, add reminder patterns
- Encode resource budgets directly in instructions

## Analysis Framework

For each set of traces, identify:
- **Successes:** What approaches worked? What reasoning was sound?
- **Failures:** What went wrong? Where did the student get stuck?
- **Patterns:** Are failures random or systematic?
- **Missing Knowledge:** What domain expertise does the student lack?
- **Efficiency Issues:** Unnecessary tool calls, redundant work, lack of planning?
- **Structural Issues:** Tool limit hits, timeout patterns, repeated errors?

## Output Requirements

Always respond using the structured schema with:
1. An entry for each component you were asked to update
2. Rewritten component text that:
   - Incorporates lessons from the trajectory analysis
   - Adds few-shot examples if the student needs more guidance
   - Encodes domain knowledge from successful traces
   - Provides explicit guardrails against observed failure modes
   - Maintains token efficiency while being sufficiently detailed

**Balance:** For capable models, stay concise. For less capable models, err on the side of being thorough and explicit, even if it costs more tokens. A longer, clearer prompt that works is better than a short prompt that fails.

## Specific Guidance on Few-Shot Examples

When adding examples to instructions, follow this pattern:

```
Example 1: [Brief description]

Input: [The task/problem]

Reasoning:
- First, I need to [understand/identify/determine] ...
- This requires [approach/tool/calculation] ...
- I should [plan/prepare/check] ...

Action: [Tool call or response]

Result: [What happened]

Next step: [Continue reasoning...]

Final answer: [Solution]
```

Include 2-4 complete examples showing:
- Simple case (baseline)
- Complex case (shows full capability)
- Edge case (shows careful thinking)
- Failure recovery (shows error handling)

**When evidence reveals:**
- Domain invariants → Encode them explicitly in instructions
- Evaluator heuristics → Surface them as success criteria
- Recurring failure motifs → Add guardrails and reminders
- Efficient patterns → Highlight and explain them
- Resource constraints → Build awareness into the instructions

## Adaptive Strategy Based on Progress

You may receive context about the optimization progress (iteration number, scores). Use this to adapt your approach:

**Early iterations (1-3):**
- Make focused, targeted improvements
- Test hypotheses about what's wrong
- Stay relatively concise

**Mid iterations (4-6) with no improvement:**
- You're likely stuck in a local optimum or missing something fundamental
- Time to get more aggressive
- Add comprehensive few-shot examples (2-4 complete examples)
- Make structural changes, not just tweaks
- Consider if you're addressing the right problem

**Late iterations (7+) still stuck:**
- Emergency measures needed
- Add extensive few-shot examples showing complete reasoning chains
- Be very explicit and detailed, even at token cost
- The student needs hand-holding - provide it
- Look for completely different angles of attack

**When you see improvement:**
- You're on the right track
- Build on what's working
- Refine and extend successful patterns

Always ensure updated components work together cohesively as a coordinated system."""


class TrajectoryAnalysis(BaseModel):
    """Analysis of what happened in the traces before proposing changes."""

    what_went_well: str = Field(
        description="Patterns and approaches that led to successful outcomes in the traces. What did the student do correctly?",
    )
    what_went_wrong: str = Field(
        description="Failure patterns, errors, and inefficiencies observed in the traces. What caused the student to fail or underperform?",
    )
    areas_to_improve: str = Field(
        description="Specific aspects to address in the updated components. What changes will most improve performance?",
    )


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

    reasoning: TrajectoryAnalysis = Field(
        description="Analysis of the traces before making changes. This helps ensure updates are grounded in evidence.",
    )
    updated_components: list[ComponentUpdate] = Field(
        default_factory=list,
        description="Updates for each requested component, informed by the reasoning above.",
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
        iteration: int | None = None,
        current_best_score: float | None = None,
        parent_score: float | None = None,
    ) -> dict[str, str]:
        """Propose new texts for each component via the structured agent.

        Args:
            candidate: The candidate program to optimize
            reflective_data: Training data with execution traces
            components: Component names to update
            model: Model to use for proposal generation
            iteration: Current optimization iteration (if available)
            current_best_score: Best score achieved so far (if available)
            parent_score: Score of the candidate being improved from (if available)
        """
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
            iteration=iteration,
            current_best_score=current_best_score,
            parent_score=parent_score,
        )

        try:
            result = await self._agent.run(prompt, model=model)
        except InspectionAborted:
            raise
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
        iteration: int | None = None,
        current_best_score: float | None = None,
        parent_score: float | None = None,
    ) -> str:
        lines = [
            "# Role: Component Optimizer for Student Agent",
            "",
            "You are optimizing prompt components for a student agent based on its production performance.",
            "",
        ]

        # Add optimization progress context if available
        if iteration is not None or current_best_score is not None or parent_score is not None:
            lines.extend([
                "## Optimization Progress",
                "",
            ])
            if iteration is not None:
                lines.append(f"- **Current iteration:** {iteration}")
            if current_best_score is not None:
                lines.append(f"- **Best score so far:** {current_best_score:.4f}")
            if parent_score is not None:
                lines.append(f"- **Score of candidate being improved:** {parent_score:.4f}")

            # Add interpretation guidance
            if iteration is not None and current_best_score is not None and parent_score is not None:
                if current_best_score == parent_score and iteration > 3:
                    lines.extend([
                        "",
                        "**⚠️ Plateau detected:** No improvement for several iterations. Time to try more aggressive changes.",
                        "Consider adding comprehensive few-shot examples or restructuring the approach.",
                    ])
                elif current_best_score > parent_score:
                    lines.extend([
                        "",
                        "**✓ Improvement detected:** Recent changes helped. Build on this success.",
                    ])

            lines.extend([
                "",
                "---",
                "",
            ])

        lines.extend([
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
        ])

        catalog_tool_defs = self._build_tool_definitions_from_candidate(candidate)

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
        tool_map: dict[str, dict[str, Any]] = {}
        for tool in tools:
            name = self._extract_tool_name(tool)
            if name:
                tool_map[name] = tool

        for name, catalog_tool in catalog_tool_defs.items():
            if name in tool_map:
                self._merge_tool_definitions(tool_map[name], catalog_tool)
            else:
                tools.append(catalog_tool)
                tool_map[name] = catalog_tool

        output_tool_names: list[str] = []
        if tools:
            for tool in tools:
                if isinstance(tool, Mapping) and tool.get("kind") == "output":
                    function_block = tool.get("function")
                    if isinstance(function_block, Mapping):
                        name = function_block.get("name")
                        if isinstance(name, str) and name:
                            output_tool_names.append(name)
            lines.append("**Tools available to student (JSON Schema):**")
            lines.append("```json")
            lines.append(json.dumps(tools, indent=2))
            lines.append("```")
            lines.append("")
            if output_tool_names:
                sample_name = output_tool_names[0]
                lines.append(
                    f"_Tools with `\"kind\": \"output\"` (e.g., `{sample_name}`) end the run."
                    " Teach the student to call the appropriate output tool when finalizing their answer._"
                )
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
                "- Are there patterns of inefficient tool usage (redundant calls, speculative calls, lack of planning)?",
                "- How can prompts guide the student to gather what's needed in fewer, well-targeted tool calls?",
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

        def merge_tool_entries(
            entries: Any,
            *,
            default_kind: str | None = None,
        ) -> None:
            if not entries:
                return
            for tool in entries:
                if not isinstance(tool, Mapping):
                    continue
                function = tool.get("function")
                if not isinstance(function, Mapping):
                    continue
                tool_name = function.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    continue
                if tool_name in tools_map:
                    continue
                normalized_tool = dict(tool)
                if default_kind and not normalized_tool.get("kind"):
                    normalized_tool["kind"] = default_kind
                tools_map[tool_name] = normalized_tool

        for record in records:
            merge_tool_entries(record.get("tools"))
            merge_tool_entries(record.get("output_tools"), default_kind="output")

        return list(tools_map.values())

    @staticmethod
    def _extract_tool_name(tool: Mapping[str, Any]) -> str | None:
        function_block = tool.get("function")
        if isinstance(function_block, Mapping):
            name = function_block.get("name")
            if isinstance(name, str) and name.strip():
                return name
        return None

    @staticmethod
    def _merge_tool_definitions(
        base: dict[str, Any],
        supplement: dict[str, Any],
    ) -> None:
        if "kind" not in base and supplement.get("kind"):
            base["kind"] = supplement["kind"]

        base_function = base.get("function")
        supplement_function = supplement.get("function")
        if not isinstance(base_function, Mapping) or not isinstance(supplement_function, Mapping):
            return

        if not isinstance(base_function, dict):
            base_function = dict(base_function)
            base["function"] = base_function

        if not base_function.get("description") and supplement_function.get("description"):
            base_function["description"] = supplement_function["description"]

        if not base_function.get("parameters") and supplement_function.get("parameters"):
            base_function["parameters"] = supplement_function["parameters"]

    def _build_tool_definitions_from_candidate(
        self,
        candidate: CandidateProgram,
    ) -> dict[str, dict[str, Any]]:
        tool_defs: dict[str, dict[str, Any]] = {}

        for component_name, component_value in candidate.components.items():
            if not component_name.startswith("tool:"):
                continue

            remainder = component_name[len("tool:") :]
            if ":" not in remainder:
                continue

            tool_name, _, key = remainder.partition(":")
            if not tool_name:
                continue

            entry = tool_defs.setdefault(tool_name, self._init_tool_entry(tool_name))
            function_block = entry["function"]
            text = component_value.text.strip()
            if not text:
                continue

            if key == "description":
                function_block["description"] = text
            elif key.startswith("param:"):
                path = key[len("param:") :]
                parameters = function_block.setdefault(
                    "parameters",
                    {"type": "object", "properties": {}},
                )
                self._inject_catalog_parameter(parameters, path, text)

        return {
            name: entry
            for name, entry in tool_defs.items()
            if self._tool_entry_has_content(entry)
        }

    @staticmethod
    def _tool_entry_has_content(entry: Mapping[str, Any]) -> bool:
        function_block = entry.get("function")
        if not isinstance(function_block, Mapping):
            return False
        if function_block.get("description"):
            return True
        parameters = function_block.get("parameters")
        if isinstance(parameters, Mapping):
            properties = parameters.get("properties")
            if isinstance(properties, Mapping) and properties:
                return True
        return False

    def _init_tool_entry(self, tool_name: str) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool_name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        kind = self._guess_tool_kind(tool_name)
        if kind:
            entry["kind"] = kind
        return entry

    @staticmethod
    def _guess_tool_kind(tool_name: str) -> str | None:
        normalized = tool_name.casefold()
        if normalized == "final_result" or normalized.startswith("final_result_"):
            return "output"
        return None

    def _inject_catalog_parameter(
        self,
        schema: dict[str, Any],
        path: str,
        description: str,
    ) -> None:
        if not path:
            return

        segments = [segment for segment in path.split(".") if segment]
        if not segments:
            return

        node = schema
        node.setdefault("type", "object")
        for idx, raw_segment in enumerate(segments):
            is_array = raw_segment.endswith("[]")
            segment = raw_segment[:-2] if is_array else raw_segment
            if not segment:
                continue

            properties = node.setdefault("properties", {})
            child = properties.setdefault(segment, {})

            if is_array:
                child.setdefault("type", "array")
                items = child.setdefault("items", {"type": "object"})
                node = items
                continue

            if idx < len(segments) - 1:
                child.setdefault("type", "object")
                node = child
            else:
                node = child

        if "type" not in node:
            node["type"] = "string"
        node["description"] = description

    @staticmethod
    def _records_for_component(
        dataset: ReflectiveDataset,
        component: str,
    ) -> Sequence[Mapping[str, Any]]:
        if isinstance(dataset, SharedReflectiveDataset):
            return dataset.records
        return dataset.records_by_component.get(component, ())

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
    "TrajectoryAnalysis",
]
