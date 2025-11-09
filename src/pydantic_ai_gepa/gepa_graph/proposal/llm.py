"""LLM-based proposal generation utilities."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import KnownModelName, Model

from ..models import CandidateProgram

DEFAULT_REFLECTION_PROMPT = (
    "You are optimizing the `{component}` component of a pydantic-ai agent.\n\n"
    "Current `{component}` text:\n"
    "```\n{current_text}\n```\n\n"
    "The following reflective dataset captures agent behavior, scores, and feedback:\n"
    "{dataset_markdown}\n\n"
    "Write an improved `{component}` that addresses the observed failures, keeps successful "
    "strategies, and incorporates domain knowledge from the examples. Respond with only the "
    "updated `{component}` text inside triple backticks."
)

_CODE_BLOCK_RE = re.compile(r"```(?:[^\n]*\n)?(.*?)```", re.DOTALL)


class LLMProposalGenerator:
    """Generate improved component texts via LLM calls."""

    def __init__(self, prompt_template: str | None = None) -> None:
        self._prompt_template = prompt_template or DEFAULT_REFLECTION_PROMPT

    async def propose_texts(
        self,
        *,
        candidate: CandidateProgram,
        reflective_data: Mapping[str, Sequence[Mapping[str, Any]]],
        components: Sequence[str],
        model: Model | KnownModelName | str | None,
    ) -> dict[str, str]:
        """Propose new texts for each component concurrently."""
        if not components:
            return {}
        if model is None:
            raise ValueError("A reflection model must be provided to generate proposals.")

        tasks = [
            asyncio.create_task(
                self._propose_for_component(
                    component=component,
                    candidate=candidate,
                    records=list(reflective_data.get(component, ())),
                    model=model,
                )
            )
            for component in components
        ]
        results = await asyncio.gather(*tasks)
        return dict(results)

    async def _propose_for_component(
        self,
        *,
        component: str,
        candidate: CandidateProgram,
        records: list[Mapping[str, Any]],
        model: Model | KnownModelName | str,
    ) -> tuple[str, str]:
        if component not in candidate.components:
            raise KeyError(f"Component '{component}' not found in candidate.")

        current_text = candidate.components[component].text
        if not records:
            return component, current_text

        prompt = self._render_prompt(
            component=component,
            current_text=current_text,
            records=records,
        )
        response = await model_request(
            model,
            [ModelRequest.user_text_prompt(prompt)],
        )
        new_text = self._extract_response_text(response.text or "")
        if not new_text:
            new_text = current_text
        return component, new_text

    def _render_prompt(
        self,
        *,
        component: str,
        current_text: str,
        records: Sequence[Mapping[str, Any]],
    ) -> str:
        dataset_markdown = self._format_dataset(records)
        return self._prompt_template.format(
            component=component,
            current_text=current_text.strip(),
            dataset_markdown=dataset_markdown,
        )

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

    @staticmethod
    def _extract_response_text(response_text: str) -> str:
        match = _CODE_BLOCK_RE.search(response_text)
        if match:
            return match.group(1).strip()
        return response_text.strip()


__all__ = ["LLMProposalGenerator", "DEFAULT_REFLECTION_PROMPT"]
