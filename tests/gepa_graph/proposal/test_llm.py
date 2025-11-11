from __future__ import annotations

from typing import Any

from inline_snapshot import snapshot
import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from pydantic_ai_gepa.adapter import (
    ComponentReflectiveDataset,
    SharedReflectiveDataset,
)
from pydantic_ai_gepa.gepa_graph.models import CandidateProgram, ComponentValue
from pydantic_ai_gepa.gepa_graph.proposal import InstructionProposalGenerator


def _make_candidate() -> CandidateProgram:
    return CandidateProgram(
        idx=0,
        components={
            "instructions": ComponentValue(
                name="instructions", text="Seed instructions"
            ),
            "tools": ComponentValue(name="tools", text="Seed tools"),
        },
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )


def _make_reflective_record() -> dict[str, Any]:
    return {
        "user_prompt": "Hello",
        "assistant_response": "Hi",
        "score": 0.5,
        "success": True,
        "feedback": "Needs more detail",
    }


@pytest.mark.asyncio
async def test_llm_generator_updates_components() -> None:
    candidate = _make_candidate()
    reflective_data = ComponentReflectiveDataset(
        records_by_component={
            "instructions": [_make_reflective_record()],
            "tools": [_make_reflective_record()],
        }
    )

    prompts: list[str] = []

    async def fake_model(messages, agent_info):
        prompt = messages[-1].parts[0].content
        prompts.append(prompt)
        content = """{
            "updated_components": [
                {"component_name": "instructions", "optimized_value": "Improved instructions"},
                {"component_name": "tools", "optimized_value": "Better tools"}
            ]
        }"""
        return ModelResponse(parts=[TextPart(content=content)])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=fake_model)
    result = await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions", "tools"],
        model=model,
    )

    assert result == {
        "instructions": "Improved instructions",
        "tools": "Better tools",
    }
    assert "components_to_update" in prompts[-1]
    assert "Component `instructions`" in prompts[-1]
    assert "Component `tools`" in prompts[-1]


@pytest.mark.asyncio
async def test_llm_generator_skips_components_without_records() -> None:
    candidate = _make_candidate()
    reflective_data = ComponentReflectiveDataset(
        records_by_component={
            "instructions": [_make_reflective_record()],
            "tools": [],
        }
    )

    async def fake_model(messages, agent_info):
        prompt = messages[-1].parts[0].content
        assert "Component `instructions`" in prompt
        assert "Component `tools`" not in prompt
        content = """{
            "updated_components": [
                {"component_name": "instructions", "optimized_value": "Improved instructions"}
            ]
        }"""
        return ModelResponse(parts=[TextPart(content=content)])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=fake_model)
    result = await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions", "tools"],
        model=model,
    )

    assert result["instructions"] == "Improved instructions"
    assert result["tools"] == "Seed tools"


@pytest.mark.asyncio
async def test_llm_generator_returns_existing_text_on_agent_failure() -> None:
    candidate = _make_candidate()
    reflective_data = ComponentReflectiveDataset(
        records_by_component={
            "instructions": [_make_reflective_record()],
            "tools": [_make_reflective_record()],
        }
    )

    async def failing_model(messages, agent_info):
        raise RuntimeError("boom")

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=failing_model)
    result = await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions", "tools"],
        model=model,
    )

    assert result == {
        "instructions": "Seed instructions",
        "tools": "Seed tools",
    }


@pytest.mark.asyncio
async def test_llm_generator_skips_entire_call_when_no_records() -> None:
    candidate = _make_candidate()
    reflective_data = ComponentReflectiveDataset(
        records_by_component={
            "instructions": [],
        }
    )
    call_count = 0

    async def tracking_model(messages, agent_info):
        nonlocal call_count
        call_count += 1
        return ModelResponse(parts=[TextPart(content="{}")])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=tracking_model)
    result = await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions"],
        model=model,
    )

    assert result["instructions"] == "Seed instructions"
    assert call_count == 0


@pytest.mark.asyncio
async def test_llm_generator_handles_shared_dataset() -> None:
    candidate = _make_candidate()
    reflective_data = SharedReflectiveDataset(records=[_make_reflective_record()])
    prompts: list[str] = []

    async def fake_model(messages, agent_info):
        prompt = messages[-1].parts[0].content
        prompts.append(prompt)
        content = """{
            "updated_components": [
                {"component_name": "instructions", "optimized_value": "Improved instructions"},
                {"component_name": "tools", "optimized_value": "Improved tools"}
            ]
        }"""
        return ModelResponse(parts=[TextPart(content=content)])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=fake_model)
    result = await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions", "tools"],
        model=model,
    )

    assert result == snapshot(
        {"instructions": "Improved instructions", "tools": "Improved tools"}
    )
    prompt = prompts[-1]
    assert prompt == snapshot("""\
components_to_update:
- `instructions`
- `tools`

Component details and reflective evidence:
### Shared Reflective Dataset
### Example 1
- **User Prompt:** Hello
- **Assistant Response:** Hi
- **Score:** 0.5
- **Success:** true
- **Feedback:** Needs more detail

### Component `instructions`
Current text:
```
Seed instructions
```

Reflective dataset:
Refer to the shared reflective dataset above.

### Component `tools`
Current text:
```
Seed tools
```

Reflective dataset:
Refer to the shared reflective dataset above.\
""")
