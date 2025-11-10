from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

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
    reflective_data = {
        "instructions": [_make_reflective_record()],
        "tools": [_make_reflective_record()],
    }

    async def fake_model(messages, agent_info):
        prompt = messages[-1].parts[0].content
        if "instructions" in prompt:
            content = "```Improved instructions```"
        else:
            content = "```Better tools```"
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


@pytest.mark.asyncio
async def test_llm_generator_parallelizes_calls() -> None:
    candidate = _make_candidate()
    reflective_data = {
        "instructions": [_make_reflective_record()],
        "tools": [_make_reflective_record()],
    }

    async def slow_model(messages, agent_info):
        await asyncio.sleep(0.1)
        return ModelResponse(parts=[TextPart(content="```done```")])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=slow_model)
    start = time.perf_counter()
    await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions", "tools"],
        model=model,
    )
    duration = time.perf_counter() - start
    assert duration < 0.18  # ~0.1s per call when parallelized


@pytest.mark.asyncio
async def test_llm_generator_skips_empty_records() -> None:
    candidate = _make_candidate()
    reflective_data = {"instructions": []}
    call_count = 0

    async def tracking_model(messages, agent_info):
        nonlocal call_count
        call_count += 1
        return ModelResponse(parts=[TextPart(content="```unused```")])

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
