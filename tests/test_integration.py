"""Integration tests for pydantic-ai-gepa."""

from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic_ai import Agent
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models.test import TestModel
import time_machine

from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapter
from pydantic_ai_gepa.components import (
    extract_seed_candidate,
    get_component_names,
)
from pydantic_ai_gepa.adapter import SharedReflectiveDataset
from pydantic_ai_gepa.types import (
    DataInst,
    DataInstWithPrompt,
    MetricResult,
    RolloutOutput,
)


def test_extract_seed_candidate():
    """Test extracting prompts from an agent."""
    agent = Agent(
        TestModel(),
        instructions="Be helpful",
    )

    candidate = extract_seed_candidate(agent)

    assert candidate["instructions"] == "Be helpful"
    assert len(candidate) == 1


def test_get_component_names():
    """Test getting optimizable component names."""
    agent = Agent(
        TestModel(),
        instructions="Instructions",
    )

    components = get_component_names(agent)

    assert "instructions" in components
    assert len(components) == 1


@pytest.mark.asyncio
async def test_process_data_instance():
    """Test processing a single data instance."""
    agent = Agent(
        TestModel(custom_output_text="Test response"), instructions="Be helpful"
    )

    def metric(data_inst: DataInst, output: RolloutOutput[Any]) -> MetricResult:
        if output.success:
            return MetricResult(score=0.8, feedback="Good")
        return MetricResult(score=0.0, feedback="Failed")

    adapter = AgentAdapter(agent, metric)

    # Test without traces
    data_inst = DataInstWithPrompt(
        user_prompt=UserPromptPart(content="Hello"),
        message_history=None,
        metadata={},
        case_id="test-4",
    )
    result = await adapter.process_data_instance(data_inst, capture_traces=False)

    assert "output" in result
    assert "score" in result
    assert result["output"].success is True
    assert result["score"] == 0.8
    assert "trajectory" not in result

    # Test with traces
    result_with_trace = await adapter.process_data_instance(
        data_inst, capture_traces=True
    )

    assert "output" in result_with_trace
    assert "score" in result_with_trace
    assert "trajectory" in result_with_trace
    assert result_with_trace["output"].success is True
    assert result_with_trace["score"] == 0.8
    assert result_with_trace["trajectory"].final_output == "Test response"


@pytest.mark.asyncio
@time_machine.travel("2023-01-01", tick=False)
async def test_make_reflective_dataset():
    """Test making a reflective dataset."""
    agent = Agent(
        TestModel(custom_output_text="Test response"), instructions="Be helpful"
    )

    def metric(data_inst: DataInst, output: RolloutOutput[Any]) -> MetricResult:
        if output.success:
            return MetricResult(score=0.8, feedback="Good")
        return MetricResult(score=0.0, feedback="Failed")

    adapter = AgentAdapter(agent, metric)
    candidate = extract_seed_candidate(agent)

    data_inst = DataInstWithPrompt(
        user_prompt=UserPromptPart(content="Hello"),
        message_history=None,
        metadata={},
        case_id="test-4",
    )
    result = await adapter.evaluate([data_inst], candidate, capture_traces=True)

    reflective_dataset = adapter.make_reflective_dataset(
        candidate=candidate,
        eval_batch=result,
        components_to_update=["instructions"],
    )
    assert reflective_dataset == snapshot(
        SharedReflectiveDataset(
            records=[
                {
                    "user_prompt": "Hello",
                    "assistant_response": "Test response",
                    "error": None,
                    "messages": [
                        {
                            "kind": "request",
                            "parts": [
                                {
                                    "type": "user_prompt",
                                    "role": "user",
                                    "content": "Hello",
                                    "timestamp": "2023-01-01T08:00:00+00:00",
                                }
                            ],
                            "instructions": "Be helpful",
                        },
                        {
                            "kind": "response",
                            "model_name": "test",
                            "timestamp": "2023-01-01T08:00:00+00:00",
                            "parts": [
                                {
                                    "type": "text",
                                    "role": "assistant",
                                    "content": "Test response",
                                }
                            ],
                            "usage": {
                                "input_tokens": 51,
                                "cache_write_tokens": 0,
                                "cache_read_tokens": 0,
                                "output_tokens": 2,
                                "input_audio_tokens": 0,
                                "cache_audio_read_tokens": 0,
                                "output_audio_tokens": 0,
                                "details": {},
                            },
                        },
                    ],
                    "run_usage": {
                        "input_tokens": 51,
                        "cache_write_tokens": 0,
                        "cache_read_tokens": 0,
                        "output_tokens": 2,
                        "input_audio_tokens": 0,
                        "cache_audio_read_tokens": 0,
                        "output_audio_tokens": 0,
                        "details": {},
                        "requests": 1,
                        "tool_calls": 0,
                    },
                    "score": 0.8,
                    "success": True,
                    "instructions": "Be helpful",
                    "feedback": "Good",
                }
            ]
        )
    )
