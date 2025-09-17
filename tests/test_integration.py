"""Integration tests for pydantic-ai-gepa."""

from typing import Any
from inline_snapshot import snapshot
from pydantic_ai_gepa.adapter import PydanticAIGEPAAdapter
from pydantic_ai_gepa.components import (
    extract_seed_candidate,
    get_component_names,
)
from pydantic_ai_gepa.types import DataInst, DataInstWithPrompt, RolloutOutput

from pydantic_ai import Agent
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models.test import TestModel


def test_extract_seed_candidate():
    """Test extracting prompts from an agent."""
    agent = Agent(
        TestModel(),
        instructions="Be helpful",
        system_prompt=["System prompt 1", "System prompt 2"],
    )

    candidate = extract_seed_candidate(agent)

    assert candidate["instructions"] == "Be helpful"
    assert candidate["system_prompt:0"] == "System prompt 1"
    assert candidate["system_prompt:1"] == "System prompt 2"
    assert "system_prompt:2" not in candidate


def test_get_component_names():
    """Test getting optimizable component names."""
    agent = Agent(
        TestModel(),
        instructions="Instructions",
        system_prompt=["Prompt 1", "Prompt 2"],
    )

    components = get_component_names(agent)

    assert "instructions" in components
    assert "system_prompt:0" in components
    assert "system_prompt:1" in components
    assert len(components) == 3


def test_process_data_instance():
    """Test processing a single data instance."""
    agent = Agent(
        TestModel(custom_output_text="Test response"), instructions="Be helpful"
    )

    def metric(
        data_inst: DataInst, output: RolloutOutput[Any]
    ) -> tuple[float, str | None]:
        if output.success:
            return (0.8, "Good")
        return (0.0, "Failed")

    adapter = PydanticAIGEPAAdapter(agent, metric)

    # Test without traces
    data_inst = DataInstWithPrompt(
        user_prompt=UserPromptPart(content="Hello"),
        message_history=None,
        metadata={},
        case_id="test-4",
    )
    result = adapter.process_data_instance(data_inst, capture_traces=False)

    assert "output" in result
    assert "score" in result
    assert result["output"].success is True
    assert result["score"] == 0.8
    assert "trajectory" not in result

    # Test with traces
    result_with_trace = adapter.process_data_instance(data_inst, capture_traces=True)

    assert "output" in result_with_trace
    assert "score" in result_with_trace
    assert "trajectory" in result_with_trace
    assert result_with_trace["output"].success is True
    assert result_with_trace["score"] == 0.8
    assert result_with_trace["trajectory"].final_output == "Test response"


def test_make_reflective_dataset():
    """Test making a reflective dataset."""
    agent = Agent(
        TestModel(custom_output_text="Test response"), instructions="Be helpful"
    )

    def metric(
        data_inst: DataInst, output: RolloutOutput[Any]
    ) -> tuple[float, str | None]:
        if output.success:
            return (0.8, "Good")
        return (0.0, "Failed")

    adapter = PydanticAIGEPAAdapter(agent, metric)
    candidate = extract_seed_candidate(agent)

    data_inst = DataInstWithPrompt(
        user_prompt=UserPromptPart(content="Hello"),
        message_history=None,
        metadata={},
        case_id="test-4",
    )
    result = adapter.evaluate([data_inst], candidate, capture_traces=True)

    reflective_dataset = adapter.make_reflective_dataset(
        candidate, result, ["instructions"]
    )
    assert reflective_dataset == snapshot(
        {
            "instructions": [
                {
                    "user_prompt": "Hello",
                    "assistant_response": "Test response",
                    "error": None,
                    "score": 0.8,
                    "success": True,
                    "feedback": "Good",
                }
            ]
        }
    )
