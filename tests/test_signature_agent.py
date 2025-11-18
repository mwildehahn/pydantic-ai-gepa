from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field
from pydantic_ai_gepa import SignatureAgent
from pydantic_ai_gepa.components import (
    apply_candidate_to_agent,
    extract_seed_candidate,
    get_component_names,
)
from pydantic_ai_gepa.gepa_graph.models import CandidateMap, ComponentValue, candidate_texts
from pydantic_ai_gepa.input_type import (
    generate_system_instructions,
    generate_user_content,
)

from pydantic_ai import Agent, ToolDefinition
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel


def _component_map(entries: dict[str, str]) -> CandidateMap:
    return {name: ComponentValue(name=name, text=value) for name, value in entries.items()}


class GeographyQuery(BaseModel):
    """Ask a question about geography."""

    question: str = Field(description="The geography question to ask")
    region: str | None = Field(
        None, description="Specific region to focus on, if applicable"
    )


class GeographyAnswer(BaseModel):
    """Answer to a geography question."""

    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    sources: list[str] = Field(
        default_factory=list, description="Sources of information"
    )


def test_signature_agent_basic():
    """Test basic SignatureAgent functionality."""
    # Create a test model with deterministic responses
    test_model = TestModel(
        custom_output_args=GeographyAnswer(
            answer="Paris",
            confidence="high",
            sources=["Common knowledge"],
        )
    )

    # Create base agent
    agent = Agent(
        test_model,
        instructions="You're an expert in geography.",
        output_type=GeographyAnswer,
        name="geography",
    )

    # Wrap with SignatureAgent
    signature_agent = SignatureAgent(
        agent,
        input_type=GeographyQuery,
        output_type=GeographyAnswer,
    )

    # Create a signature instance
    sig = GeographyQuery(
        question="What's the capital of France?", region="Western Europe"
    )

    # Test sync run
    result = signature_agent.run_signature_sync(sig)

    assert result.output.answer == "Paris"
    assert result.output.confidence == "high"
    assert result.output.sources == ["Common knowledge"]
    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    expected_signature_instructions = generate_system_instructions(sig)
    assert expected_signature_instructions == snapshot(
        """\
Ask a question about geography.

Inputs:

- `<question>` (str): The geography question to ask
- `<region>` (UnionType[str, NoneType]): Specific region to focus on, if applicable\
"""
    )
    assert request.instructions == snapshot("""\
You're an expert in geography.
Ask a question about geography.

Inputs:

- `<question>` (str): The geography question to ask
- `<region>` (UnionType[str, NoneType]): Specific region to focus on, if applicable\
""")


def test_signature_agent_with_override_candidate():
    """Test SignatureAgent with candidate override."""
    test_model = TestModel(
        custom_output_args=GeographyAnswer(
            answer="Rome",
            confidence="high",
            sources=["Historical records"],
        ),
    )

    agent = Agent(
        test_model,
        instructions="You're an expert in geography.",
        output_type=GeographyAnswer,
        name="geography",
    )

    signature_agent = SignatureAgent(
        agent,
        input_type=GeographyQuery,
        output_type=GeographyAnswer,
    )
    sig = GeographyQuery(
        question="What's the capital of Italy?", region="Southern Europe"
    )

    # Test with override candidate
    override_candidate = {
        "signature:GeographyQuery:instructions": "Focus on European capitals.",
        "signature:GeographyQuery:question:desc": "The capital city question",
        "instructions": "Be concise and accurate.",
    }

    result = signature_agent.run_signature_sync(sig, candidate=override_candidate)
    assert result.output.answer == "Rome"
    assert result.output.confidence == "high"
    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    expected_signature_instructions = generate_system_instructions(
        sig, candidate=override_candidate
    )
    assert expected_signature_instructions == snapshot(
        """\
Focus on European capitals.

Inputs:

- `<question>` (str): The capital city question
- `<region>` (UnionType[str, NoneType]): Specific region to focus on, if applicable\
"""
    )
    assert request.instructions is not None
    assert request.instructions == snapshot("""\
Be concise and accurate.
Focus on European capitals.

Inputs:

- `<question>` (str): The capital city question
- `<region>` (UnionType[str, NoneType]): Specific region to focus on, if applicable\
""")


def test_signature_agent_without_output_type():
    """Test SignatureAgent with text output."""
    test_model = TestModel(custom_output_text="The capital of France is Paris.")

    # Create base agent without output_type
    agent = Agent(
        test_model,
        instructions="You're an expert in geography.",
        name="geography",
    )

    # Wrap with SignatureAgent
    signature_agent = SignatureAgent(
        agent,
        input_type=GeographyQuery,
        output_type=str,
    )

    # Create and run with a signature
    sig = GeographyQuery(
        question="What's the capital of France?", region="Western Europe"
    )
    result = signature_agent.run_signature_sync(sig)

    assert result.output == "The capital of France is Paris."


@pytest.mark.asyncio
async def test_signature_agent_async():
    """Test async execution with SignatureAgent."""
    test_model = TestModel(
        custom_output_args=GeographyAnswer(
            answer="London",
            confidence="high",
            sources=["UK Government"],
        )
    )

    agent = Agent(
        test_model,
        instructions="Geography expert",
        output_type=GeographyAnswer,
        name="geo",
    )

    signature_agent = SignatureAgent(
        agent,
        input_type=GeographyQuery,
        output_type=GeographyAnswer,
    )
    sig = GeographyQuery(
        question="What's the capital of the UK?", region="Western Europe"
    )

    result = await signature_agent.run_signature(sig)
    assert result.output.answer == "London"
    assert result.output.confidence == "high"


@pytest.mark.asyncio
async def test_signature_agent_streaming():
    """Test streaming execution with SignatureAgent."""
    test_model = TestModel(custom_output_text="The capital is Tokyo.")

    agent = Agent(
        test_model,
        instructions="Geography expert",
        name="geo",
    )

    signature_agent = SignatureAgent(
        agent,
        input_type=GeographyQuery,
        output_type=str,
    )
    sig = GeographyQuery(question="What's the capital of Japan?", region=None)

    async with signature_agent.run_signature_stream(sig) as stream:
        output = await stream.get_output()
        assert output == snapshot("The capital is Tokyo.")


def test_prompt_generation_from_signature():
    """Test that prompts are correctly generated from signatures."""
    sig = GeographyQuery(
        question="What are the major rivers in Africa?", region="Sub-Saharan Africa"
    )

    # Test without candidate
    user_content = generate_user_content(sig)
    assert len(user_content) == 1
    assert user_content[0] == snapshot("""\
<question>What are the major rivers in Africa?</question>

<region>Sub-Saharan Africa</region>\
""")


def test_prompt_generation_with_candidate():
    """Test prompt generation with GEPA candidate optimization."""
    sig = GeographyQuery(
        question="What are the major rivers in Africa?", region="Sub-Saharan Africa"
    )

    # Test with candidate
    candidate = {
        "signature:GeographyQuery:instructions": "Focus on major waterways and their importance.",
        "signature:GeographyQuery:question:desc": "Geographic inquiry:",
        "signature:GeographyQuery:region:desc": "Area of focus:",
    }

    system_instructions = generate_system_instructions(sig, candidate=candidate)
    assert system_instructions == snapshot("""\
Focus on major waterways and their importance.

Inputs:

- `<question>` (str): Geographic inquiry:
- `<region>` (UnionType[str, NoneType]): Area of focus:\
""")

    user_content = generate_user_content(sig)
    assert len(user_content) == 1
    assert user_content[0] == snapshot("""\
<question>What are the major rivers in Africa?</question>

<region>Sub-Saharan Africa</region>\
""")


def test_signature_agent_rejects_user_prompt_without_history():
    """user_prompt requires message history."""
    test_model = TestModel(custom_output_text="Initial response.")
    agent = Agent(test_model, instructions="Geography expert", name="geo")
    signature_agent = SignatureAgent(
        agent,
        input_type=GeographyQuery,
        output_type=str,
    )
    sig = GeographyQuery(question="What's the capital of Spain?", region="Europe")

    with pytest.raises(ValueError):
        signature_agent.run_signature_sync(sig, user_prompt="Follow-up question?")


def test_signature_agent_followup_uses_custom_prompt():
    """Follow-up runs should relay the provided user prompt."""
    test_model = TestModel(custom_output_text="Follow-up response.")
    agent = Agent(test_model, instructions="Geography expert", name="geo")
    signature_agent = SignatureAgent(
        agent,
        input_type=GeographyQuery,
        output_type=str,
    )
    sig = GeographyQuery(question="What's the capital of Spain?", region="Europe")

    initial_result = signature_agent.run_signature_sync(sig)
    message_history = initial_result.all_messages()

    followup_result = signature_agent.run_signature_sync(
        sig,
        message_history=message_history,
        user_prompt="Can you also list major museums?",
    )

    new_messages = followup_result.new_messages()
    request_messages = [msg for msg in new_messages if isinstance(msg, ModelRequest)]
    assert request_messages
    request = request_messages[0]
    user_parts = [part for part in request.parts if isinstance(part, UserPromptPart)]
    assert user_parts
    first_content = user_parts[0].content
    if isinstance(first_content, str):
        actual_prompt = first_content
    elif isinstance(first_content, list):
        actual_prompt = "".join(str(item) for item in first_content)
    else:
        actual_prompt = str(first_content)
    assert "Can you also list major museums?" in actual_prompt


def test_signature_agent_followup_uses_signature_prompt():
    """Follow-up runs should default to the signature values when no prompt is provided."""
    test_model = TestModel(custom_output_text="Follow-up response.")
    agent = Agent(test_model, instructions="Geography expert", name="geo")
    signature_agent = SignatureAgent(
        agent,
        input_type=GeographyQuery,
        output_type=str,
    )

    sig_initial = GeographyQuery(
        question="What's the capital of Spain?", region="Europe"
    )
    initial_result = signature_agent.run_signature_sync(sig_initial)
    message_history = initial_result.all_messages()

    sig_followup = GeographyQuery(
        question="What's the capital of Germany?", region="Central Europe"
    )

    followup_result = signature_agent.run_signature_sync(
        sig_followup,
        message_history=message_history,
    )

    new_messages = followup_result.new_messages()
    request_messages = [msg for msg in new_messages if isinstance(msg, ModelRequest)]
    assert request_messages
    request = request_messages[0]
    user_parts = [part for part in request.parts if isinstance(part, UserPromptPart)]
    assert user_parts
    first_content = user_parts[0].content
    if isinstance(first_content, str):
        actual_prompt = first_content
    elif isinstance(first_content, list):
        actual_prompt = "".join(str(item) for item in first_content)
    else:
        actual_prompt = str(first_content)
    assert actual_prompt == snapshot("""\
<question>What's the capital of Germany?</question>

<region>Central Europe</region>\
""")


class FormatRequest(BaseModel):
    """Request payload for formatting."""

    text: str = Field(description="Original text that needs formatting")
    style: str = Field(description="Formatting style or tone to apply")


def _build_formatter_agent() -> SignatureAgent[Any, str]:
    test_model = TestModel(custom_output_text="done")
    agent = Agent(
        test_model,
        instructions="You format copy with precision.",
        output_type=str,
        name="formatter",
    )

    @agent.tool_plain
    def format_text(text: str, style: str) -> str:
        """Format content for downstream processing.

        Args:
            text: Raw text to format.
            style: Formatting instructions to apply.
        """

        return f"{style}:{text}"

    return SignatureAgent(
        agent,
        input_type=FormatRequest,
        output_type=str,
        optimize_tools=True,
    )


def test_signature_agent_tool_components_seed():
    """Tool components are exposed when optimization is enabled."""
    signature_agent = _build_formatter_agent()

    seed = extract_seed_candidate(signature_agent)
    assert candidate_texts(seed) == snapshot(
        {
            "instructions": "You format copy with precision.",
            "tool:format_text:description": "Format content for downstream processing.",
            "tool:format_text:param:text": "Raw text to format.",
            "tool:format_text:param:style": "Formatting instructions to apply.",
        }
    )

    component_names = get_component_names(signature_agent)
    assert component_names == snapshot(
        [
            "instructions",
            "tool:format_text:description",
            "tool:format_text:param:text",
            "tool:format_text:param:style",
        ]
    )


def test_signature_agent_tool_candidate_modifies_definitions():
    """Tool candidates modify descriptions during signature runs."""
    signature_agent = _build_formatter_agent()
    test_model = signature_agent.wrapped.model
    assert isinstance(test_model, TestModel)

    sig = FormatRequest(text="hello", style="formal")

    # Baseline run without tool overrides
    _ = signature_agent.run_signature_sync(sig)
    assert test_model.last_model_request_parameters
    tool_defs = test_model.last_model_request_parameters.function_tools
    assert tool_defs == snapshot(
        [
            ToolDefinition(
                name="format_text",
                parameters_json_schema={
                    "additionalProperties": False,
                    "properties": {
                        "text": {
                            "description": "Raw text to format.",
                            "type": "string",
                        },
                        "style": {
                            "description": "Formatting instructions to apply.",
                            "type": "string",
                        },
                    },
                    "required": ["text", "style"],
                    "type": "object",
                },
                description="Format content for downstream processing.",
            )
        ]
    )

    candidate = {
        "tool:format_text:description": "Polish the incoming copy for publication.",
        "tool:format_text:param:text": "Draft prose that needs polish.",
        "tool:format_text:param:style": "Desired finishing style or tone.",
    }

    # Direct candidate override on the signature run
    _ = signature_agent.run_signature_sync(sig, candidate=candidate)
    tool_defs = test_model.last_model_request_parameters.function_tools
    assert tool_defs == snapshot(
        [
            ToolDefinition(
                name="format_text",
                parameters_json_schema={
                    "additionalProperties": False,
                    "properties": {
                        "text": {
                            "description": "Draft prose that needs polish.",
                            "type": "string",
                        },
                        "style": {
                            "description": "Desired finishing style or tone.",
                            "type": "string",
                        },
                    },
                    "required": ["text", "style"],
                    "type": "object",
                },
                description="Polish the incoming copy for publication.",
            )
        ]
    )

    # Revert to baseline when candidate not provided
    _ = signature_agent.run_signature_sync(sig)
    tool_defs = test_model.last_model_request_parameters.function_tools
    assert tool_defs == snapshot(
        [
            ToolDefinition(
                name="format_text",
                parameters_json_schema={
                    "additionalProperties": False,
                    "properties": {
                        "text": {
                            "description": "Raw text to format.",
                            "type": "string",
                        },
                        "style": {
                            "description": "Formatting instructions to apply.",
                            "type": "string",
                        },
                    },
                    "required": ["text", "style"],
                    "type": "object",
                },
                description="Format content for downstream processing.",
            )
        ]
    )

    # Context manager application (emulates GEPA adapter)
    tool_only_candidate = _component_map(
        {
            "tool:format_text:description": "Apply brand voice polishing.",
            "tool:format_text:param:text": "Source text awaiting adjustments.",
        }
    )
    with apply_candidate_to_agent(signature_agent, tool_only_candidate):
        _ = signature_agent.run_signature_sync(sig)
        tool_defs = test_model.last_model_request_parameters.function_tools
        assert tool_defs == snapshot(
            [
                ToolDefinition(
                    name="format_text",
                    parameters_json_schema={
                        "additionalProperties": False,
                        "properties": {
                            "text": {
                                "description": "Source text awaiting adjustments.",
                                "type": "string",
                            },
                            "style": {
                                "description": "Formatting instructions to apply.",
                                "type": "string",
                            },
                        },
                        "required": ["text", "style"],
                        "type": "object",
                    },
                    description="Apply brand voice polishing.",
                )
            ]
        )

    # After context exit, baseline should be restored
    _ = signature_agent.run_signature_sync(sig)
    tool_defs = test_model.last_model_request_parameters.function_tools
    assert tool_defs == snapshot(
        [
            ToolDefinition(
                name="format_text",
                parameters_json_schema={
                    "additionalProperties": False,
                    "properties": {
                        "text": {
                            "description": "Raw text to format.",
                            "type": "string",
                        },
                        "style": {
                            "description": "Formatting instructions to apply.",
                            "type": "string",
                        },
                    },
                    "required": ["text", "style"],
                    "type": "object",
                },
                description="Format content for downstream processing.",
            )
        ]
    )
