from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field
from pydantic_ai_gepa import Signature, SignatureAgent

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, SystemPromptPart
from pydantic_ai.models.test import TestModel


class GeographyQuery(Signature):
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
    signature_agent = SignatureAgent(agent)

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
    assert isinstance(request.parts[0], SystemPromptPart)
    assert request.parts[0].content == snapshot(
        """\
Ask a question about geography.

Inputs

- `<question>` (str): The geography question to ask
- `<region>` (UnionType[str, NoneType]): Specific region to focus on, if applicable\
"""
    )
    assert request.instructions == snapshot("You're an expert in geography.")


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

    signature_agent = SignatureAgent(agent)
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
    assert isinstance(request.parts[0], SystemPromptPart)
    assert request.parts[0].content == snapshot(
        """\
Ask a question about geography.

Inputs

- `<question>` (str): The geography question to ask
- `<region>` (UnionType[str, NoneType]): Specific region to focus on, if applicable\
"""
    )
    assert request.instructions == snapshot("You're an expert in geography.")


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
    signature_agent = SignatureAgent(agent)

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

    signature_agent = SignatureAgent(agent)
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

    signature_agent = SignatureAgent(agent)
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
    user_content = sig.to_user_content()
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

    system_instructions = sig.to_system_instructions(candidate=candidate)
    assert system_instructions == snapshot("""\
Focus on major waterways and their importance.

Inputs

- `<question>` (str): Geographic inquiry:
- `<region>` (UnionType[str, NoneType]): Area of focus:\
""")

    user_content = sig.to_user_content()
    assert len(user_content) == 1
    assert user_content[0] == snapshot("""\
<question>What are the major rivers in Africa?</question>

<region>Sub-Saharan Africa</region>\
""")
