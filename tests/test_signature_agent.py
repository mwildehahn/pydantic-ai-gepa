from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field
from pydantic_ai_gepa import Signature, SignatureAgent

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel


class GeographyQuery(Signature):
    """Ask a question about geography."""

    question: str = Field(description='The geography question to ask')
    region: str | None = Field(None, description='Specific region to focus on, if applicable')


class GeographyAnswer(BaseModel):
    """Answer to a geography question."""

    answer: str = Field(description='The answer to the question')
    confidence: str = Field(description='Confidence level: high, medium, or low')
    sources: list[str] = Field(default_factory=list, description='Sources of information')


def test_signature_agent_basic():
    """Test basic SignatureAgent functionality."""
    # Create a test model with deterministic responses
    test_model = TestModel(
        custom_output_args=GeographyAnswer(
            answer='Paris',
            confidence='high',
            sources=['Common knowledge'],
        )
    )

    # Create base agent
    agent = Agent(
        test_model,
        instructions="You're an expert in geography.",
        output_type=GeographyAnswer,
        name='geography',
    )

    # Wrap with SignatureAgent
    signature_agent = SignatureAgent(agent)

    # Create a signature instance
    sig = GeographyQuery(question="What's the capital of France?", region='Western Europe')

    # Test sync run
    result = signature_agent.run_signature_sync(sig)

    assert result.output.answer == 'Paris'
    assert result.output.confidence == 'high'
    assert result.output.sources == ['Common knowledge']
    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert isinstance(request.parts[0], UserPromptPart)
    assert request.parts[0].content == snapshot(
        [
            """\
Ask a question about geography.

The geography question to ask
Question: What's the capital of France?

Specific region to focus on, if applicable
Region: Western Europe\
"""
        ]
    )
    assert request.instructions == snapshot("You're an expert in geography.")


def test_signature_agent_with_default_candidate():
    """Test SignatureAgent with default GEPA candidate optimization."""
    test_model = TestModel(
        custom_output_args=GeographyAnswer(
            answer='Berlin',
            confidence='high',
            sources=['Atlas', 'Encyclopedia'],
        )
    )

    agent = Agent(
        test_model,
        instructions="You're an expert in geography.",
        output_type=GeographyAnswer,
        name='geography',
    )

    # Create SignatureAgent with default candidate
    default_candidate = {'instructions': 'You are a world-class geography expert with deep knowledge.'}
    signature_agent = SignatureAgent(agent, default_candidate=default_candidate)

    sig = GeographyQuery(question="What's the capital of Germany?", region='Central Europe')

    # Test with default candidate
    result = signature_agent.run_signature_sync(sig)
    assert result.output.answer == 'Berlin'
    assert result.output.confidence == 'high'
    assert result.output.sources == ['Atlas', 'Encyclopedia']
    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert isinstance(request.parts[0], UserPromptPart)
    assert request.parts[0].content == snapshot(
        [
            """\
Ask a question about geography.

The geography question to ask
Question: What's the capital of Germany?

Specific region to focus on, if applicable
Region: Central Europe\
"""
        ]
    )
    assert request.instructions == snapshot('You are a world-class geography expert with deep knowledge.')


def test_signature_agent_with_override_candidate():
    """Test SignatureAgent with candidate override."""
    test_model = TestModel(
        custom_output_args=GeographyAnswer(
            answer='Rome',
            confidence='high',
            sources=['Historical records'],
        ),
    )

    agent = Agent(
        test_model,
        instructions="You're an expert in geography.",
        output_type=GeographyAnswer,
        name='geography',
    )

    signature_agent = SignatureAgent(agent)
    sig = GeographyQuery(question="What's the capital of Italy?", region='Southern Europe')

    # Test with override candidate
    override_candidate = {
        'signature:GeographyQuery:instructions': 'Focus on European capitals.',
        'signature:GeographyQuery:question:desc': 'The capital city question',
        'instructions': 'Be concise and accurate.',
    }

    result = signature_agent.run_signature_sync(sig, candidate=override_candidate)
    assert result.output.answer == 'Rome'
    assert result.output.confidence == 'high'
    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert isinstance(request.parts[0], UserPromptPart)
    assert request.parts[0].content == snapshot(
        [
            """\
Focus on European capitals.

The capital city question
Question: What's the capital of Italy?

Specific region to focus on, if applicable
Region: Southern Europe\
"""
        ]
    )
    assert request.instructions == snapshot('Be concise and accurate.')


def test_signature_agent_context_manager():
    """Test SignatureAgent with_candidate context manager."""
    test_model = TestModel(
        custom_output_args=GeographyAnswer(
            answer='Madrid',
            confidence='high',
            sources=['Government records'],
        )
    )

    agent = Agent(
        test_model,
        instructions="You're an expert in geography.",
        output_type=GeographyAnswer,
        name='geography',
    )

    signature_agent = SignatureAgent(agent)
    sig = GeographyQuery(question="What's the capital of Spain?")

    candidate = {
        'instructions': 'Be precise about European capitals.',
    }

    # Test with context manager
    with signature_agent.with_candidate(candidate):
        result = signature_agent.run_signature_sync(sig)
        assert result.output.answer == 'Madrid'
        assert result.output.confidence == 'high'
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert isinstance(request.parts[0], UserPromptPart)
        assert request.parts[0].content == snapshot(
            [
                """\
Ask a question about geography.

The geography question to ask
Question: What's the capital of Spain?\
"""
            ]
        )
        assert request.instructions == snapshot('Be precise about European capitals.')


def test_signature_agent_without_output_type():
    """Test SignatureAgent with text output."""
    test_model = TestModel(custom_output_text='The capital of France is Paris.')

    # Create base agent without output_type
    agent = Agent(
        test_model,
        instructions="You're an expert in geography.",
        name='geography',
    )

    # Wrap with SignatureAgent
    signature_agent = SignatureAgent(agent)

    # Create and run with a signature
    sig = GeographyQuery(question="What's the capital of France?")
    result = signature_agent.run_signature_sync(sig)

    assert result.output == 'The capital of France is Paris.'


@pytest.mark.asyncio
async def test_signature_agent_async():
    """Test async execution with SignatureAgent."""
    test_model = TestModel(
        custom_output_args=GeographyAnswer(
            answer='London',
            confidence='high',
            sources=['UK Government'],
        )
    )

    agent = Agent(
        test_model,
        instructions='Geography expert',
        output_type=GeographyAnswer,
        name='geo',
    )

    signature_agent = SignatureAgent(agent)
    sig = GeographyQuery(question="What's the capital of the UK?")

    result = await signature_agent.run_signature(sig)
    assert result.output.answer == 'London'
    assert result.output.confidence == 'high'


@pytest.mark.asyncio
async def test_signature_agent_streaming():
    """Test streaming execution with SignatureAgent."""
    test_model = TestModel(custom_output_text='The capital is Tokyo.')

    agent = Agent(
        test_model,
        instructions='Geography expert',
        name='geo',
    )

    signature_agent = SignatureAgent(agent)
    sig = GeographyQuery(question="What's the capital of Japan?")

    async with signature_agent.run_signature_stream(sig) as stream:
        output = await stream.get_output()
        assert output == snapshot('The capital is Tokyo.')


def test_prompt_generation_from_signature():
    """Test that prompts are correctly generated from signatures."""
    sig = GeographyQuery(question='What are the major rivers in Africa?', region='Sub-Saharan Africa')

    # Test without candidate
    user_content = sig.to_user_content()
    assert len(user_content) == 1
    assert user_content[0] == snapshot("""\
Ask a question about geography.

The geography question to ask
Question: What are the major rivers in Africa?

Specific region to focus on, if applicable
Region: Sub-Saharan Africa\
""")


def test_prompt_generation_with_candidate():
    """Test prompt generation with GEPA candidate optimization."""
    sig = GeographyQuery(question='What are the major rivers in Africa?', region='Sub-Saharan Africa')

    # Test with candidate
    candidate = {
        'signature:GeographyQuery:instructions': 'Focus on major waterways and their importance.',
        'signature:GeographyQuery:question:desc': 'Geographic inquiry:',
        'signature:GeographyQuery:region:desc': 'Area of focus:',
    }

    user_content_optimized = sig.to_user_content(candidate=candidate)
    assert len(user_content_optimized) == 1
    assert user_content_optimized[0] == snapshot("""\
Focus on major waterways and their importance.

Geographic inquiry:
Question: What are the major rivers in Africa?

Area of focus:
Region: Sub-Saharan Africa\
""")
