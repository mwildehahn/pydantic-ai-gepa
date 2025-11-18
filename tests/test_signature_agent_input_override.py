from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from inline_snapshot import snapshot
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai_gepa import SignatureAgent

class SimpleQuery(BaseModel):
    q: str = Field(description="Simple question")

class DetailedQuery(BaseModel):
    question: str = Field(description="Detailed question")
    context: str = Field(description="Context")

def test_signature_agent_override_input_type():
    """Test overriding input_type in run_signature."""
    test_model = TestModel(custom_output_text="Paris")
    
    agent = Agent(
        test_model, 
        instructions="Base instructions", 
        name="test_agent"
    )
    
    # Initialize with SimpleQuery
    sig_agent = SignatureAgent(agent, input_type=SimpleQuery, output_type=str)
    
    # 1. Run with default input type
    sig1 = SimpleQuery(q="Capital?")
    result1 = sig_agent.run_signature_sync(sig1)
    messages1 = result1.all_messages()
    request1 = messages1[0]
    assert isinstance(request1, ModelRequest)
    instructions1 = request1.instructions or ""
    # Check generated instructions from SimpleQuery
    assert "Simple question" in instructions1
    assert "Detailed question" not in instructions1

    # 2. Run with overridden input type
    sig2 = DetailedQuery(question="Capital?", context="Europe")
    result2 = sig_agent.run_signature_sync(sig2, input_type=DetailedQuery)
    messages2 = result2.all_messages()
    request2 = messages2[0]
    assert isinstance(request2, ModelRequest)
    instructions2 = request2.instructions or ""
    
    # Check generated instructions from DetailedQuery
    assert "Detailed question" in instructions2
    assert "Context" in instructions2
    assert "Simple question" not in instructions2

@pytest.mark.asyncio
async def test_signature_agent_override_input_type_async():
    """Test overriding input_type in run_signature async."""
    test_model = TestModel(custom_output_text="Paris")
    agent = Agent(test_model, instructions="Base", name="test_agent")
    sig_agent = SignatureAgent(agent, input_type=SimpleQuery, output_type=str)
    
    sig2 = DetailedQuery(question="Capital?", context="Europe")
    result = await sig_agent.run_signature(sig2, input_type=DetailedQuery)
    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    instructions = request.instructions or ""
    assert "Detailed question" in instructions

@pytest.mark.asyncio
async def test_signature_agent_override_input_type_stream():
    """Test overriding input_type in run_signature_stream."""
    test_model = TestModel(custom_output_text="Paris")
    agent = Agent(test_model, instructions="Base", name="test_agent")
    sig_agent = SignatureAgent(agent, input_type=SimpleQuery, output_type=str)
    
    sig2 = DetailedQuery(question="Capital?", context="Europe")
    
    with capture_run_messages() as messages:
        async with sig_agent.run_signature_stream(sig2, input_type=DetailedQuery) as stream:
            await stream.get_output()
    
    assert len(messages) >= 1
    request = messages[0]
    assert isinstance(request, ModelRequest)
    instructions = request.instructions or ""
    assert "Detailed question" in instructions

def test_signature_agent_input_type_override_parity():
    """Test that input_type works as expected when overridden."""
    # This test now just confirms everything works with required input_type in init
    test_model = TestModel(custom_output_text="Paris")
    agent = Agent(test_model, instructions="Base", name="test_agent")
    
    sig_agent = SignatureAgent(agent, input_type=SimpleQuery, output_type=str)
    
    sig = DetailedQuery(question="Capital?", context="Europe")
    # Overriding input_type
    result = sig_agent.run_signature_sync(sig, input_type=DetailedQuery)
    assert result.output == "Paris"
