from __future__ import annotations

from typing import Any

from inline_snapshot import snapshot
import pytest
import time_machine
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from pydantic_ai_gepa.adapter import (
    ComponentReflectiveDataset,
    SharedReflectiveDataset,
)
from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapter
from pydantic_ai_gepa.gepa_graph.models import CandidateProgram, ComponentValue
from pydantic_ai_gepa.gepa_graph.proposal import InstructionProposalGenerator
from pydantic_ai_gepa.types import DataInstWithPrompt, MetricResult, RolloutOutput


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
    """Create a realistic reflective record with full message history."""
    return {
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {"type": "system_prompt", "role": "system", "content": "You are a helpful assistant."},
                    {"type": "user_prompt", "role": "user", "content": "Hello"},
                ],
            },
            {
                "kind": "response",
                "model_name": "openai:gpt-4",
                "parts": [
                    {"type": "text", "role": "assistant", "content": "Hi"},
                ],
            },
        ],
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
    assert "## Components to update" in prompts[-1]
    assert "### Component: `instructions`" in prompts[-1]
    assert "### Component: `tools`" in prompts[-1]


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
        assert "### Component: `instructions`" in prompt
        assert "### Component: `tools`" not in prompt
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
# Role: Component Optimizer for Student Agent

You are optimizing prompt components for a student agent based on its production performance.

## Context
- A student agent has been running with the configuration shown below
- We've collected traces from real production runs
- Your job is to improve specific components so the student agent performs better

---

## Full student agent configuration

This is the complete configuration the student agent was running with:

**`instructions` given to student:**
```
Seed instructions
```

**`tools` given to student:**
```
Seed tools
```

---

## Production traces from student agent runs

Each trace contains:
- `messages`: Full conversation history with system prompts, user inputs, assistant responses, tool calls, and tool returns
- `tools`: Tool definitions that were available (if any)
- `score`: Performance score (0.0-1.0, higher is better)
- `success`: Whether the run completed successfully
- `feedback`: Evaluator feedback on this specific run

**Use these traces to optimize the components listed below:**

### Trace 1
- **Score:** 0.5
- **Success:** true
- **Feedback:** Needs more detail

- **Messages:**
```json
[
  {
    "kind": "request",
    "parts": [
      {
        "type": "system_prompt",
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "type": "user_prompt",
        "role": "user",
        "content": "Hello"
      }
    ]
  },
  {
    "kind": "response",
    "model_name": "openai:gpt-4",
    "parts": [
      {
        "type": "text",
        "role": "assistant",
        "content": "Hi"
      }
    ]
  }
]
```


### Analysis guidance
- What failure patterns repeat across runs?
- Are components misaligned (e.g., instructions referencing tools that don't exist)?
- Which successful patterns should be preserved or extended?
- What domain knowledge should be codified in the prompts?

---

## Components to update

Rewrite these components as a coordinated update based on the evidence above:

### Component: `instructions`
Current value:
```
Seed instructions
```

### Component: `tools`
Current value:
```
Seed tools
```
""")


@time_machine.travel("2025-01-15 12:00:00 UTC", tick=False)
@pytest.mark.asyncio
async def test_end_to_end_with_real_agent_and_tools() -> None:
    """End-to-end test: normal agent with tools -> adapter -> reflective dataset -> proposal.

    Demonstrates optimizing both instructions AND tool descriptions.
    """

    # Create a model that simulates tool call behavior
    async def weather_model(messages, agent_info):
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            ToolCallPart,
            ToolReturnPart,
        )

        # Check if there's a tool return in the messages
        has_tool_return = any(
            isinstance(msg, ModelRequest) and
            any(isinstance(part, ToolReturnPart) for part in msg.parts)
            for msg in messages
        )

        if has_tool_return:
            # Second call: after tool return, respond with final text
            # Extract the location from the tool return to customize response
            for msg in messages:
                if isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart):
                            # Parse the tool return to get location
                            content = part.content
                            if isinstance(content, str) and ":" in content:
                                location = content.split(":")[0].replace("Weather in", "").strip()
                                return ModelResponse(parts=[TextPart(content=f"The weather in {location} is sunny and 72°F.")])
            return ModelResponse(parts=[TextPart(content="It's sunny and 72°F.")])
        else:
            # First call: extract location from user prompt and call tool
            location = "Unknown"
            for msg in messages:
                if isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if hasattr(part, 'content'):
                            content = str(part.content)
                            if "San Francisco" in content:
                                location = "San Francisco"
                            elif "New York" in content:
                                location = "New York"

            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_weather",
                        args={"location": location},
                        tool_call_id=f"call_{location.replace(' ', '_')}",
                    )
                ],
            )

    model = FunctionModel(function=weather_model)

    # Create an agent with a tool
    agent = Agent(
        model,
        instructions="Use the weather tool to answer questions about weather.",
        output_type=str,
    )

    @agent.tool_plain
    def get_weather(location: str) -> str:
        """Get current weather for a location.

        Args:
            location: The city name to get weather for.
        """
        return f"Weather in {location}: Sunny, 72°F"

    # Set up adapter with a simple metric
    def metric(data_inst: DataInstWithPrompt, output: RolloutOutput[Any]) -> MetricResult:
        # Score based on whether it used the tool
        success = output.success and output.result is not None
        return MetricResult(
            score=0.9 if success else 0.0,
            feedback="Good use of tools" if success else "Failed to use tools",
        )

    adapter = AgentAdapter(agent, metric, optimize_tools=True)

    # Create test data with multiple runs
    test_data = [
        DataInstWithPrompt(
            user_prompt=UserPromptPart(content="What's the weather in San Francisco?"),
            message_history=None,
            metadata={},
            case_id="weather_1",
        ),
        DataInstWithPrompt(
            user_prompt=UserPromptPart(content="What's the weather in New York?"),
            message_history=None,
            metadata={},
            case_id="weather_2",
        ),
    ]

    # Run evaluation with traces
    candidate = adapter.get_components()
    eval_batch = await adapter.evaluate(
        batch=test_data,
        candidate=candidate,
        capture_traces=True,
    )

    # Get tool component names
    tool_components = [name for name in candidate.keys() if name.startswith("tool:")]
    assert tool_components, "Expected tool components to be exposed for plain agents"

    # Build reflective dataset - include instructions + all tool components
    components_to_optimize = ["instructions"] + tool_components
    reflective_data = adapter.make_reflective_dataset(
        candidate=candidate,
        eval_batch=eval_batch,
        components_to_update=components_to_optimize,
    )

    # Convert to CandidateProgram format for the proposal generator
    candidate_program = CandidateProgram(
        idx=0,
        components={
            name: ComponentValue(name=name, text=text)
            for name, text in candidate.items()
        },
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )

    # Capture the prompt that would be sent to the optimizer
    captured_prompt: str | None = None

    async def capture_model(messages, agent_info):
        nonlocal captured_prompt
        captured_prompt = messages[-1].parts[0].content
        # Return a dummy response with updates for all components
        updated = [
            {"component_name": "instructions", "optimized_value": "Improved instructions with tool guidance"}
        ]
        for tool_comp in tool_components:
            updated.append({"component_name": tool_comp, "optimized_value": f"Improved {tool_comp}"})

        import json
        content = json.dumps({"updated_components": updated})
        return ModelResponse(parts=[TextPart(content=content)])

    # Run the proposal generator
    generator = InstructionProposalGenerator()
    model = FunctionModel(function=capture_model)
    result = await generator.propose_texts(
        candidate=candidate_program,
        reflective_data=reflective_data,
        components=components_to_optimize,
        model=model,
    )

    # Verify the result includes updated instructions and tool components
    assert result["instructions"] == "Improved instructions with tool guidance"
    for tool_comp in tool_components:
        assert tool_comp in result
        assert "Improved" in result[tool_comp]

    # Verify the complete prompt with snapshot
    assert captured_prompt == snapshot("""\
# Role: Component Optimizer for Student Agent

You are optimizing prompt components for a student agent based on its production performance.

## Context
- A student agent has been running with the configuration shown below
- We've collected traces from real production runs
- Your job is to improve specific components so the student agent performs better

---

## Full student agent configuration

This is the complete configuration the student agent was running with:

**`instructions` given to student:**
```
Use the weather tool to answer questions about weather.
```

**Tools available to student (JSON Schema):**
```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "location": {
            "description": "The city name to get weather for.",
            "type": "string"
          }
        },
        "required": [
          "location"
        ],
        "type": "object"
      },
      "description": "Get current weather for a location."
    }
  }
]
```

---

## Production traces from student agent runs

Each trace contains:
- `messages`: Full conversation history with system prompts, user inputs, assistant responses, tool calls, and tool returns
- `tools`: Tool definitions that were available (if any)
- `score`: Performance score (0.0-1.0, higher is better)
- `success`: Whether the run completed successfully
- `feedback`: Evaluator feedback on this specific run

**Use these traces to optimize the components listed below:**

### Trace 1: What's the weather in San Francisco?
- **Score:** 0.9
- **Success:** true
- **Feedback:** Good use of tools
- **Assistant Response:** The weather in San Francisco is sunny and 72°F.

- **Messages:**
```json
[
  {
    "kind": "request",
    "parts": [
      {
        "type": "user_prompt",
        "role": "user",
        "content": "What's the weather in San Francisco?",
        "timestamp": "2025-01-15T12:00:00+00:00"
      }
    ]
  },
  {
    "kind": "response",
    "model_name": "function:weather_model:",
    "timestamp": "2025-01-15T12:00:00+00:00",
    "parts": [
      {
        "type": "tool_call",
        "role": "assistant",
        "tool_name": "get_weather",
        "arguments": "{\\"location\\":\\"San Francisco\\"}",
        "tool_call_id": "call_San_Francisco"
      }
    ],
    "usage": {
      "input_tokens": 56,
      "cache_write_tokens": 0,
      "cache_read_tokens": 0,
      "output_tokens": 6,
      "input_audio_tokens": 0,
      "cache_audio_read_tokens": 0,
      "output_audio_tokens": 0,
      "details": {}
    }
  },
  {
    "kind": "request",
    "parts": [
      {
        "type": "tool_return",
        "tool_name": "get_weather",
        "content": "Weather in San Francisco: Sunny, 72\\u00b0F",
        "tool_call_id": "call_San_Francisco",
        "timestamp": "2025-01-15T12:00:00+00:00"
      }
    ]
  },
  {
    "kind": "response",
    "model_name": "function:weather_model:",
    "timestamp": "2025-01-15T12:00:00+00:00",
    "parts": [
      {
        "type": "text",
        "role": "assistant",
        "content": "The weather in San Francisco is sunny and 72\\u00b0F."
      }
    ],
    "usage": {
      "input_tokens": 62,
      "cache_write_tokens": 0,
      "cache_read_tokens": 0,
      "output_tokens": 16,
      "input_audio_tokens": 0,
      "cache_audio_read_tokens": 0,
      "output_audio_tokens": 0,
      "details": {}
    }
  }
]
```
- **Usage:**
```json
{
  "input_tokens": 118,
  "cache_write_tokens": 0,
  "cache_read_tokens": 0,
  "output_tokens": 22,
  "input_audio_tokens": 0,
  "cache_audio_read_tokens": 0,
  "output_audio_tokens": 0,
  "details": {},
  "requests": 2,
  "tool_calls": 1
}
```

### Trace 2: What's the weather in New York?
- **Score:** 0.9
- **Success:** true
- **Feedback:** Good use of tools
- **Assistant Response:** The weather in New York is sunny and 72°F.

- **Messages:**
```json
[
  {
    "kind": "request",
    "parts": [
      {
        "type": "user_prompt",
        "role": "user",
        "content": "What's the weather in New York?",
        "timestamp": "2025-01-15T12:00:00+00:00"
      }
    ]
  },
  {
    "kind": "response",
    "model_name": "function:weather_model:",
    "timestamp": "2025-01-15T12:00:00+00:00",
    "parts": [
      {
        "type": "tool_call",
        "role": "assistant",
        "tool_name": "get_weather",
        "arguments": "{\\"location\\":\\"New York\\"}",
        "tool_call_id": "call_New_York"
      }
    ],
    "usage": {
      "input_tokens": 56,
      "cache_write_tokens": 0,
      "cache_read_tokens": 0,
      "output_tokens": 6,
      "input_audio_tokens": 0,
      "cache_audio_read_tokens": 0,
      "output_audio_tokens": 0,
      "details": {}
    }
  },
  {
    "kind": "request",
    "parts": [
      {
        "type": "tool_return",
        "tool_name": "get_weather",
        "content": "Weather in New York: Sunny, 72\\u00b0F",
        "tool_call_id": "call_New_York",
        "timestamp": "2025-01-15T12:00:00+00:00"
      }
    ]
  },
  {
    "kind": "response",
    "model_name": "function:weather_model:",
    "timestamp": "2025-01-15T12:00:00+00:00",
    "parts": [
      {
        "type": "text",
        "role": "assistant",
        "content": "The weather in New York is sunny and 72\\u00b0F."
      }
    ],
    "usage": {
      "input_tokens": 62,
      "cache_write_tokens": 0,
      "cache_read_tokens": 0,
      "output_tokens": 16,
      "input_audio_tokens": 0,
      "cache_audio_read_tokens": 0,
      "output_audio_tokens": 0,
      "details": {}
    }
  }
]
```
- **Usage:**
```json
{
  "input_tokens": 118,
  "cache_write_tokens": 0,
  "cache_read_tokens": 0,
  "output_tokens": 22,
  "input_audio_tokens": 0,
  "cache_audio_read_tokens": 0,
  "output_audio_tokens": 0,
  "details": {},
  "requests": 2,
  "tool_calls": 1
}
```


### Analysis guidance
- What failure patterns repeat across runs?
- Are components misaligned (e.g., instructions referencing tools that don't exist)?
- Which successful patterns should be preserved or extended?
- What domain knowledge should be codified in the prompts?

---

## Components to update

Rewrite these components as a coordinated update based on the evidence above:

### Component: `instructions`
Current value:
```
Use the weather tool to answer questions about weather.
```

### Component: `tool:get_weather:description`
Current value:
```
Get current weather for a location.
```

### Component: `tool:get_weather:param:location`
Current value:
```
The city name to get weather for.
```
""")
