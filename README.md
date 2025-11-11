# pydantic-ai-gepa

GEPA-driven prompt optimization for [pydantic-ai](https://github.com/pydantic/pydantic-ai) agents. This library provides evolutionary optimization of agent prompts, structured input schemas, and tool descriptions within the pydantic-ai ecosystem.

## About

This library is heavily inspired by and based on [gepa-ai/gepa](https://github.com/gepa-ai/gepa), which pioneered the GEPA (Genetic Evolution with Perspective Aggregation) algorithm for optimizing text-based system components. We've adapted the core concepts for a native integration with pydantic-ai.

### Key Differentiators

While based on gepa-ai's algorithm, this implementation:

- **Native pydantic-ai integration**: Works directly with pydantic-ai agents and structured outputs
- **Async-first**: Built from the ground up for async LLM operations
- **pydantic-graph execution**: Uses [pydantic-graph](https://ai.pydantic.dev/graph/) to manage the optimization workflow with automatic checkpointing and resumption
- **Simplified architecture**: Focused specifically on pydantic-ai use cases, reducing configuration complexity

### New Contributions

This library introduces two novel concepts to pydantic-ai:

**1. SignatureAgent - DSPy-Inspired Structured Inputs**

Inspired by [DSPy's signatures](https://dspy-docs.vercel.app/docs/building-blocks/signatures), we introduce `SignatureAgent` which adds structured `input_type` support to pydantic-ai agents. Similar to how pydantic-ai uses `output_type` for structured outputs, SignatureAgent enables defining agent inputs as rich Pydantic models:

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai_gepa import SignatureAgent

class AnalysisInput(BaseModel):
    """Analyze the provided data and extract insights."""

    data: str = Field(description="The raw data to analyze")
    focus_area: str = Field(description="Which aspect to focus on")
    format: str = Field(description="Output format preference")

# Create base agent
base_agent = Agent(
    model="openai:gpt-4o",
    output_type=str,
)

# Wrap with SignatureAgent to add input_type support
agent = SignatureAgent(
    base_agent,
    input_type=AnalysisInput,
)

# Run with structured input
result = await agent.run_signature(
    AnalysisInput(
        data="...",
        focus_area="performance",
        format="bullet points"
    )
)
```

Like DSPy signatures, the model docstring becomes system instructions, and field descriptions become input specifications. This creates a declarative interface between your data and the agent.

**2. Optimizable Components**

GEPA can optimize multiple aspects of your agent using evolutionary search:

- **System prompts** - Agent instructions and context
- **Signature fields** - Input/output schema descriptions (when using SignatureAgent)
  - Model docstrings → System instructions
  - Field descriptions → Input/output specifications
- **Tool descriptions** - Tool docstrings and parameter schemas (when `optimize_tools=True`)

Each component is treated as an evolvable text field. GEPA uses reflective mutation (LLM-guided improvements) to iteratively improve all components together:

```python
# Optimize agent with SignatureAgent
result = await optimize_agent(
    agent=agent,  # SignatureAgent instance
    trainset=examples,
    metric=metric,
)

# Access all optimized components
print(result.best_candidate.components)
# {
#   "instructions": "...",                           # System prompt
#   "signature:AnalysisInput:instructions": "...",   # Input schema docstring
#   "signature:AnalysisInput:data:desc": "...",      # Field description
#   "signature:AnalysisInput:focus_area:desc": "...",
#   "tool:my_tool:description": "...",               # If optimize_tools=True
#   "tool:my_tool:param_x:description": "...",
#   ...
# }
```

This unifies prompt engineering, schema design, and tool documentation into a single optimization target.

### Credits

This work builds on:

- **[gepa-ai/gepa](https://github.com/gepa-ai/gepa)**: Original GEPA algorithm and evolutionary optimization framework
- **[pydantic-ai](https://github.com/pydantic/pydantic-ai)**: Agent framework and structured output system
- **[pydantic-graph](https://ai.pydantic.dev/graph/)**: Workflow execution and state management
- Conceptually influenced by **[dspy](https://github.com/stanfordnlp/dspy)** for structured optimization patterns

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Run examples
uv run python examples/classification.py
uv run python examples/math_tools.py
```

## How It Works

### GEPA Graph Architecture

This library reimplements the GEPA algorithm on top of pydantic-graph, providing an async-native, checkpointed workflow:

```
┌─────────────────────────────────────────────────────────────┐
│ GEPA Optimization Graph (pydantic-graph)                    │
│                                                              │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│  │  Start   │─────▶│ Evaluate │─────▶│ Continue │         │
│  │  Node    │      │   Node   │      │  or Stop │         │
│  └──────────┘      └──────────┘      └─────┬────┘         │
│                           ▲                 │              │
│                           │                 ▼              │
│                    ┌──────────┐      ┌──────────┐         │
│                    │  Merge   │◀─────│  Reflect │         │
│                    │  Node    │      │   Node   │         │
│                    └──────────┘      └──────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Node Responsibilities:**

- **StartNode** - Extract seed candidate from agent, initialize state
- **EvaluateNode** - Run validation set evaluation (parallel), update Pareto fronts
- **ContinueNode** - Check stopping conditions, decide next action (reflect/merge/stop)
- **ReflectNode** - Sample minibatch, analyze failures, propose improvements via LLM
- **MergeNode** - Genetic crossover of successful candidates (when enabled)

**Key Benefits:**

- **Async-native** - All LLM operations use async/await for better concurrency
- **Parallel evaluation** - Validate multiple examples concurrently (~10x speedup)
- **Automatic checkpointing** - Every node transition saved, resume from any point
- **Observable** - Full visibility into optimization state at each step
- **Type-safe** - Pydantic models for all state and configuration

### Optimization Process

GEPA optimizes agents through evolutionary search:

1. **Evaluate** - Score candidate prompts on validation examples (parallel)
2. **Reflect** - Use LLM to analyze failures and propose improvements
3. **Merge** - Combine successful strategies (genetic crossover)
4. **Repeat** - Iterate until convergence or budget exhausted

The optimization maintains:

- Pareto front of best-performing candidates per validation instance
- Genealogy tracking (parent-child relationships)
- LLM result caching to avoid redundant API calls
- Configurable stopping conditions (max evaluations, iterations, or custom)

## Example

### Basic Optimization

```python
from pydantic_ai_gepa import optimize_agent
from pydantic_ai import Agent

# Define your agent
agent = Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
)

# Define evaluation metric
def metric(input_data, output) -> float:
    # Return 0.0-1.0 score
    return score

# Optimize
result = await optimize_agent(
    agent=agent,
    trainset=training_examples,
    metric=metric,
    config=GepaConfig(max_evaluations=100),
)

print(f"Best prompt: {result.best_candidate.system_prompt}")
print(f"Best score: {result.best_score}")
```

### With Structured Inputs (SignatureAgent Optimization)

```python
from pydantic import BaseModel, Field
from pydantic_ai_gepa import optimize_agent, SignatureAgent
from pydantic_ai import Agent

# Define structured input
class SentimentInput(BaseModel):
    """Analyze the sentiment of the given text."""

    text: str = Field(description="The text to analyze for sentiment")
    context: str | None = Field(
        default=None,
        description="Additional context about the text"
    )

# Create base agent
base_agent = Agent(
    model="openai:gpt-4o",
    output_type=str,
)

# Wrap with SignatureAgent to add input_type
agent = SignatureAgent(
    base_agent,
    input_type=SentimentInput,
)

# GEPA will optimize:
# - The class docstring ("Analyze the sentiment...")
# - Each field description
# - How they work together

result = await optimize_agent(
    agent=agent,
    trainset=examples,  # List[SentimentInput]
    metric=sentiment_metric,
)

# Access optimized signature components
optimized_instructions = result.best_candidate.components[
    "signature:SentimentInput:instructions"
]
optimized_text_desc = result.best_candidate.components[
    "signature:SentimentInput:text:desc"
]
```

## Project Structure

```
src/pydantic_ai_gepa/
├── runner.py          # Main optimize_agent entry point
├── components/        # GEPA optimization components
├── caching/          # LLM result caching
├── signature.py      # Agent signature adapters
└── ...

examples/             # Example optimization workflows
tests/                # Test suite
```

## Core Concepts

### Algorithm Reference

For detailed algorithm specifications, see:

- **[docs/gepa.md](docs/gepa.md)** - Complete GEPA algorithm documentation
- **[gepa-ai/gepa docs](https://github.com/gepa-ai/gepa)** - Original implementation and research

### Implementation Patterns

This library uses pydantic-graph for workflow execution. For implementation details:

- **[pydantic-graph docs](https://ai.pydantic.dev/graph/)** - Graph execution patterns, state management, checkpointing
- **[pydantic-ai docs](https://ai.pydantic.dev/)** - Agent framework and structured outputs

Key patterns we use:

- **Class-based nodes**: Each GEPA phase (Evaluate, Reflect, Generate, Merge) is a graph node
- **Shared state**: Single `GepaState` dataclass tracks all candidates, scores, and Pareto fronts
- **Signature components**: Input schemas (docstrings, field descriptions) treated as optimizable components
- **Automatic checkpointing**: Graph snapshots enable resumption after crashes
- **Parallel evaluation**: Concurrent LLM calls within nodes using `asyncio.gather()`
- **Result caching**: LLM outputs cached per (input, candidate) pair

### Async Architecture

All LLM operations are async-native:

- Uses `async/await` throughout
- Concurrent evaluation of validation examples
- Parallel reflection across multiple components
- Rate-limited to respect API constraints

## Configuration

```python
from pydantic_ai_gepa import GepaConfig

config = GepaConfig(
    # Stopping conditions
    max_evaluations=200,
    max_iterations=50,

    # Reflection settings
    reflection_model="openai:gpt-4o",
    minibatch_size=5,

    # Parallelism
    max_concurrent_evaluations=10,

    # Merging
    use_merge=True,
    merges_per_accept=2,

    # Strategy selection
    candidate_selector="pareto",  # or "best", "epsilon_greedy"
    component_selector="round_robin",  # or "all"
)
```

## Advanced Features

### Checkpointing & Resumption

```python
# Optimization automatically checkpoints
result = await optimize_agent(
    agent=agent,
    trainset=trainset,
    metric=metric,
    checkpoint_dir="./runs/",
)

# Resume from checkpoint
result = await optimize_agent(
    agent=agent,
    trainset=trainset,
    metric=metric,
    resume_from="./runs/run_123.json",
)
```

### Custom Metrics

```python
from pydantic_ai_gepa import MetricResult

def custom_metric(input_data, output) -> MetricResult:
    """Metric with score and feedback."""
    score = evaluate_output(output)
    feedback = generate_feedback(input_data, output) if score < 1.0 else None

    return MetricResult(score=score, feedback=feedback)
```

### Result Caching

```python
from pydantic_ai_gepa import CacheManager

cache = CacheManager(
    cache_dir=".gepa_cache",
    enabled=True,
)

result = await optimize_agent(
    agent=agent,
    trainset=trainset,
    metric=metric,
    cache_manager=cache,
)
# Second run reuses cached LLM results
```

## Experimental Status

This library is experimental and depends on:

- **pydantic-ai PR #2926** (not yet merged)

Expect API changes as both pydantic-ai and this library evolve.

## Contributing

See `AGENTS.md` for coding standards and contribution guidelines.

## License

MIT License - see LICENSE file for details.

## References

- **GEPA Algorithm**: [gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- **pydantic-ai**: [ai.pydantic.dev](https://ai.pydantic.dev/)
- **pydantic-graph**: [ai.pydantic.dev/graph/](https://ai.pydantic.dev/graph/)
- **DSPy**: [dspy docs](https://dspy-docs.vercel.app/)
