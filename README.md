# pydantic-ai-gepa

[GEPA-driven](https://github.com/gepa-ai/gepa) prompt optimization tools for [pydantic-ai](https://github.com/pydantic/pydantic-ai). The library wraps GEPA's reflective search so agent authors can iterate on prompts, structured inputs, and evaluation metrics without leaving the pydantic-ai ecosystem. This is heavily influenced by [dspy](https://github.com/stanfordnlp/dspy). We've adapted their concept of a structured input spec to work with pydantic-ai's inputs/outputs.

This is still very much an experimental work in progress and depends on this PR awaiting to be merged to pydantic-ai: https://github.com/pydantic/pydantic-ai/pull/2926.

## Highlights

- **High-level optimizer**: `optimize_agent_prompts` coordinates training/validation datasets, scoring callbacks, and reflection-enabled candidate selection.
- **Adapter layer**: `PydanticAIGEPAAdapter` and `SignatureAgent` bridge GEPA's rollout interfaces with pydantic-ai agents and structured outputs.
- **Caching + telemetry**: Optional cache (`CacheManager`, `create_cached_metric`) and logging hooks keep optimization runs reproducible.

## Quick Start

1. Install dependencies: `uv sync --all-extras`
2. Run the sentiment example: `uv run python examples/classification.py`
3. Run the tool-using math example: `uv run python examples/math_tools.py`
4. Inspect outputs under `optimization_results/` to compare candidates and scores.

## Project Layout

- Library code in `src/pydantic_ai_gepa/` (runner, components, caching, signature helpers)
- Tests in `tests/` mirroring module names; run with `uv run pytest`
- Examples in `examples/` for end-to-end GEPA agent workflows

## Contributing

See `AGENTS.md` for coding standards, test expectations, and pull request guidelines before opening changes.
