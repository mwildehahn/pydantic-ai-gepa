# Problem: search_examples Tool Optimization

## Issue

The `search_examples` tool optimization was overly complicated, with the tool being created in 4 different places:

1. **Pre-registration in `evaluate()`** (agent_adapter.py lines 722-738) - Created tool just to register with optimizer
2. **Agent invocation in `AgentAdapter._invoke_agent()`** (agent_adapter.py lines 1265-1271)
3. **Agent invocation in `SignatureAgentAdapter._invoke_agent()`** (agent_adapter.py lines 1351-1357)
4. **Reflection visibility in `instruction.py`** (lines 465-483) - Fallback to show tool to reflection model

This violated the principle of single source of truth and made the code harder to maintain.

## Simplified Approach

### Key Insight

The `ToolOptimizationManager._prepare_wrapper` hook (tool_components.py line 325) calls `self._catalog.ingest(prepared)` which captures **any tool that passes through** - no filter needed. The `allowed_tools` filter only affects which tools have their optimized descriptions **applied from candidates**.

Since evaluation always runs *before* reflection, and `_collect_tools()` extracts tools from the `reflective_data` traces, the tool will naturally appear in traces by the time reflection sees them.

### Solution

1. **Always include the toolset** when `example_bank is not None` (not just when it has examples)
2. **Handle empty banks gracefully** in the tool itself (return friendly message)
3. **Remove pre-registration code** - tool captured via normal `_prepare_wrapper` flow
4. **Remove fallback in instruction.py** - tool appears naturally in traces

## Implementation Progress

- [x] **1. Empty bank handling** (`student_tools.py`)
- [x] **2. Always include toolset** (`agent_adapter.py` - both `_invoke_agent` methods)
- [x] **3. Remove pre-registration** (`agent_adapter.py` - `evaluate()` method)
- [x] **4. Remove fallback** (`instruction.py`)
- [x] **5. Run tests and verify** (184 tests pass)

## Changes

### 1. Empty bank handling (`student_tools.py`) - DONE

```python
def search_examples(query: str) -> str:
    if len(bank) == 0:
        return "No examples have been added to the example bank yet."
    results = bank.search(query, k=k)
    # ... rest of implementation
```

### 2. Always include toolset (`agent_adapter.py`) - DONE

Changed both `_invoke_agent` methods:

```python
# Before:
if example_bank is not None and len(example_bank) > 0:

# After:
if example_bank is not None:
```

### 3. Remove pre-registration (`agent_adapter.py`) - DONE

Removed lines 719-738 from `evaluate()`.

### 4. Remove fallback (`instruction.py`) - DONE

Removed lines 464-483 and the now-unused `create_example_search_tool` import.

## Files Changed

| File | Change | Status |
|------|--------|--------|
| `src/pydantic_ai_gepa/gepa_graph/proposal/student_tools.py` | Handle empty bank in search function | Done |
| `src/pydantic_ai_gepa/adapters/agent_adapter.py` | Always include toolset in `_invoke_agent` | Done |
| `src/pydantic_ai_gepa/adapters/agent_adapter.py` | Remove pre-registration in `evaluate()` | Done |
| `src/pydantic_ai_gepa/gepa_graph/proposal/instruction.py` | Remove fallback code | Done |

## How It Works Now

```
1. evaluate() called
   ↓
2. _invoke_agent() creates toolset (if example_bank is not None)
   ↓
3. agent.run(toolsets=[search_tool])
   ↓
4. _prepare_wrapper captures tool via _catalog.ingest()
   ↓
5. Tool appears in agent run traces
   ↓
6. propose() called with reflective_data containing traces
   ↓
7. _collect_tools() finds search_examples in traces
   ↓
8. Reflection model sees tool naturally
```

## Benefits

1. **Simpler code** - Removed ~35 lines of special-case code
2. **Single source of truth** - Tool created in one place only
3. **Standard path** - Tool captured via same mechanism as all other tools
4. **Better UX** - Friendly message when bank is empty instead of missing tool
5. **Consistent behavior** - Tool always available when bank is configured

## Note on `allowed_tools`

The tool will be captured regardless of `allowed_tools`. For optimization to be applied, `search_examples` must be in `allowed_tools`. This happens automatically when `optimize_tools=True` (all tools allowed). For `optimize_tools={"specific_tools"}`, users should include `"search_examples"` explicitly if they want it optimized.
