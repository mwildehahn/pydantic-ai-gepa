# GEPA Refactor: Native pydantic-graph Integration

## Executive Summary

This document outlines a refactor to replace the current `gepa.optimize()` integration with a native pydantic-graph-based implementation at `src/pydantic_ai_gepa/gepa_graph/`. This will provide:

- **Async-native workflow** with step-by-step control
- **Automatic checkpointing** at every optimization step
- **Better observability** into the optimization process
- **Cleaner architecture** leveraging pydantic-graph patterns
- **Type-safe state management** with Pydantic models
- **Simplified API** with fewer configuration parameters
- **~10x performance improvement** through parallelism

## Motivation

### Current Approach: Adapter to External Library

```python
# Current: Wraps external gepa library
result = await optimize_agent(
    agent=agent,
    training_set=training_set,
    metric=metric,
    # ... 40+ configuration parameters
)
# Black box - no visibility into steps, hard to customize
```

**Problems:**

1. **40+ parameters** with complex interactions
2. **Black box execution** - can't inspect intermediate states
3. **String-focused** - optimizes `dict[str, str]` instead of structured types
4. **Complex state management** - 15+ tightly coupled mutable fields
5. **No async support** - synchronous evaluation loop
6. **No parallelism** - ~10x slower than it could be
7. **Hard to extend** - rigid proposer/selector architecture

### Desired Approach: Native pydantic-graph

```python
# Future: Native async-first with graph execution (matches pydantic-ai pattern)
graph = create_gepa_graph(adapter=adapter, config=config)
state = GepaState(config=config, training_set=training_set, validation_set=validation_set)
ctx = GraphRunContext(state=state, deps=deps)

async with graph.iter(ctx=ctx, start_node=StartNode()) as run:
    async for node in run:
        # Graph framework auto-executes each node
        # Full visibility into each optimization step
        print(f"Node: {node.__class__.__name__}")
        print(f"  Iteration: {state.iteration}")
        print(f"  Current best score: {state.best_score if state.best_candidate_idx else 'N/A'}")
        print(f"  Candidates explored: {len(state.candidates)}")

        # Can pause, inspect, or intervene at any point
        if state.best_candidate_idx is not None:
            best = state.candidates[state.best_candidate_idx]
            if best.avg_validation_score > 0.95:
                print("Reached target score!")
                break
```

**Benefits:**

1. **Step-by-step control** - pause, inspect, resume at any point
2. **Automatic checkpointing** - every step saved for resumability
3. **Async-native** - parallel evaluation, better LLM utilization
4. **Type-safe** - Pydantic models for all state
5. **Observable** - full visibility into optimization process after each node
6. **Extensible** - easy to add custom nodes or modify flow
7. **~10x faster** - parallel evaluation of validation sets

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│ GepaGraph (pydantic-graph Graph instance)                   │
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

Each node:
- Operates on shared GepaState (Pydantic model)
- Returns next node instance based on logic
- Automatically checkpointed before/after execution
- Can be async (for LLM calls, parallel evaluation)
```

### State Model

```python
class GepaState(BaseModel):
    """Shared state across all GEPA nodes."""

    # Core data
    iteration: int = -1
    candidates: list[CandidateProgram] = Field(default_factory=list)

    # Tracking
    pareto_front: dict[str, ParetoFrontEntry] = Field(default_factory=dict)
    genealogy: list[Genealogy] = Field(default_factory=list)

    # Execution control
    last_accepted: bool = False
    merge_scheduled: int = 0
    stopped: bool = False
    stop_reason: str | None = None

    # Budget tracking
    total_evaluations: int = 0
    full_validations: int = 0

    # Results
    best_candidate_idx: int | None = None

    # Configuration (immutable during run)
    config: GepaConfig

    # Data loaders (not serialized)
    training_set: list[DataInst] = Field(exclude=True)
    validation_set: list[DataInst] = Field(exclude=True)
```

**Key improvements over current gepa.GEPAState:**

- ✅ Structured types instead of loose dicts
- ✅ Pydantic validation on all fields
- ✅ Clear separation of mutable vs immutable
- ✅ Easier to checkpoint/restore
- ✅ Type-safe access throughout

### Node Definitions

#### 1. StartNode

**Purpose:** Initialize the optimization, evaluate seed candidate

```python
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

@dataclass
class StartNode(BaseNode[GepaState, GepaDeps, None]):
    """Initialize GEPA optimization."""

    async def run(self, ctx: GraphRunContext[GepaState]) -> EvaluateNode:
        """Set up initial state and trigger seed evaluation."""
        state = ctx.state

        # Create seed candidate from agent
        seed_candidate = self._extract_seed_candidate(ctx)
        state.candidates.append(seed_candidate)

        # Mark for evaluation
        state.iteration = 0

        return EvaluateNode()

    def _extract_seed_candidate(self, ctx) -> CandidateProgram:
        """Extract seed candidate from agent and signature."""
        # Implementation
        pass
```

**Returns:** `EvaluateNode()` → Evaluate the seed candidate

#### 2. EvaluateNode

**Purpose:** Run evaluation on validation set (WITH PARALLELISM)

````python
@dataclass
class EvaluateNode(BaseNode[GepaState, GepaDeps, None]):
    """Evaluate a candidate on the validation set."""

    async def run(self, ctx: GraphRunContext[GepaState]) -> ContinueNode:
        """Evaluate the most recent candidate."""
        state = ctx.state
        candidate = state.candidates[-1]

        # Get validation batch (could be sparse or full)
        validation_batch = self._get_validation_batch(state)

        # Evaluate asynchronously IN PARALLEL
        results = await ctx.deps.evaluator.evaluate_batch(
            candidate=candidate,
            batch=validation_batch,
            adapter=ctx.deps.adapter,
            max_concurrent=state.config.max_concurrent_evaluations,
        )

        # Update candidate with scores
        candidate.validation_scores = results.scores
        candidate.validation_outputs = results.outputs

        # Update Pareto front (per-instance fronts, not global)
        # Current GEPA implementation:
        # - For each validation instance (DataId):
        #   - If new score > prev best: replace front with {candidate_idx}
        #   - If new score == prev best: add candidate_idx to front set
        #   - Track best outputs for each instance (optional)
        # - Supports sparse evaluation: only updates instances that were evaluated
        ctx.deps.pareto_manager.update_fronts(state, candidate.idx, results)

        # Update best candidate
        self._update_best_candidate(state)

        # Increment counters
        state.total_evaluations += len(validation_batch)
        state.full_validations += 1

        return ContinueNode()

    def _get_validation_batch(self, state: GepaState) -> list[DataInst]:
        """Get validation instances to evaluate.

        Current GEPA implementation details:
        - Uses EvaluationPolicy protocol to select which validation IDs to evaluate
        - FullEvaluationPolicy (default) returns all validation IDs
        - State supports sparse evaluation via dict[DataId, float] storage
        - Could implement IncrementalEvaluationPolicy or AdaptiveEvaluationPolicy

        Future sparse strategies could include:
        - Evaluate only on examples where candidate differs from parent
        - Adaptive sampling based on uncertainty
        - Stratified sampling across difficulty levels
        """
        # For now, always evaluate on full validation set
        # Policy-based sparse evaluation will be added later
        return state.validation_set

    def _update_best_candidate(self, state: GepaState) -> None:
        """Update best_candidate_idx based on scores.

        Current GEPA implementation:
        - Iterates through all candidates' validation scores
        - Computes average score from sparse dict[DataId, float]
        - Selects candidate with highest average (ties broken by coverage)
        - This is handled by val_evaluation_policy.get_best_program()

        Implementation:
        ```python
        best_idx, best_score, best_coverage = -1, float("-inf"), -1
        for idx, candidate in enumerate(state.candidates):
            scores = candidate.validation_scores  # dict[DataId, float]
            if not scores:
                continue
            coverage = len(scores)
            avg = sum(scores.values()) / coverage
            if avg > best_score or (avg == best_score and coverage > best_coverage):
                best_score = avg
                best_idx = idx
                best_coverage = coverage
        state.best_candidate_idx = best_idx
        ```
        """
        # Implementation would mirror FullEvaluationPolicy.get_best_program()
        pass
````

**Returns:** `ContinueNode()` → Always go to continue/decision node

#### 3. ContinueNode

**Purpose:** Decide whether to continue, merge, or stop

```python
@dataclass
class ContinueNode(BaseNode[GepaState, GepaDeps, None]):
    """Decide next step in optimization."""

    def run(self, ctx: GraphRunContext[GepaState]) -> ReflectNode | MergeNode | End[GepaResult]:
        """Determine next action based on state."""
        state = ctx.state

        # Check stopping conditions
        if self._should_stop(state):
            state.stopped = True
            result = self._create_result(state)
            raise End(result)

        # Increment iteration
        state.iteration += 1

        # Check if merge is scheduled
        if (state.config.use_merge and
            state.merge_scheduled > 0 and
            state.last_accepted):
            state.merge_scheduled -= 1
            return MergeNode()

        # Default: reflective mutation
        return ReflectNode()

    def _should_stop(self, state: GepaState) -> bool:
        """Check all stopping conditions."""
        if state.total_evaluations >= state.config.max_evaluations:
            state.stop_reason = "Max evaluations reached"
            return True

        if state.config.max_iterations and state.iteration >= state.config.max_iterations:
            state.stop_reason = "Max iterations reached"
            return True

        return False

    def _create_result(self, state: GepaState) -> GepaResult:
        """Create final result object."""
        # Implementation
        pass
```

**Returns:**

- `ReflectNode()` → Do reflective mutation
- `MergeNode()` → Do merge operation
- `raise End(result)` → Stop optimization

#### 4. ReflectNode

**Purpose:** Reflective mutation - propose improvements based on execution traces

````python
@dataclass
class ReflectNode(BaseNode[GepaState, GepaDeps, None]):
    """Propose candidate improvements through reflection."""

    async def run(self, ctx: GraphRunContext[GepaState]) -> EvaluateNode | ContinueNode:
        """Generate and test a new candidate via reflection."""
        state = ctx.state
        deps = ctx.deps

        # 1. Select candidate to improve
        # Current GEPA implementation:
        # - ParetoCandidateSelector (default):
        #   - Find non-dominated programs via remove_dominated_programs()
        #   - Count frequency of each program in Pareto fronts
        #   - Sample weighted by frequency (more frequent = more likely)
        #   - Programs that excel on many instances are selected more often
        # - CurrentBestCandidateSelector (alternative):
        #   - Always select program with highest average validation score
        #   - Deterministic, greedy approach
        parent_idx = deps.candidate_selector.select(state)
        parent = state.candidates[parent_idx]

        # 2. Sample minibatch
        # Current GEPA implementation (EpochShuffledBatchSampler):
        # - Shuffle all training IDs at start of each epoch
        # - Pad to multiple of minibatch_size with least-frequent IDs
        # - Slice shuffled list: [iteration * batch_size : (iteration + 1) * batch_size]
        # - Deterministic via RNG seed
        # - Ensures all examples seen once per epoch, balanced sampling
        minibatch = deps.batch_sampler.sample(state.training_set, state, state.config.minibatch_size)

        # 3. Evaluate with trajectory capture (PARALLEL)
        eval_results = await deps.evaluator.evaluate_batch(
            candidate=parent,
            batch=minibatch,
            adapter=deps.adapter,
            max_concurrent=state.config.max_concurrent_evaluations,
            capture_traces=True,
        )

        # 4. Check if perfect score (optional skip)
        if self._should_skip_perfect(eval_results, state.config):
            state.last_accepted = False
            return ContinueNode()

        # 5. Select components to update
        # Current GEPA implementation:
        # - RoundRobinReflectionComponentSelector (default):
        #   - Track next component index per candidate in state
        #   - Select single component, increment index (mod num_components)
        #   - Ensures all components get updated fairly over time
        #   - Returns list with single component name
        # - AllReflectionComponentSelector (alternative):
        #   - Returns all component names (update everything)
        #   - More aggressive, requires more LLM calls
        #   - Useful for tightly coupled components
        components = deps.component_selector.select(state, parent_idx)

        # 6. Build reflective dataset
        reflective_data = self._build_reflective_dataset(
            eval_results=eval_results,
            components=components,
        )

        # 7. Propose new texts via LLM (PARALLEL)
        new_texts = await self._propose_texts_parallel(
            parent=parent,
            reflective_data=reflective_data,
            components=components,
            model=deps.reflection_model,
        )

        # 8. Create new candidate
        new_candidate = self._create_candidate(
            parent=parent,
            new_texts=new_texts,
            parent_indices=[parent_idx],
            state=state,
        )

        # 9. Evaluate new candidate on same minibatch (PARALLEL)
        new_results = await deps.evaluator.evaluate_batch(
            candidate=new_candidate,
            batch=minibatch,
            adapter=deps.adapter,
            max_concurrent=state.config.max_concurrent_evaluations,
            capture_traces=False,
        )

        # 10. Accept if improved (sum of scores)
        if sum(new_results.scores) > sum(eval_results.scores):
            state.candidates.append(new_candidate)
            state.last_accepted = True
            state.merge_scheduled += state.config.merges_per_accept
            return EvaluateNode()  # Evaluate on full validation set
        else:
            state.last_accepted = False
            return ContinueNode()  # Try again

    def _should_skip_perfect(self, results, config) -> bool:
        """Check if should skip due to perfect scores.

        Current GEPA implementation:
        - If skip_perfect_score is True and all scores >= perfect_score
        - Skip reflection iteration (no room for improvement)
        - This saves LLM calls when minibatch is already perfect
        """
        if not config.skip_perfect_score:
            return False
        return all(score >= config.perfect_score for score in results.scores)

    def _build_reflective_dataset(self, eval_results, components) -> dict[str, list[dict]]:
        """Build reflective dataset from trajectories.

        Current GEPA implementation:
        - Delegates to adapter.make_reflective_dataset()
        - Takes: candidate program, evaluation results (with trajectories), component names
        - Returns: dict[component_name -> list[reflection_records]]
        - Each reflection record contains:
          - Input example
          - Trajectory/execution trace
          - Expected output
          - Actual output
          - Score/feedback
        - Format is adapter-specific (depends on task domain)

        Example for DSPy adapter:
        ```python
        reflective_dataset = {
            "system_prompt": [
                {
                    "input": "What is 2+2?",
                    "expected": "4",
                    "actual": "5",
                    "score": 0.0,
                    "trace": {...},
                },
                ...
            ],
            "few_shot_examples": [...]
        }
        ```
        """
        # Implementation delegates to adapter
        pass

    async def _propose_texts_parallel(
        self,
        parent: CandidateProgram,
        reflective_data: dict[str, list[dict]],
        components: list[str],
        model,
    ) -> dict[str, str]:
        """Propose new texts for all components in parallel.

        Current GEPA implementation (sequential):
        - For each component to update:
          - Get current text
          - Get reflective dataset for that component
          - Call InstructionProposalSignature.run() with LLM
          - Extract new text from LLM response
        - All LLM calls are sequential (not parallelized)

        New parallel implementation:
        - Launch all LLM calls concurrently
        - Use asyncio.gather() to wait for all
        - ~Nx speedup for N components

        Reflection prompt format:
        ```
        You are given a current instruction and a dataset with feedback.

        Current instruction: {current_instruction}

        Dataset with feedback:
        {examples with inputs, outputs, scores}

        Propose a new instruction that addresses the failures.
        ```
        """
        async def propose_for_component(comp: str) -> tuple[str, str]:
            # Build reflection prompt
            current_text = parent.components[comp].text
            feedback_data = reflective_data[comp]

            # Call LLM (via adapter or InstructionProposalSignature)
            new_text = await self._call_reflection_llm(
                current_text=current_text,
                feedback_data=feedback_data,
                model=model,
            )
            return comp, new_text

        # Parallelize LLM calls (~Nx speedup)
        tasks = [propose_for_component(c) for c in components]
        results = await asyncio.gather(*tasks)
        return dict(results)

    def _create_candidate(
        self,
        parent: CandidateProgram,
        new_texts: dict[str, str],
        parent_indices: list[int],
        state: GepaState,
    ) -> CandidateProgram:
        """Create new candidate from parent and new texts."""
        # Implementation
        pass
````

**Returns:**

- `EvaluateNode()` → If accepted, evaluate on full validation set
- `ContinueNode()` → If rejected, try again

#### 5. MergeNode

**Purpose:** Genetic crossover - combine successful candidates

````python
@dataclass
class MergeNode(BaseNode[GepaState, GepaDeps, None]):
    """Merge two candidates via genetic crossover."""

    async def run(self, ctx: GraphRunContext[GepaState]) -> EvaluateNode | ContinueNode:
        """Attempt to merge two Pareto front candidates."""
        state = ctx.state
        deps = ctx.deps

        # 1. Find merge candidates from Pareto front
        # Current GEPA implementation:
        # - remove_dominated_programs() iteratively removes programs that are
        #   dominated across ALL validation instances
        # - A program is dominated if for every instance it's on the Pareto front,
        #   there exists another program also on that front with higher score
        # - Returns list of non-dominated program indices (dominators)
        # - These are the "best" candidates worthy of merging
        dominators = deps.pareto_manager.find_dominators(state)

        if len(dominators) < 2:
            state.last_accepted = False
            return ContinueNode()

        # 2. Sample two candidates
        parent1_idx, parent2_idx = self._sample_merge_pair(dominators)

        # 3. Find common ancestor
        ancestor_idx = self._find_common_ancestor(
            state=state,
            idx1=parent1_idx,
            idx2=parent2_idx,
        )

        if ancestor_idx is None:
            state.last_accepted = False
            return ContinueNode()

        # 4. Build merged candidate
        merged_candidate = self._build_merged_candidate(
            state=state,
            parent1_idx=parent1_idx,
            parent2_idx=parent2_idx,
            ancestor_idx=ancestor_idx,
        )

        # 5. Check if already explored
        if self._is_duplicate(merged_candidate, state):
            state.last_accepted = False
            return ContinueNode()

        # 6. Select stratified subsample
        subsample = self._select_merge_subsample(
            state=state,
            parent1_idx=parent1_idx,
            parent2_idx=parent2_idx,
        )

        # 7. Evaluate merged candidate (PARALLEL)
        merged_results = await deps.evaluator.evaluate_batch(
            candidate=merged_candidate,
            batch=subsample,
            adapter=deps.adapter,
            max_concurrent=state.config.max_concurrent_evaluations,
        )

        # 8. Get parent scores on same subsample
        parent1_scores = self._get_subsample_scores(state, parent1_idx, subsample)
        parent2_scores = self._get_subsample_scores(state, parent2_idx, subsample)

        # 9. Accept if merged >= max(parent1, parent2)
        if sum(merged_results.scores) >= max(sum(parent1_scores), sum(parent2_scores)):
            state.candidates.append(merged_candidate)
            state.last_accepted = True
            return EvaluateNode()
        else:
            state.last_accepted = False
            return ContinueNode()

    def _sample_merge_pair(self, dominators: list[int]) -> tuple[int, int]:
        """Sample two candidates from dominators.

        Current GEPA implementation:
        - Takes list of non-dominated program indices
        - Randomly samples 2 different programs
        - Returns (idx1, idx2) where idx1 < idx2 (ordered)

        Implementation:
        ```python
        if len(dominators) < 2:
            return None
        idx1, idx2 = random.sample(dominators, 2)
        if idx2 < idx1:
            idx1, idx2 = idx2, idx1
        return idx1, idx2
        ```
        """
        pass

    def _find_common_ancestor(self, state, idx1, idx2) -> int | None:
        """Find common ancestor for merge.

        Current GEPA implementation:
        - Walk genealogy graph to find all ancestors of idx1 and idx2
        - Find intersection (common ancestors)
        - Filter out ancestors that are:
          - Already merged (tracked in merges_performed)
          - Better than either descendant (prevents regression)
          - Don't have desirable predictors (no divergence to merge)
        - Select ancestor weighted by aggregate score
        - Returns ancestor index or None if no valid ancestor found

        Desirable predictor check:
        - At least one component where:
          - ancestor == parent1 AND parent1 != parent2, OR
          - ancestor == parent2 AND parent1 != parent2
        - This ensures there's something to merge (divergence from ancestor)

        Implementation uses BFS/DFS to traverse parent_program_for_candidate.
        """
        pass

    def _build_merged_candidate(self, state, parent1_idx, parent2_idx, ancestor_idx) -> CandidateProgram:
        """Build merged candidate.

        Current GEPA implementation merges components as follows:
        For each component (predictor):
        1. If ancestor == parent1 AND parent1 != parent2:
           → Take parent2's value (parent2 evolved, use it)
        2. If ancestor == parent2 AND parent1 != parent2:
           → Take parent1's value (parent1 evolved, use it)
        3. If ancestor != parent1 AND ancestor != parent2:
           → Both evolved: take from parent with higher score
           → If scores equal: randomly choose
        4. If parent1 == parent2:
           → Both same: take either (use parent1)

        This implements "genetic crossover" that combines beneficial mutations
        from both parents while preserving unevolved components.

        Example:
        ```
        Ancestor: {"system": "A", "user": "B"}
        Parent1:  {"system": "A", "user": "C"}  (evolved user)
        Parent2:  {"system": "D", "user": "B"}  (evolved system)
        Merged:   {"system": "D", "user": "C"}  (best of both)
        ```
        """
        pass

    def _select_merge_subsample(self, state, parent1_idx, parent2_idx) -> list[DataInst]:
        """Stratified subsample for merge evaluation.

        Current GEPA implementation:
        - Find common validation IDs (both parents evaluated on these)
        - Create 3 buckets:
          - p1: IDs where parent1 > parent2
          - p2: IDs where parent2 > parent1
          - p3: IDs where parent1 == parent2 (ties)
        - Sample evenly from each bucket (ceil(subsample_size / 3) each)
        - If not enough IDs, fill remainder from other buckets
        - Default subsample_size = 5 (from val_overlap_floor)

        This ensures merge is tested on representative distribution:
        - Cases where parent1 excels
        - Cases where parent2 excels
        - Cases where they tie

        Prevents bias toward one parent's strengths.
        """
        pass

    def _is_duplicate(self, candidate: CandidateProgram, state: GepaState) -> bool:
        """Check if candidate already exists."""
        # Implementation
        pass

    def _get_subsample_scores(self, state, parent_idx, subsample) -> list[float]:
        """Get parent scores on subsample."""
        # Implementation
        pass
````

**Returns:**

- `EvaluateNode()` → If accepted, evaluate on full validation set
- `ContinueNode()` → If rejected or failed

### Graph Construction

```python
def create_gepa_graph(
    adapter: PydanticAIGEPAAdapter,
    config: GepaConfig,
) -> Graph[GepaState, StartNode | EvaluateNode | ContinueNode | ReflectNode | MergeNode, GepaResult]:
    """Create the GEPA optimization graph.

    The graph structure is implicit from node return types:
    - StartNode -> EvaluateNode
    - EvaluateNode -> ContinueNode
    - ContinueNode -> ReflectNode | MergeNode | End
    - ReflectNode -> EvaluateNode | ContinueNode
    - MergeNode -> EvaluateNode | ContinueNode
    """

    # List node types - edges are implicit from return type annotations
    nodes = [
        StartNode,
        EvaluateNode,
        ContinueNode,
        ReflectNode,
    ]

    if config.use_merge:
        nodes.append(MergeNode)

    return Graph(
        nodes=nodes,
        deps_type=GepaDeps,
    )
```

## Data Models

### CandidateProgram

```python
class ComponentValue(BaseModel):
    """A single component with metadata."""
    name: str
    text: str
    version: int = 0  # Track updates to this component

class CandidateProgram(BaseModel):
    """A complete program candidate."""
    idx: int
    components: dict[str, ComponentValue]

    # Genealogy
    parent_indices: list[int] = Field(default_factory=list)
    creation_type: Literal['seed', 'reflection', 'merge']

    # Evaluation results (sparse)
    validation_scores: dict[str, float] = Field(default_factory=dict)  # DataId -> score
    validation_outputs: dict[str, RolloutOutput] = Field(default_factory=dict)

    # Minibatch scores (for reflection)
    minibatch_scores: list[float] | None = None

    # Metadata
    discovered_at_iteration: int
    discovered_at_evaluation: int

    def to_dict_str(self) -> dict[str, str]:
        """Convert to legacy dict[str, str] format for adapter."""
        return {name: comp.text for name, comp in self.components.items()}

    @property
    def avg_validation_score(self) -> float:
        """Average validation score."""
        if not self.validation_scores:
            return 0.0
        return sum(self.validation_scores.values()) / len(self.validation_scores)
```

**Benefits over current approach:**

- ✅ Structured components with metadata
- ✅ Track component versions for merge logic
- ✅ Type-safe access to scores and outputs
- ✅ Clear genealogy tracking

### ParetoFrontEntry

```python
class ParetoFrontEntry(BaseModel):
    """Pareto front for a single validation instance."""
    data_id: str
    best_score: float
    candidate_indices: set[int] = Field(default_factory=set)
    best_outputs: list[tuple[int, RolloutOutput]] = Field(default_factory=list)
```

### GepaConfig

```python
class GepaConfig(BaseModel):
    """Immutable configuration for GEPA optimization."""

    # Budget
    max_evaluations: int = 200
    max_iterations: int | None = None

    # Reflection
    minibatch_size: int = 3
    perfect_score: float = 1.0
    skip_perfect_score: bool = True

    # Component selection
    component_selector: Literal['round_robin', 'all'] = 'round_robin'
    candidate_selector: Literal['pareto', 'current_best'] = 'pareto'

    # Merge
    use_merge: bool = False
    merges_per_accept: int = 1
    max_total_merges: int = 5
    min_shared_validation: int = 3

    # Parallelism (NEW - ~10x speedup!)
    max_concurrent_evaluations: int = 10
    enable_parallel_evaluation: bool = True
    enable_parallel_minibatch: bool = True
    enable_parallel_reflection: bool = True

    # Evaluation policy
    validation_policy: Literal['full', 'sparse'] = 'full'

    # Reproducibility
    seed: int = 0
```

**Simplified from 40+ parameters:**

- ✅ Grouped into logical categories
- ✅ Type-safe with Pydantic validation
- ✅ Clear defaults
- ✅ Immutable during run

### GepaDeps

```python
@dataclass
class GepaDeps:
    """Dependencies injected into all nodes."""
    adapter: PydanticAIGEPAAdapter
    reflection_model: Model
    candidate_selector: CandidateSelector
    component_selector: ComponentSelector
    batch_sampler: BatchSampler
    evaluator: ParallelEvaluator
    pareto_manager: ParetoFrontManager
```

## Usage Pattern

The GEPA graph is designed to be used via manual iteration for maximum flexibility and control:

### Basic Usage

```python
# 1. Create adapter
adapter = PydanticAIGEPAAdapter(
    agent=agent,
    metric=metric,
    input_type=input_type,
    reflection_model=reflection_model,
    cache_manager=CacheManager(...) if enable_cache else None,
)

# 2. Create config
config = GepaConfig(
    max_evaluations=200,
    minibatch_size=3,
    use_merge=False,
    seed=0,
)

# 3. Create dependencies
deps = create_deps(adapter, config)

# 4. Create graph
graph = create_gepa_graph(adapter=adapter, config=config)

# 5. Initialize state
state = GepaState(
    config=config,
    training_set=list(training_set),
    validation_set=list(validation_set) if validation_set else list(training_set),
)

# 6. Create context
ctx = GraphRunContext(
    state=state,
    deps=deps,
)

# 7. Run optimization - graph framework auto-executes nodes
async with graph.iter(ctx=ctx, start_node=StartNode()) as run:
    async for node in run:
        # Graph framework has already executed node.run()
        # Can inspect state after each node
        if isinstance(node, EvaluateNode):
            logger.info(f"Evaluated candidate {len(state.candidates)}")

# 8. Get final result
result = run.end_data
```

### With Custom Logic and Early Stopping

```python
# Setup (same as above)
adapter = PydanticAIGEPAAdapter(...)
config = GepaConfig(...)
deps = create_deps(adapter, config)
graph = create_gepa_graph(adapter=adapter, config=config)
state = GepaState(config=config, training_set=list(training_set), validation_set=list(validation_set))
ctx = GraphRunContext(state=state, deps=deps)

# Run with custom logic
async with graph.iter(ctx=ctx, start_node=StartNode()) as run:
    async for node in run:
        # Custom logging
        print(f"Iteration {state.iteration}: {node.__class__.__name__}")

        # Early stopping based on custom logic
        if state.best_candidate_idx is not None:
            best = state.candidates[state.best_candidate_idx]
            if best.avg_validation_score > 0.95:
                print("Reached target score!")
                state.stopped = True
                state.stop_reason = "User requested stop"
                break

# Get result
result = run.end_data
```

### Helper Function: create_deps

```python
def create_deps(adapter: PydanticAIGEPAAdapter, config: GepaConfig) -> GepaDeps:
    """Create dependencies from config."""
    # Create selectors based on config
    candidate_selector = (
        ParetoCandidateSelector() if config.candidate_selector == 'pareto'
        else CurrentBestCandidateSelector()
    )

    component_selector = (
        RoundRobinComponentSelector() if config.component_selector == 'round_robin'
        else AllComponentSelector()
    )

    batch_sampler = BatchSampler(seed=config.seed, size=config.minibatch_size)

    return GepaDeps(
        adapter=adapter,
        reflection_model=adapter.reflection_model,
        candidate_selector=candidate_selector,
        component_selector=component_selector,
        batch_sampler=batch_sampler,
        evaluator=ParallelEvaluator(),
        pareto_manager=ParetoFrontManager(),
    )
```

**Note:** While the async helper `optimize_agent()` is available, starting with manual iteration provides the most flexibility during initial development.

## Resumability & Checkpointing

### Automatic Checkpointing

**Every node transition is automatically checkpointed** by pydantic-graph:

```python
# Enable checkpointing via store
from pydantic_graph import FileStatePersistence

ctx = GraphRunContext(
    state=state,
    deps=deps,
    store=FileStatePersistence(checkpoint_dir="./gepa_checkpoints"),
)

async with graph.iter(ctx=ctx, start_node=StartNode()) as run:
    async for node in run:
        # Before each node: checkpoint saved
        # After each node: checkpoint updated
        pass
```

**Checkpoint structure:**

```
./gepa_checkpoints/
├── run_20250108_143022/
│   ├── snapshot_000_before_StartNode.json
│   ├── snapshot_001_after_StartNode.json
│   ├── snapshot_002_before_EvaluateNode.json
│   ├── snapshot_003_after_EvaluateNode.json
│   ├── snapshot_004_before_ContinueNode.json
│   └── ...
```

### Resuming from Checkpoint

```python
# Load checkpoint
store = FileStatePersistence(checkpoint_dir=Path(checkpoint_path).parent)
snapshot = await store.load_snapshot(checkpoint_path)

# Recreate adapter, deps, graph
adapter = PydanticAIGEPAAdapter(agent=agent, metric=metric, ...)
deps = create_deps(adapter, snapshot.state.config)
graph = create_gepa_graph(adapter=adapter, config=snapshot.state.config)

# Create context with store
ctx = GraphRunContext(
    state=snapshot.state,
    deps=deps,
    store=store,
)

# Resume from checkpoint state and node
async with graph.iter(ctx=ctx, start_node=snapshot.next_node) as run:
    async for node in run:
        # Continue execution
        pass

# Get result
result = run.end_data
```

### Caching vs Persistence

Two complementary mechanisms work together for optimal performance:

| Aspect          | FileStatePersistence (Graph-level)                   | CacheManager (Call-level)                              |
| --------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| **What**        | Graph execution state (nodes, iteration, candidates) | LLM call results (agent runs, metrics)                 |
| **When**        | Each node transition                                 | Each LLM/metric call                                   |
| **Granularity** | Coarse (whole optimization state)                    | Fine (individual evaluations)                          |
| **Storage**     | Single JSON file per run                             | Many pickle files across runs                          |
| **Scope**       | Single run (resume from checkpoint)                  | Across runs (persistent cache)                         |
| **Purpose**     | Resume workflow after crash                          | Avoid duplicate LLM calls                              |
| **Example**     | "Resume from iteration 50"                           | "Don't re-run evaluation of candidate A on instance 5" |

**How They Work Together:**

```python
# Setup both mechanisms
cache_manager = CacheManager(cache_dir=".gepa_cache", enabled=True)
persistence = FileStatePersistence(json_file=Path("runs/run_001.json"))

adapter = PydanticAIGEPAAdapter(
    agent=agent,
    metric=metric,
    cache_manager=cache_manager,  # Call-level caching
)

ctx = GraphRunContext(
    state=state,
    deps=deps,
    store=persistence,  # Graph-level checkpointing
)

# Scenario: Crash at iteration 50
# 1. FileStatePersistence: Resume from last checkpoint (iteration 50, ReflectNode)
# 2. CacheManager: All previous LLM calls (iterations 0-49) still cached
# 3. Result: Fast resume + no duplicate work
```

**Benefits:**

- ✅ **Fast resume**: Load state from last node
- ✅ **No duplicate LLM calls**: Cache persists across runs
- ✅ **Full parallelism**: Concurrent evaluations within nodes
- ✅ **Transparent**: Caching happens automatically in adapter
- ✅ **Human-readable checkpoints**: JSON for inspection
- ✅ **Cross-run optimization**: Cache reused when re-running with tweaks

## Benefits Summary

### vs Current gepa.optimize()

| Aspect                  | Current (gepa.optimize)       | New (pydantic-graph)            |
| ----------------------- | ----------------------------- | ------------------------------- |
| **Execution Model**     | Synchronous, blocking         | Async-native, non-blocking      |
| **Observability**       | Black box                     | Step-by-step visibility         |
| **Checkpointing**       | Manual state saves            | Automatic at every step         |
| **Resumability**        | Load saved state              | Resume from any checkpoint      |
| **Parallel Evaluation** | Sequential                    | Async concurrent (~10x faster)  |
| **Type Safety**         | Loose dicts, Any types        | Pydantic models throughout      |
| **State Management**    | 15+ mutable fields            | Structured, validated models    |
| **Configuration**       | 40+ flat parameters           | Grouped config object           |
| **Extensibility**       | Rigid proposer system         | Add/modify nodes easily         |
| **Testing**             | Hard to test individual steps | Each node testable in isolation |
| **Debugging**           | Limited visibility            | Full state at every step        |

### vs String-Based Optimization

| Aspect             | Current (dict[str, str]) | Future (Structured Types)       |
| ------------------ | ------------------------ | ------------------------------- |
| **Component Type** | Untyped strings          | Pydantic models with metadata   |
| **Validation**     | None                     | Automatic with Pydantic         |
| **Versioning**     | Not tracked              | Component.version field         |
| **Merge Logic**    | String comparison        | Structured component comparison |
| **Serialization**  | JSON dict                | Pydantic JSON with validation   |

## Implementation Steps

**Key Implementation Notes:**

1. **Parallelism is critical** - Implement from the start, not as optimization

   - Validation evaluation: ~10x speedup
   - Minibatch evaluation: ~5x speedup per iteration
   - Multi-component reflection: ~3x speedup
   - Overall: ~10x end-to-end speedup

2. **Use existing adapters** - Reuse `PydanticAIGEPAAdapter`, `CacheManager`, etc.

3. **Follow dependencies** - Each step builds on previous steps

4. **Test as you go** - Write tests for each step before moving to next

### Step 1: Data Models Foundation

**Build the core data structures first** - everything else depends on these.

**Files to create:**

- `src/pydantic_ai_gepa/gepa_graph/models/__init__.py`
- `src/pydantic_ai_gepa/gepa_graph/models/state.py`
- `src/pydantic_ai_gepa/gepa_graph/models/candidate.py`
- `src/pydantic_ai_gepa/gepa_graph/models/pareto.py`
- `src/pydantic_ai_gepa/gepa_graph/models/result.py`

**What to implement:**

1. **GepaState** - Complete state model with all fields

   - Include all fields from design above
   - Add Pydantic validators
   - Add helper methods: `get_best_candidate()`, `add_candidate()`, etc.

2. **GepaConfig** - Configuration with parallelism settings

   ```python
   class GepaConfig(BaseModel):
       # Budget
       max_evaluations: int = 200
       max_iterations: int | None = None

       # Reflection
       minibatch_size: int = 3
       perfect_score: float = 1.0
       skip_perfect_score: bool = True

       # Selection
       component_selector: Literal['round_robin', 'all'] = 'round_robin'
       candidate_selector: Literal['pareto', 'current_best'] = 'pareto'

       # Merge
       use_merge: bool = False
       merges_per_accept: int = 1
       max_total_merges: int = 5

       # Parallelism (NEW)
       max_concurrent_evaluations: int = 10
       enable_parallel_evaluation: bool = True
       enable_parallel_minibatch: bool = True
       enable_parallel_reflection: bool = True

       # Reproducibility
       seed: int = 0
   ```

3. **CandidateProgram** and **ComponentValue** - Structured candidates

   - Replace `dict[str, str]` with proper models
   - Track component versions
   - Include genealogy fields

4. **ParetoFrontEntry** - Per-instance Pareto tracking

5. **GepaResult** - Final result model

**Tests to write:**

- `tests/gepa_graph/models/test_state.py` - Test state creation, validation, helpers
- `tests/gepa_graph/models/test_candidate.py` - Test candidate creation, to_dict_str()
- `tests/gepa_graph/models/test_config.py` - Test config validation

**Acceptance criteria:**

- All models have complete type hints
- All models have Pydantic validation
- All models can round-trip to/from JSON
- Unit tests pass

### Step 2: Selection Strategies

**Implement selection logic** - needed by nodes.

**Files to create:**

- `src/pydantic_ai_gepa/gepa_graph/selectors/__init__.py`
- `src/pydantic_ai_gepa/gepa_graph/selectors/candidate.py`
- `src/pydantic_ai_gepa/gepa_graph/selectors/component.py`
- `src/pydantic_ai_gepa/gepa_graph/selectors/batch.py`

**What to implement:**

1. **Candidate Selection**

   ```python
   class CandidateSelector(Protocol):
       def select(self, state: GepaState) -> int:
           """Select which candidate to improve."""
           ...

   class ParetoCandidateSelector:
       """Select from Pareto front by frequency."""
       def select(self, state: GepaState) -> int:
           # Find dominators, weight by frequency
           pass

   class CurrentBestCandidateSelector:
       """Always select best candidate."""
       def select(self, state: GepaState) -> int:
           return state.best_candidate_idx or 0
   ```

2. **Component Selection**

   ```python
   class ComponentSelector(Protocol):
       def select(self, state: GepaState, candidate_idx: int) -> list[str]:
           """Select which components to update."""
           ...

   class RoundRobinComponentSelector:
       """Cycle through components."""
       pass

   class AllComponentSelector:
       """Update all components."""
       pass
   ```

3. **Batch Sampling**
   ```python
   class BatchSampler:
       """Sample minibatches deterministically."""
       def sample(
           self,
           training_set: list[DataInst],
           state: GepaState,
           size: int,
       ) -> list[DataInst]:
           # Epoch-based shuffled sampling
           pass
   ```

**Tests to write:**

- Test each selector with various states
- Test deterministic behavior (same seed = same selection)

**Acceptance criteria:**

- Selectors match current gepa behavior
- Deterministic given same seed
- Unit tests pass

### Step 3: Evaluation & Pareto Logic

**Implement evaluation helpers** - needed by EvaluateNode.

**Files to create:**

- `src/pydantic_ai_gepa/gepa_graph/evaluation/__init__.py`
- `src/pydantic_ai_gepa/gepa_graph/evaluation/evaluator.py`
- `src/pydantic_ai_gepa/gepa_graph/evaluation/pareto.py`

**What to implement:**

1. **Parallel Evaluator with Caching**

   ```python
   class ParallelEvaluator:
       """Evaluate candidates in parallel with rate limiting.

       Caching happens transparently within the adapter - the evaluator
       just coordinates parallel execution and rate limiting.
       """

       async def evaluate_batch(
           self,
           candidate: CandidateProgram,
           batch: list[DataInst],
           adapter: Adapter,
           max_concurrent: int = 10,
           capture_traces: bool = False,
       ) -> EvaluationResults:
           """Evaluate batch in parallel.

           Caching strategy:
           - Each evaluate_one() call checks the adapter's CacheManager first
           - Cache key: hash(data_inst, candidate, capture_traces)
           - Cache hit: Returns cached (trajectory, output, score) instantly
           - Cache miss: Runs LLM, computes metric, caches result
           - Parallelism: All tasks launch concurrently, semaphore limits active LLM calls

           Example with 20 instances:
           - First run: 20 LLM calls (max 10 concurrent)
           - Second run (same candidate): 20 cache hits (instant)
           - Third run (new candidate): 20 LLM calls again
           - After crash: Resume + cache hits for already-evaluated instances
           """
           semaphore = asyncio.Semaphore(max_concurrent)

           async def evaluate_one(instance: DataInst):
               async with semaphore:
                   # Adapter handles caching internally:
                   # 1. Check cache_manager.get_cached_agent_run(instance, candidate, capture_traces)
                   # 2. If hit: return cached result
                   # 3. If miss: run LLM, cache result, return
                   return await adapter.evaluate(
                       [instance],
                       candidate.to_dict_str(),
                       capture_traces,
                   )

           # Launch all tasks concurrently
           # Semaphore ensures max_concurrent LLM calls active at once
           # Cache hits return instantly without consuming semaphore time
           tasks = [evaluate_one(inst) for inst in batch]
           results = await asyncio.gather(*tasks)
           return self._merge_results(results)
   ```

   **Cache Implementation Details:**

   The adapter (reused from existing `src/pydantic_ai_gepa/adapter.py`) handles caching:

   ```python
   # Inside adapter.evaluate() or adapter.process_data_instance():

   # 1. Check agent run cache
   if self.cache_manager and candidate:
       cached_agent_result = self.cache_manager.get_cached_agent_run(
           data_inst,
           candidate,
           capture_traces,
       )

       if cached_agent_result is not None:
           trajectory, output = cached_agent_result  # Cache hit!
       else:
           # Cache miss - run agent
           if capture_traces:
               trajectory, output = self._run_with_trace(data_inst)
           else:
               output = self._run_simple(data_inst)
               trajectory = None

           # Cache the agent run result
           self.cache_manager.cache_agent_run(
               data_inst,
               candidate,
               trajectory,
               output,
               capture_traces,
           )

   # 2. Check metric cache
   if self.cache_manager and candidate:
       cached_result = self.cache_manager.get_cached_metric_result(
           data_inst,
           output,
           candidate,
       )

       if cached_result is not None:
           score, metric_feedback = cached_result  # Cache hit!
       else:
           # Call metric and cache result
           score, metric_feedback = self.metric(data_inst, output)
           self.cache_manager.cache_metric_result(
               data_inst,
               output,
               candidate,
               score,
               metric_feedback,
           )
   ```

   **Key Points:**

   - **Caching is per-call**, not per-batch
   - **Cache key** = SHA256(data_inst + candidate + capture_traces)
   - **Storage**: `.gepa_cache/{hash}.pkl` files
   - **Reused across runs**: Same candidate + instance = cache hit even after restart
   - **Transparent**: Evaluator doesn't need to know about caching
   - **Parallelism preserved**: Cache checks are fast, don't block other tasks

2. **Pareto Front Manager**

   ```python
   class ParetoFrontManager:
       """Manage per-instance Pareto fronts."""

       def update_fronts(
           self,
           state: GepaState,
           candidate_idx: int,
           eval_results: EvaluationResults,
       ) -> None:
           """Update Pareto fronts with new results."""
           for data_id, score, output in eval_results:
               self._update_instance_front(
                   state.pareto_front,
                   data_id,
                   candidate_idx,
                   score,
                   output,
               )

       def find_dominators(
           self,
           state: GepaState,
       ) -> list[int]:
           """Find non-dominated candidates."""
           # Implementation from GEPA spec
           pass
   ```

**Tests to write:**

- Test parallel evaluation (verify concurrency)
- Test caching behavior (cache hits/misses)
- Test cache invalidation (different candidates)
- Test Pareto front updates
- Test dominator finding

**Acceptance criteria:**

- Parallel evaluation works with rate limiting
- Caching works correctly (adapter integration)
- Cache hits return instantly without LLM calls
- Cache misses trigger LLM calls and cache updates
- Pareto logic matches GEPA spec
- Tests verify ~10x speedup from parallelism

### Step 4: Core Nodes (Start, Evaluate, Continue)

**Implement the basic graph flow** - can run a simple loop.

**Files to create:**

- `src/pydantic_ai_gepa/gepa_graph/nodes/__init__.py`
- `src/pydantic_ai_gepa/gepa_graph/nodes/base.py`
- `src/pydantic_ai_gepa/gepa_graph/nodes/start.py`
- `src/pydantic_ai_gepa/gepa_graph/nodes/evaluate.py`
- `src/pydantic_ai_gepa/gepa_graph/nodes/continue_node.py`

**What to implement:**

1. **StartNode** - Initialize optimization

   ```python
   @dataclass
   class StartNode(BaseNode[GepaState, GepaDeps, None]):
       async def run(self, ctx) -> EvaluateNode:
           # Extract seed candidate from agent
           # Add to state
           # Return EvaluateNode()
           pass
   ```

2. **EvaluateNode** - Evaluate candidates (WITH PARALLELISM)

   ```python
   @dataclass
   class EvaluateNode(BaseNode[GepaState, GepaDeps, None]):
       async def run(self, ctx) -> ContinueNode:
           # Get validation batch
           # Evaluate in parallel
           # Update Pareto fronts
           # Update best candidate
           # Return ContinueNode()
           pass
   ```

3. **ContinueNode** - Decision point
   ```python
   @dataclass
   class ContinueNode(BaseNode[GepaState, GepaDeps, None]):
       def run(self, ctx) -> ReflectNode | MergeNode | End[GepaResult]:
           # Check stopping conditions
           # Increment iteration
           # Return ReflectNode() or MergeNode() or raise End(result)
           pass
   ```

**Tests to write:**

- Test each node independently
- Test basic flow: Start → Evaluate → Continue → Stop

**Acceptance criteria:**

- Can initialize and evaluate seed candidate
- Stopping conditions work
- Basic graph executes

### Step 5: Reflection Proposal Logic

**Implement reflection dataset building and LLM proposal** - needed by ReflectNode.

**Files to create:**

- `src/pydantic_ai_gepa/gepa_graph/proposal/__init__.py`
- `src/pydantic_ai_gepa/gepa_graph/proposal/reflective.py`
- `src/pydantic_ai_gepa/gepa_graph/proposal/llm.py`

**What to implement:**

1. **Reflective Dataset Builder**

   ```python
   class ReflectiveDatasetBuilder:
       """Build reflection records from trajectories."""

       def build_dataset(
           self,
           eval_results: EvaluationResults,
           components: list[str],
       ) -> dict[str, list[dict]]:
           """Build reflective dataset per component."""
           # Extract from trajectories
           # Format as records
           # Return dict[component -> records]
           pass
   ```

2. **LLM Proposal Generator** (WITH PARALLELISM)

   ```python
   class LLMProposalGenerator:
       """Generate improved component texts via LLM."""

       async def propose_texts(
           self,
           candidate: CandidateProgram,
           reflective_data: dict[str, list[dict]],
           components: list[str],
           model: Model,
       ) -> dict[str, str]:
           """Propose new texts for components in parallel."""

           async def propose_for_component(comp: str):
               # Build reflection prompt
               # Call LLM
               # Extract new text
               return comp, new_text

           # Parallelize LLM calls
           tasks = [propose_for_component(c) for c in components]
           results = await asyncio.gather(*tasks)
           return dict(results)
   ```

**Tests to write:**

- Test dataset building with mock trajectories
- Test LLM proposal with mock model

**Acceptance criteria:**

- Dataset format matches adapter expectations
- LLM calls are parallelized
- Proposal extraction works

### Step 6: ReflectNode (Core Optimization)

**Implement reflective mutation** - the heart of GEPA.

**Files to create:**

- `src/pydantic_ai_gepa/gepa_graph/nodes/reflect.py`

**What to implement:**

Full 10-step algorithm as shown in the Node Definitions section above, including:

- Helper methods for each step
- Parallel minibatch evaluation
- Parallel LLM proposal
- Acceptance criteria (strict improvement)

**Tests to write:**

- Test full reflection flow with mocks
- Test each helper method
- Test parallelism (verify speedup)

**Acceptance criteria:**

- All 10 steps implemented
- Parallelism works (minibatch + LLM)
- Acceptance criteria correct (sum improvement)

### Step 7: Merge Logic

**Implement merge proposal** - needed by MergeNode.

**Files to create:**

- `src/pydantic_ai_gepa/gepa_graph/proposal/merge.py`

**What to implement:**

```python
class MergeProposalBuilder:
    """Build merged candidates from Pareto front."""

    def find_merge_pair(
        self,
        state: GepaState,
        dominators: list[int],
    ) -> tuple[int, int] | None:
        """Find two candidates to merge."""
        # Sample from dominators
        pass

    def find_common_ancestor(
        self,
        state: GepaState,
        idx1: int,
        idx2: int,
    ) -> int | None:
        """Find common ancestor for merge."""
        # Walk genealogy
        # Find intersection
        # Filter and select
        pass

    def build_merged_candidate(
        self,
        state: GepaState,
        parent1_idx: int,
        parent2_idx: int,
        ancestor_idx: int,
    ) -> CandidateProgram:
        """Build merged candidate."""
        # For each component:
        #   - If one evolved: take evolved
        #   - If both evolved: take from better parent
        #   - If same: keep
        pass

    def select_merge_subsample(
        self,
        state: GepaState,
        parent1_idx: int,
        parent2_idx: int,
    ) -> list[DataInst]:
        """Stratified subsample for merge evaluation."""
        # Bucket by: parent1 better, parent2 better, tie
        # Sample evenly from buckets
        pass
```

**Tests to write:**

- Test ancestor finding
- Test merge building
- Test stratified sampling

**Acceptance criteria:**

- Merge logic matches GEPA spec
- Genealogy traversal works
- Deduplication works

### Step 8: MergeNode

**Implement merge operation** - completes optimization features.

**Files to create:**

- `src/pydantic_ai_gepa/gepa_graph/nodes/merge.py`

**What to implement:**

Full merge algorithm as shown in the Node Definitions section above, including:

- All helper methods
- Deduplication
- Non-strict acceptance (>= not >)

**Tests to write:**

- Test full merge flow
- Test acceptance criteria

**Acceptance criteria:**

- Merge logic complete
- Non-strict acceptance (>= not >)
- Works with genealogy tracking

**Status:** ✅ Implemented — MergeNode now uses the real merge proposal pipeline, deduplicates via MergeProposalBuilder history, and is covered by dedicated unit tests (`tests/gepa_graph/nodes/test_merge_node.py`).

### Step 9: Graph Construction & Dependencies

**Wire everything together** - create the graph.

**Files to create:**

- `src/pydantic_ai_gepa/gepa_graph/graph.py`
- `src/pydantic_ai_gepa/gepa_graph/deps.py`

**What to implement:**

```python
@dataclass
class GepaDeps:
    """Dependencies injected into all nodes."""
    adapter: PydanticAIGEPAAdapter
    reflection_model: Model
    candidate_selector: CandidateSelector
    component_selector: ComponentSelector
    batch_sampler: BatchSampler
    evaluator: ParallelEvaluator
    pareto_manager: ParetoFrontManager

def create_gepa_graph(
    adapter: PydanticAIGEPAAdapter,
    config: GepaConfig,
) -> Graph:
    """Create and configure the GEPA optimization graph."""

    # Just list node types - edges implicit from return types!
    nodes = [
        StartNode,
        EvaluateNode,
        ContinueNode,
        ReflectNode,
    ]

    if config.use_merge:
        nodes.append(MergeNode)

    return Graph(
        nodes=nodes,
        deps_type=GepaDeps,
    )
```

**Tests to write:**

- Test graph construction
- Test graph with/without merge

**Acceptance criteria:**

- Graph builds correctly
- Dependencies injected properly
- Node types listed correctly

**Status:** ✅ Implemented — `create_gepa_graph` wires all nodes (with forward-ref patching to break cycles) and is covered by `tests/gepa_graph/test_graph.py` for configs with and without merge enabled. *(Follow-up: refactor node modules to remove the temporary `_ensure_forward_refs()` monkey-patching once import cycles are resolved.)*

**Implementation notes (from Step 8 work):**

- `GepaDeps` now carries a long-lived `MergeProposalBuilder` instance (seeded via `config.seed`) so merge deduplication history persists. `create_deps()` should instantiate it once and pass through to every node.
- The MergeNode stub is replaced with the real implementation and already returns `EvaluateNode`/`ContinueNode`. Graph wiring can rely on those async nodes without additional glue.

### Step 10: Helper Functions & Exports

**Create helper functions and public exports** - make it easy to use.

**Files to create:**

- `src/pydantic_ai_gepa/gepa_graph/helpers.py`
- `src/pydantic_ai_gepa/gepa_graph/__init__.py`

**What to implement:**

1. **`create_deps()` helper** - Build dependencies from config

   ```python
   def create_deps(adapter: PydanticAIGEPAAdapter, config: GepaConfig) -> GepaDeps:
       """Create dependencies from config."""
       # Implementation shown in Usage Pattern section
   ```

2. **Public exports** - Export key classes and functions
   ```python
   # In __init__.py
   from .graph import create_gepa_graph
   from .helpers import create_deps
   from .models.state import GepaState, GepaConfig
   from .models.candidate import CandidateProgram, ComponentValue
   from .deps import GepaDeps
   ```

**Tests to write:**

- Test `create_deps()` with different configs
- Test graph construction
- Test end-to-end manual iteration

**Acceptance criteria:**

- Helper functions work correctly
- Exports are clean and documented
- Manual iteration pattern is well-tested

**Status:** ✅ Implemented — `create_deps()` now lives in `src/pydantic_ai_gepa/gepa_graph/helpers.py`, wiring selectors, batch sampling, LLM proposal tooling, and the seeded `MergeProposalBuilder` straight from `GepaConfig`. The public surface is exported via `src/pydantic_ai_gepa/gepa_graph/__init__.py`, so downstream code can `from pydantic_ai_gepa.gepa_graph import create_deps, create_gepa_graph, GepaConfig, GepaState, ...` without reaching into internals. Coverage comes from `tests/gepa_graph/test_helpers.py` (selector/config permutations) and `tests/gepa_graph/test_manual_iteration.py`, which exercises the documented manual iteration loop with the real `Graph.iter(...)` API and ensures the run result improves before stopping.

### Step 11: (Optional) Legacy API Wrapper

**Optionally wrap old API to use new implementation** - can be added later if needed.

**Note:** This step is optional and can be deferred. The manual iteration pattern provides full control and is the recommended approach for initial development, even though the `optimize_agent()` helper exists for convenience.

### Step 12: Integration Testing

**Test the complete system** - verify everything works together.

**Files to create:**

- `tests/gepa_graph/test_integration.py`
- `tests/gepa_graph/test_e2e.py`
- `tests/gepa_graph/test_parallelism.py`

**What to test:**

1. **Full optimization flow**

   ```python
   async def test_full_optimization():
       # Create agent and dataset
       # Run optimization
       # Verify improvement
       # Verify parallelism was used
       pass
   ```

2. **Checkpoint/resume**

   ```python
   async def test_checkpoint_resume():
       # Run optimization partially
       # Save checkpoint
       # Resume from checkpoint
       # Verify continues correctly
       pass
   ```

3. **Parallelism verification**

   ```python
   async def test_parallel_speedup():
       # Time sequential vs parallel
       # Verify ~10x speedup
       pass
   ```

4. **Backward compatibility**
   ```python
   def test_old_api_works():
       # Use old optimize_agent
       # Verify same results
       pass
   ```

**Acceptance criteria:**

- All integration tests pass
- Parallelism verified
- Backward compatibility verified
- Performance targets met (~10x speedup)

**Status:** 🟡 In progress — `tests/gepa_graph/test_integration.py::test_checkpoint_resume_restores_progress` covers the checkpoint/resume scenario via `FullStatePersistence` + `iter_from_persistence`. Remaining items (full optimization flow, parallel speed test, legacy API verification) are still outstanding.

> **Open issue:** Durable (disk-backed) persistence is currently blocked because `GepaState.training_set` and `.validation_set` are marked `exclude=True`, so serialized snapshots omit the datasets entirely. `FullStatePersistence.dump_json()` therefore writes an incomplete state, and `load_json()` raises validation errors when those required fields are missing. We either need to (a) capture dataset identifiers that can be rehydrated on resume, (b) stop excluding the datasets and ensure they serialize cleanly, or (c) expose a helper that injects the datasets before resuming. Until one of those is implemented the supported path is in-memory persistence only.

## Design Decisions

### 1. Async-First API ✅

**Decision:** Use async as primary API with sync wrapper

**Rationale:**

- pydantic-ai is async-native
- LLM calls benefit greatly from async
- Parallelism requires async
- ~10x performance improvement
- Modern Python standard

**Implementation:**

- `optimize_agent()` is the async entry point for the high-level API.
- Both have same signature (except async/await)

### 2. JSON Serialization via Pydantic ✅

**Decision:** Use Pydantic's JSON serialization for all state

**Rationale:**

- Human-readable checkpoints (can inspect in editor)
- Cross-language compatibility
- Built-in Pydantic support
- Easy debugging
- Type validation on deserialization

**Implementation:**

- All models inherit from `BaseModel`
- Use `model_dump_json()` and `model_validate_json()`
- Store checkpoints as `.json` files

### 3. Semaphore-Based Rate Limiting ✅

**Decision:** Use `asyncio.Semaphore` for rate limiting

**Rationale:**

- LLM APIs have rate limits
- Prevents overwhelming services
- Configurable concurrency
- Standard Python pattern

**Implementation:**

```python
semaphore = asyncio.Semaphore(max_concurrent)
async def evaluate_one(item):
    async with semaphore:
        return await adapter.evaluate(...)
```

**Default:** `max_concurrent_evaluations = 10`

### 4. Class-Based Nodes ✅

**Decision:** Use class-based `BaseNode` approach (not beta builder API)

**Rationale:**

- Better for complex node logic (10+ helper methods)
- Natural encapsulation
- Easier testing
- Better IDE support
- Production-proven (pydantic-ai uses it)

See `GRAPH_API_COMPARISON.md` for detailed analysis.

### 5. Return-Type-Based Routing ✅

**Decision:** Nodes return node instances, graph edges implicit from return types

**Rationale:**

- This is how pydantic-graph class-based API works
- No separate edge definitions needed
- Type-safe routing
- Clear from return type annotations

**Implementation:**

```python
@dataclass
class ContinueNode(BaseNode[GepaState, GepaDeps, None]):
    def run(self, ctx) -> ReflectNode | MergeNode | End[GepaResult]:
        if should_merge:
            return MergeNode()  # Returns node instance
        return ReflectNode()    # Returns node instance

# Graph construction - just list node types
Graph(nodes=[StartNode, EvaluateNode, ContinueNode, ReflectNode, MergeNode])
```

### 6. Single MergeNode ✅

**Decision:** Implement merge as a single node with helper methods

**Rationale:**

- Merge is conceptually atomic
- Breaking into sub-nodes adds graph complexity
- Helper methods provide same organization
- Easier to test as unit
- Simpler graph visualization

**Implementation:**

- `MergeNode` with 8+ helper methods
- All logic encapsulated in one class

### 7. Automatic Checkpointing ✅

**Decision:** Let pydantic-graph handle checkpointing automatically

**Rationale:**

- Checkpoints at every node boundary (before/after)
- Free resumability
- No manual state saving needed
- Built-in to pydantic-graph

**Retention:** Keep only most recent checkpoint per iteration to save space

## File Structure

```
src/pydantic_ai_gepa/
├── __init__.py                      # Public API (backward compat)
├── adapter.py                       # Existing adapter (reused)
├── cache.py                         # Existing cache (reused)
├── signature.py                     # Existing signature (reused)
├── types.py                         # Existing types (reused)
├── runner.py                        # Legacy API (wrapper to new)
│
└── gepa_graph/                      # NEW: Native pydantic-graph implementation
    ├── __init__.py                  # Public API exports
    │
    ├── models/                      # Data models
    │   ├── __init__.py
    │   ├── state.py                 # GepaState, GepaConfig
    │   ├── candidate.py             # CandidateProgram, ComponentValue
    │   ├── pareto.py                # ParetoFrontEntry
    │   └── result.py                # GepaResult
    │
    ├── nodes/                       # Graph nodes
    │   ├── __init__.py
    │   ├── base.py                  # Base node types
    │   ├── start.py                 # StartNode
    │   ├── evaluate.py              # EvaluateNode
    │   ├── continue_node.py         # ContinueNode
    │   ├── reflect.py               # ReflectNode
    │   └── merge.py                 # MergeNode
    │
    ├── selectors/                   # Selection strategies
    │   ├── __init__.py
    │   ├── candidate.py             # Candidate selection
    │   ├── component.py             # Component selection
    │   └── batch.py                 # Batch sampling
    │
    ├── evaluation/                  # Evaluation logic
    │   ├── __init__.py
    │   ├── evaluator.py             # Parallel evaluation
    │   └── pareto.py                # Pareto front management
    │
    ├── proposal/                    # Proposal generation
    │   ├── __init__.py
    │   ├── reflective.py            # Reflective dataset building
    │   ├── merge.py                 # Merge logic
    │   └── llm.py                   # LLM-based proposal
    │
    ├── graph.py                     # Graph construction
    ├── api.py                       # Public API functions
    ├── deps.py                      # GepaDeps definition
    └── utils.py                     # Shared utilities
```

## Testing Strategy

### Unit Tests

Each component tested in isolation:

```python
# tests/gepa_graph/models/test_candidate.py
def test_candidate_program_creation():
    candidate = CandidateProgram(
        idx=0,
        components={"system": ComponentValue(name="system", text="...")},
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )
    assert candidate.to_dict_str() == {"system": "..."}

# tests/gepa_graph/nodes/test_evaluate.py
async def test_evaluate_node():
    node = EvaluateNode()
    state = GepaState(...)
    deps = GepaDeps(...)
    ctx = GraphRunContext(state=state, deps=deps)

    result = await node.run(ctx)
    assert isinstance(result, ContinueNode)
    assert state.total_evaluations > 0
```

### Integration Tests

Test node interactions:

```python
# tests/gepa_graph/test_integration.py
async def test_full_optimization_flow():
    graph = create_gepa_graph(...)
    state = GepaState(...)
    ctx = GraphRunContext(state=state, deps=deps)

    async with graph.iter(ctx=ctx, start_node=StartNode()) as run:
        nodes_executed = []
        async for node in run:
            nodes_executed.append(node.__class__.__name__)

    assert 'StartNode' in nodes_executed
    assert 'EvaluateNode' in nodes_executed
    assert 'ReflectNode' in nodes_executed
    assert len(state.candidates) > 1
```

### End-to-End Tests

Full optimization scenarios:

```python
# tests/gepa_graph/test_e2e.py
async def test_classification_optimization():
    # Setup
    agent = Agent(model="openai:gpt-4o", ...)
    training_set = [...]
    adapter = PydanticAIGEPAAdapter(agent=agent, metric=my_metric, ...)
    config = GepaConfig(max_evaluations=50)
    deps = create_deps(adapter, config)
    graph = create_gepa_graph(adapter=adapter, config=config)
    state = GepaState(config=config, training_set=training_set, validation_set=training_set)
    ctx = GraphRunContext(state=state, deps=deps)

    # Run optimization
    async with graph.iter(ctx=ctx, start_node=StartNode()) as run:
        async for node in run:
            pass

    # Verify results
    result = run.end_data
    assert result.best_score > result.original_score
    assert len(state.candidates) > 1
```

## Implementation Checklist

Use this to track progress:

- [x] Step 1: Data Models Foundation
  - [x] GepaState model
  - [x] GepaConfig model with parallelism settings
  - [x] CandidateProgram and ComponentValue models
  - [x] ParetoFrontEntry model
  - [x] GepaResult model
  - [x] Unit tests for all models

- [x] Step 2: Selection Strategies
  - [x] CandidateSelector (Pareto, CurrentBest)
  - [x] ComponentSelector (RoundRobin, All)
  - [x] BatchSampler (epoch-based shuffling)
  - [x] Unit tests for selectors

- [x] Step 3: Evaluation & Pareto Logic
  - [x] ParallelEvaluator with semaphore rate limiting
  - [x] ParetoFrontManager
  - [x] Unit tests with parallelism verification

- [x] Step 4: Core Nodes
  - [x] StartNode
  - [x] EvaluateNode with parallel evaluation
  - [x] ContinueNode with stopping conditions
  - [x] Unit tests for each node

- [x] Step 5: Reflection Proposal Logic
  - [x] ReflectiveDatasetBuilder
  - [x] LLMProposalGenerator with parallel LLM calls
  - [x] Unit tests with mocks

- [x] Step 6: ReflectNode
  - [x] Full 10-step algorithm
  - [x] All helper methods
  - [x] Parallel minibatch evaluation
  - [x] Parallel LLM proposal
  - [x] Unit tests

- [x] Step 7: Merge Logic
  - [x] MergeProposalBuilder
  - [x] Ancestor finding
  - [x] Merge building
  - [x] Stratified subsample selection
  - [x] Unit tests

- [x] Step 8: MergeNode
  - [x] Full merge algorithm
  - [x] All helper methods
  - [x] Deduplication
  - [x] Unit tests

- [x] Step 9: Graph Construction
  - [x] GepaDeps dataclass
  - [x] create_gepa_graph function
  - [x] Unit tests

- [x] Step 10: Helper Functions & Exports
  - [x] create_deps() helper function
  - [x] Public exports (**init**.py)
  - [x] End-to-end tests with manual iteration

- [ ] Step 11: Integration Testing
  - [ ] Full optimization flow test
  - [x] Checkpoint/resume test

## References

- [GEPA_SPEC.md](./GEPA_SPEC.md) - Full GEPA technical specification
- [NOTES.md](./NOTES.md) - Async workflow implementation analysis
- [GRAPH_API_COMPARISON.md](./GRAPH_API_COMPARISON.md) - Beta vs class-based API comparison
- [GRAPH_PATTERN_FIX.md](./GRAPH_PATTERN_FIX.md) - Correct pydantic-graph pattern examples
- [PARALLELISM_IN_GEPA.md](./PARALLELISM_IN_GEPA.md) - Parallelism strategy and implementation
- [pydantic-graph documentation](../pydantic-ai/pydantic_graph/) - Graph execution framework
- [Current implementation](./src/pydantic_ai_gepa/) - Existing code to migrate from

---

**Status:** Ready for Implementation
**Last Updated:** 2025-01-08
**Pattern:** Class-based nodes with return-type routing (no explicit edges)
