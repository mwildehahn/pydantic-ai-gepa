# Session Notes (Math Tools GEPA)

## 2025-11-14T15:45-08:00 kickoff
- Pulled latest repo state (`mh/hypothesis-metadata` branch) and inspected math_tools optimization context.
- Observed most recent run `optimization_results/math_tools_optimization_20251114_144658.json` (best score 0.75 after 22 iterations, best candidate idx=1) with plateau at 75%.
- Captured current `instructions` component text for reference (emphasizes single `run_python` call, interval semantics, rounding guardrails).
- Verified GEPA config (`examples/math_tools.py`): `max_evaluations=300`, `CandidateSelectorStrategy.PARETO`, reflection model `gpt-5.1` w/high reasoning effort.
- Setup Logfire MCP access (project id `9b9cc59a-8499-4187-8efe-21661baf2268`) for querying reflection traces per user guidance.

## Next steps
1. Mine Logfire traces for `propose new texts` spans to understand hypothesis metadata, scratchpad usage, and candidate drift.
2. Identify instruction bottlenecks (e.g., student stuck repeating single-playbook, missing exploration) and design meta-instruction updates to encourage reflection & scratchpad iteration.
3. Update `src/pydantic_ai_gepa/gepa_graph/proposal/instruction.py` (and related components) with higher-level evolution cues, then re-run `uv run --env-file .env python examples/math_tools.py` to validate.
4. Continue logging each experiment and result snapshot here so the session can resume cleanly if interrupted.

## 2025-11-14T16:15-08:00 logfire review
- Pulled Logfire trace `019a84a54ad978e70a13fdf5df0dc657` (`span_name=propose new texts`) to confirm the reflection agent is still using the stock `DEFAULT_AGENT_INSTRUCTIONS` block; HTTP child spans show the actual student instructions being issued to `gpt-5-nano`.
- Sampled validation artifacts from `optimization_results/math_tools_20251114_144658.json` → best candidate idx=1 still records 3 reflection minibatch failures due to `tool_calls_limit` exhaustion on `digit-sum-2-200`, `primorial-product`, and `mixed-boundaries`.
- Aggregated 18 evaluation errors overall, all with `error_message` = "The next tool call(s) would exceed the tool_calls_limit of 5", indicating our prompt mutations still fail to enforce the "one run_python call, then final_result" discipline on a significant fraction of the data.
- Observation: metadata for the winning component fixates on the same playbook (single run + interval guardrails). We need instructions that deliberately explore *new* levers (scratchpad planning, comparative reasoning, adaptive retries) rather than restating the same pattern each iteration.

## 2025-11-14T16:40-08:00 instruction meta-prompt update
- Rewrote `DEFAULT_AGENT_INSTRUCTIONS` with an explicit **Evolution Mandate** (requires each reflection to declare two "Evolution Moves" and tie them to scratchpad fields) plus a Scratchpad Relay protocol so ideas get recorded as Keep/Change/Experiment bullets.
- Highlighted concrete move menu (planning scaffolds, self-check loops, tool handshake rewrites, persona shifts, etc.) and emphasized generalization beyond the math dataset.
- Added guidance to mark where the new moves show up in the emitted instructions so future iterations can trace experiments.
- Tests: `uv run pytest tests/gepa_graph/proposal/test_instruction.py`.

## 2025-11-14T17:05-08:00 100-eval run
- Added CLI arg `--max-evaluations` (default 100) so we can throttle GEPA runs without editing code; `GepaConfig.max_evaluations` now respects the flag.
- Ran `uv run --env-file .env python examples/math_tools.py --results-dir optimization_results --max-evaluations 100`.
  - Output file: `optimization_results/math_tools_optimization_20251114_153604.json`.
  - Result: best score plateaued at 0.75 after 8 iterations / 104 metric calls (same as prior best), indicating instruction meta-prompt changes alone haven't moved math_tools past the plateau yet.
- Next: inspect fresh traces from this run via Logfire to confirm whether reflection agent is now logging Evolution Moves, and consider tightening student instruction scratchpad (e.g., enforce self-check loop in final_result description).

## 2025-11-14T17:25-08:00 failure analysis
- Parsed `optimization_results/math_tools_20251114_153604.json`.
  - Best candidate (idx=1) still caps at 0.75 mean score; minibatches perfect but validation shows `tribonacci-20` score 0 (answer 66012 instead of 35890) and `empty-range-edge` score 0 (interprets "Sum all integers from 20 to 10" as 10..20 inclusive instead of empty range).
  - Evaluation errors dropped to 12 (all tool_call limit) but only on non-best candidates; best candidate had zero tool-call violations → previous Evolution Moves fixed the tool overuse issue.
- Conclusion: plateau now caused by **math/semantic correctness**, not tool usage. Need to steer the reflection agent so at least one Evolution Move targets "math validation / edge-case reasoning" each iteration (e.g., range-direction heuristics, recurrence sanity checks), instead of repeatedly focusing on tool budgets.

## 2025-11-14T17:35-08:00 edge-move enforcement
- Strengthened `DEFAULT_AGENT_INSTRUCTIONS` again: Evolution Mandate now requires at least one "Edge Reasoning" move per reflection plus an "Edge Insight" scratchpad entry describing the unsolved math/logic failure. Added escalation rule if an edge persists across multiple iterations.
- Tests: `uv run pytest tests/gepa_graph/proposal/test_instruction.py`.

## 2025-11-14T17:45-08:00 edge wording tweak
- Generalized the Edge-Case Forcing Function text so it applies to any domain (not just math) while keeping the "Edge Insight" requirement. Tests: `uv run pytest tests/gepa_graph/proposal/test_instruction.py`.

## 2025-11-14T18:00-08:00 proposal schema bumps
- Extended `TrajectoryAnalysis` to include `edge_insight`, `evolution_moves`, and `success_checkpoint` so reflection outputs can explicitly capture the new Evolution/Edge requirements. Component metadata now stores these fields (and prints them in the "Stored hypotheses" block) for downstream iterations to inherit.
- Updated tests to expect the richer reasoning payload. Command: `uv run pytest tests/gepa_graph/proposal/test_instruction.py`.
