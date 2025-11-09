"""Tests for the ParallelEvaluator helper."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from pydantic_ai.messages import UserPromptPart

from pydantic_ai_gepa.gepa_graph.evaluation import ParallelEvaluator
from pydantic_ai_gepa.gepa_graph.models import CandidateProgram, ComponentValue
from pydantic_ai_gepa.types import DataInstWithPrompt, RolloutOutput, Trajectory


def _make_candidate() -> CandidateProgram:
    return CandidateProgram(
        idx=0,
        components={"system": ComponentValue(name="system", text="Hello")},
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )


def _make_data_inst(case_id: str) -> DataInstWithPrompt:
    return DataInstWithPrompt(
        user_prompt=UserPromptPart(content=f"prompt-{case_id}"),
        message_history=None,
        metadata={},
        case_id=case_id,
    )


@dataclass
class _FakeEvaluationBatch:
    outputs: list[RolloutOutput[str]]
    scores: list[float]
    trajectories: list[Trajectory] | None = None


class _RecordingAdapter:
    def __init__(self, delay: float = 0.01) -> None:
        self._delay = delay
        self.inflight = 0
        self.max_inflight = 0
        self.call_count = 0

    async def evaluate(self, batch, candidate, capture_traces):
        self.call_count += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self._delay)
            trajectories = None
            if capture_traces:
                trajectories = [
                    Trajectory(messages=[], final_output=None, data_inst=batch[0])
                ]

            return _FakeEvaluationBatch(
                outputs=[RolloutOutput.from_success(candidate["system"])],
                scores=[1.0],
                trajectories=trajectories,
            )
        finally:
            self.inflight -= 1


class _CachingAdapter:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[tuple[str, str], ...]], _FakeEvaluationBatch] = {}
        self.llm_calls = 0
        self.cache_hits = 0

    async def evaluate(self, batch, candidate, capture_traces):
        key = (batch[0].case_id, tuple(sorted(candidate.items())))
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        self.llm_calls += 1
        batch_result = _FakeEvaluationBatch(
            outputs=[RolloutOutput.from_success(f"out-{self.llm_calls}")],
            scores=[float(self.llm_calls)],
        )
        self._cache[key] = batch_result
        return batch_result


@pytest.mark.asyncio
async def test_parallel_evaluator_limits_concurrency() -> None:
    evaluator = ParallelEvaluator()
    adapter = _RecordingAdapter(delay=0.05)
    batch = [_make_data_inst(str(i)) for i in range(5)]

    result = await evaluator.evaluate_batch(
        candidate=_make_candidate(),
        batch=batch,
        adapter=adapter,
        max_concurrent=2,
    )

    assert len(result) == len(batch)
    assert adapter.max_inflight == 2


@pytest.mark.asyncio
async def test_parallel_evaluator_integrates_with_adapter_cache() -> None:
    evaluator = ParallelEvaluator()
    adapter = _CachingAdapter()
    batch = [_make_data_inst(str(i)) for i in range(4)]
    candidate = _make_candidate()

    await evaluator.evaluate_batch(
        candidate=candidate,
        batch=batch,
        adapter=adapter,
    )
    await evaluator.evaluate_batch(
        candidate=candidate,
        batch=batch,
        adapter=adapter,
    )

    assert adapter.llm_calls == len(batch)
    assert adapter.cache_hits == len(batch)


@pytest.mark.asyncio
async def test_parallel_evaluator_collects_trajectories() -> None:
    evaluator = ParallelEvaluator()
    adapter = _RecordingAdapter(delay=0)
    batch = [_make_data_inst(str(i)) for i in range(3)]

    result = await evaluator.evaluate_batch(
        candidate=_make_candidate(),
        batch=batch,
        adapter=adapter,
        capture_traces=True,
    )

    assert result.trajectories is not None
    assert len(result.trajectories) == len(batch)
