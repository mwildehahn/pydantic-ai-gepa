"""Tests covering the math_tools example metric adjustments."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
from pydantic_ai import usage as _usage

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import math_tools  # type: ignore[import-not-found]
from pydantic_ai_gepa.types import DataInstWithInput, RolloutOutput


def _make_data_inst() -> DataInstWithInput[math_tools.MathProblemInput]:
    return DataInstWithInput(
        input=math_tools.MathProblemInput(problem="Compute 2 + 2."),
        message_history=None,
        metadata={
            "expected_answer": 4.0,
            "tolerance": 1e-9,
            "feedback": "Use the sandbox to verify arithmetic.",
            "ideal_expression": "print(4)",
        },
        case_id="penalty-case",
    )


def _make_output(answer: float = 4.0) -> math_tools.MathProblemOutput:
    return math_tools.MathProblemOutput(
        explanation="Solved via sandbox",
        expression="print(4)",
        answer=answer,
    )


def test_metric_penalizes_multiple_run_python_invocations() -> None:
    data_inst = _make_data_inst()
    baseline = math_tools.metric(
        data_inst,
        RolloutOutput.from_success(
            _make_output(), usage=_usage.RunUsage(tool_calls=1)
        ),
    )
    penalized = math_tools.metric(
        data_inst,
        RolloutOutput.from_success(
            _make_output(), usage=_usage.RunUsage(tool_calls=3)
        ),
    )

    assert baseline.score == pytest.approx(1.0)
    assert penalized.score == pytest.approx(0.8)
    assert penalized.score < baseline.score
    assert "run_python" in (penalized.feedback or "")
