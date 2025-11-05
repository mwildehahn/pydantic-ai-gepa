"""End-to-end test of optimize_agent_prompts using a small pydantic_evals dataset.

This exercises a full (minimal) GEPA optimization flow with:
- TestModel() as both the agent model and the reflection model
- A tiny categorization dataset (10 items)
- A low metric budget to keep the run short
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic_ai_gepa.components import (
    extract_seed_candidate,
    extract_seed_candidate_with_signature,
)
from pydantic_ai_gepa.reflection import ProposalOutput, UpdatedComponent
from pydantic_ai_gepa.runner import optimize_agent_prompts
from pydantic_ai_gepa.signature_agent import SignatureAgent
from pydantic_ai_gepa.types import DataInst, DataInstWithInput, DataInstWithPrompt, RolloutOutput

from pydantic_ai import Agent
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset


def test_optimize_agent_prompts_minimal_flow():
    """Run a minimal optimization flow over a tiny categorization dataset.

    We don't expect meaningful optimization here; we just validate a complete run
    finishes and returns a structured result within a small metric budget.
    """

    # Build a small categorization dataset (10 items) using pydantic_evals
    def _label_for_token(token: str) -> str:
        return {
            "good": "positive",
            "bad": "negative",
            "ok": "neutral",
        }[token]

    tokens = [
        "good" if i % 3 == 0 else ("bad" if i % 3 == 1 else "ok") for i in range(10)
    ]

    dataset = Dataset(
        cases=[
            Case(
                name=f"case-{i}",
                inputs={"text": f"Sample {i} describing something {tok}."},
                expected_output=_label_for_token(tok),
            )
            for i, tok in enumerate(tokens)
        ]
    )

    # Convert the dataset to GEPA DataInst entries
    trainset: list[DataInst] = [
        DataInstWithPrompt(
            user_prompt=UserPromptPart(
                content=(
                    "Categorize the following input as one of: positive, negative, or neutral.\n"
                    f"Input: {case.inputs['text']}"
                )
            ),
            message_history=None,
            metadata={"label": str(case.expected_output)},
            case_id=case.name or f"case-{i}",
        )
        for i, case in enumerate(dataset.cases)
    ]

    # Agent returns a fixed label; we are not testing real model behavior here
    agent = Agent(
        TestModel(custom_output_text="neutral"),
        instructions=(
            "You are a concise classifier. Output exactly one of: positive, negative, neutral."
        ),
    )

    seed = extract_seed_candidate(agent)

    # Simple metric: 1.0 if predicted label matches expected label, else 0.0
    def metric(
        data_inst: DataInst, output: RolloutOutput[Any]
    ) -> tuple[float, str | None]:
        predicted = (
            str(output.result).strip().lower()
            if output.success and output.result is not None
            else ""
        )
        expected = str(data_inst.metadata.get("label", "")).strip().lower()
        score = 1.0 if predicted == expected else 0.0
        return score, ("Correct" if score == 1.0 else "Incorrect")

    reflection_output = ProposalOutput(
        updated_components=[
            UpdatedComponent(component_name="instructions", optimized_value="Optimized")
        ]
    )
    reflection_model = TestModel(
        custom_output_args=reflection_output.model_dump(mode="python")
    )

    # Keep the budget low; use TestModel() for the reflection model to exercise the full path
    result = optimize_agent_prompts(
        agent=agent,
        trainset=trainset,
        metric=metric,
        reflection_model=reflection_model,
        max_metric_calls=20,
        display_progress_bar=False,
        track_best_outputs=False,
        seed=0,
    )

    # Basic result shape checks
    assert isinstance(result.best_candidate, dict)
    assert "instructions" in result.best_candidate
    assert isinstance(result.best_score, float)
    assert result.original_candidate == seed
    assert result.num_metric_calls > 0
    assert result.num_metric_calls <= 30


def test_optimize_agent_prompts_minimal_flow_with_signature():
    """Run a minimal optimization flow over a tiny categorization dataset.

    We don't expect meaningful optimization here; we just validate a complete run
    finishes and returns a structured result within a small metric budget.
    """

    class Input(BaseModel):
        text: str

    # Build a small categorization dataset (10 items) using pydantic_evals
    def _label_for_token(token: str) -> str:
        return {
            "good": "positive",
            "bad": "negative",
            "ok": "neutral",
        }[token]

    tokens = [
        "good" if i % 3 == 0 else ("bad" if i % 3 == 1 else "ok") for i in range(10)
    ]

    dataset = Dataset(
        cases=[
            Case(
                name=f"case-{i}",
                inputs={"text": f"Sample {i} describing something {tok}."},
                expected_output=_label_for_token(tok),
            )
            for i, tok in enumerate(tokens)
        ]
    )

    # Convert the dataset to GEPA DataInst entries
    trainset: list[DataInst] = [
        DataInstWithInput(
            input=Input(
                text=case.inputs["text"],
            ),
            message_history=None,
            metadata={"label": str(case.expected_output)},
            case_id=case.name or f"case-{i}",
        )
        for i, case in enumerate(dataset.cases)
    ]

    # Agent returns a fixed label; we are not testing real model behavior here
    agent = Agent(
        TestModel(custom_output_text="neutral"),
        instructions=(
            "You are a concise classifier. Output exactly one of: positive, negative, neutral."
        ),
    )
    signature_agent = SignatureAgent(
        agent,
        input_type=Input,
    )

    seed = extract_seed_candidate_with_signature(signature_agent, input_type=Input)

    # Simple metric: 1.0 if predicted label matches expected label, else 0.0
    def metric(
        data_inst: DataInst, output: RolloutOutput[Any]
    ) -> tuple[float, str | None]:
        predicted = (
            str(output.result).strip().lower()
            if output.success and output.result is not None
            else ""
        )
        expected = str(data_inst.metadata.get("label", "")).strip().lower()
        score = 1.0 if predicted == expected else 0.0
        return score, ("Correct" if score == 1.0 else "Incorrect")

    reflection_output = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="instructions",
                optimized_value="Optimized",
            )
        ]
    )
    reflection_model = TestModel(
        custom_output_args=reflection_output.model_dump(mode="python")
    )

    # Keep the budget low; use TestModel() for the reflection model to exercise the full path
    result = optimize_agent_prompts(
        agent=signature_agent,
        trainset=trainset,
        input_type=Input,
        metric=metric,
        reflection_model=reflection_model,
        max_metric_calls=20,
        display_progress_bar=False,
        track_best_outputs=False,
        seed=0,
    )

    # Basic result shape checks
    assert isinstance(result.best_candidate, dict)
    assert "instructions" in result.best_candidate
    assert isinstance(result.best_score, float)
    assert result.original_candidate == seed
    assert result.num_metric_calls > 0
    assert result.num_metric_calls <= 30
