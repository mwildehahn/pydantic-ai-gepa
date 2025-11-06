import json
import math
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_evals import Case, Dataset

from pydantic_ai_gepa.runner import optimize_agent_prompts
from pydantic_ai_gepa.signature_agent import SignatureAgent
from pydantic_ai_gepa.types import DataInstWithInput, RolloutOutput

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)


ALLOWED_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
}

EVAL_GLOBALS: dict[str, Any] = {
    "__builtins__": ALLOWED_BUILTINS,
    "Decimal": Decimal,
    "Fraction": Fraction,
    "math": math,
}


def evaluate_expression(expression: str) -> tuple[float, str]:
    """Evaluate a Python expression inside the constrained math sandbox."""
    stripped = expression.strip()
    if not stripped:
        raise ValueError("Expression is empty.")

    try:
        compiled = compile(stripped, "<python_eval>", "eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Expression must be a valid Python expression: {exc}"
        ) from exc

    try:
        result = eval(compiled, EVAL_GLOBALS, {})
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Expression raised an error: {exc}") from exc

    if isinstance(result, bool):
        raise ValueError("Expression returned a boolean, expected a numeric result.")

    if isinstance(result, Decimal):
        numeric = float(result)
        display = format(result, "f")
    elif isinstance(result, Fraction):
        numeric = float(result)
        display = f"{result.numerator}/{result.denominator}"
    elif isinstance(result, (int, float)):
        numeric = float(result)
        display = str(result)
    else:
        raise TypeError(
            f"Unsupported result type {type(result).__name__}; expected numeric output."
        )

    return numeric, display


class MathProblemInput(BaseModel):
    """A structured math prompt describing the calculation to perform."""

    problem: str = Field(
        description="The math task that needs an exact numeric answer."
    )


class MathProblemOutput(BaseModel):
    """The solved value and the expression that produced it."""

    explanation: str = Field(
        description="Two sentences max summarizing how the expression solves the problem."
    )
    expression: str = Field(
        description="The Python expression evaluated via python_eval."
    )
    answer: float = Field(description="Numeric answer rounded only if necessary.")


CASE_DEFINITIONS: list[dict[str, Any]] = [
    # TIER 1: Baseline cases (clear single operations)
    {
        "name": "comb-100-5",
        "prompt": "Compute 100 choose 5.",
        "expression": "math.comb(100, 5)",
        "expected": 75_287_520.0,
        "tolerance": 1e-9,
        "difficulty_tier": 1,
        "feedback": "Use the combinatorics function from the math module to compute binomial coefficients directly.",
    },
    {
        "name": "digit-sum-2-200",
        "prompt": "Compute the sum of the digits of 2 raised to the 200th power.",
        "expression": "sum(int(d) for d in str(2 ** 200))",
        "expected": 256.0,
        "tolerance": 1e-9,
        "difficulty_tier": 1,
        "feedback": "Convert the large integer to a string first, then sum each digit character after converting back to int.",
    },
    {
        "name": "primorial-product",
        "prompt": "Multiply the primes [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] together.",
        "expression": "math.prod([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])",
        "expected": 6_469_693_230.0,
        "tolerance": 1e-9,
        "difficulty_tier": 1,
        "feedback": "Use the product aggregation function from the math module to multiply list elements.",
    },
    # TIER 2: Conceptual ambiguity cases (boundary interpretation)
    {
        "name": "range-ambiguity-between",
        "prompt": "Sum all integers between 10 and 20.",
        "expression": "sum(range(11, 20))",
        "expected": 135.0,
        "tolerance": 1e-9,
        "difficulty_tier": 2,
        "feedback": "The phrase 'between A and B' typically excludes both endpoints. Verify whether the count matches your interpretation.",
    },
    {
        "name": "range-ambiguity-from-through",
        "prompt": "Calculate the average of squares from 5 through 15.",
        "expression": "sum(n**2 for n in range(5, 16)) / len(range(5, 16))",
        "expected": 110.0,
        "tolerance": 1e-9,
        "difficulty_tier": 2,
        "feedback": "The phrase 'from A through B' indicates inclusive bounds. Check that your range includes both endpoints.",
    },
    {
        "name": "implicit-inclusive-up-to",
        "prompt": "Find the product of all even numbers up to 12.",
        "expression": "math.prod(range(2, 13, 2))",
        "expected": 46080.0,
        "tolerance": 1e-9,
        "difficulty_tier": 2,
        "feedback": "The phrase 'up to N' is ambiguous—it may include or exclude N. Verify against the expected result which interpretation is correct.",
    },
    {
        "name": "rounding-specification",
        "prompt": "Approximate the square root of 50 to the nearest integer.",
        "expression": "round(math.sqrt(50))",
        "expected": 7.0,
        "tolerance": 1e-9,
        "difficulty_tier": 2,
        "feedback": "Use the rounding function explicitly when the problem requests rounding to a specific precision.",
    },
    {
        "name": "floor-vs-truncate",
        "prompt": "What is 100 divided by 7, rounded down?",
        "expression": "math.floor(100 / 7)",
        "expected": 14.0,
        "tolerance": 1e-9,
        "difficulty_tier": 2,
        "feedback": "Rounded down means floor division. Use the appropriate math function for floor operations.",
    },
    {
        "name": "mixed-boundaries",
        "prompt": "Sum integers greater than 5 and less than or equal to 15.",
        "expression": "sum(range(6, 16))",
        "expected": 105.0,
        "tolerance": 1e-9,
        "difficulty_tier": 2,
        "feedback": "Pay attention to strict inequalities (>) versus inclusive inequalities (≤). Translate each bound correctly.",
    },
    # TIER 3: Multi-step reasoning cases
    {
        "name": "conditional-prime-product",
        "prompt": "Find the sum of all primes less than 50, then multiply that sum by the largest prime less than 50.",
        "expression": "(lambda primes: sum(primes) * max(primes))([n for n in range(2, 50) if all(n % d for d in range(2, int(n**0.5) + 1))])",
        "expected": 15416.0,
        "tolerance": 1e-9,
        "difficulty_tier": 3,
        "feedback": "Break the problem into steps: first identify all primes in the range, then compute the sum and find the maximum, then multiply them.",
    },
    {
        "name": "nested-digit-sum",
        "prompt": "Calculate the sum of the digits of 15 factorial.",
        "expression": "sum(int(d) for d in str(math.factorial(15)))",
        "expected": 45.0,
        "tolerance": 1e-9,
        "difficulty_tier": 3,
        "feedback": "Compute the factorial first, convert to string, then sum the individual digit characters.",
    },
    {
        "name": "tribonacci-20",
        "prompt": "Find the 20th Tribonacci number, where T(0)=0, T(1)=1, T(2)=1, and T(n)=T(n-1)+T(n-2)+T(n-3).",
        "expression": "(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(17)], t[-1]][2])()",
        "expected": 35890.0,
        "tolerance": 1e-9,
        "difficulty_tier": 3,
        "feedback": "Iteratively compute the sequence using a list to track the last three values, updating as you progress.",
    },
    {
        "name": "gcd-lcm-chain",
        "prompt": "Compute the LCM of 12, 18, and 24, then find the GCD of that result and 144.",
        "expression": "math.gcd((lambda a, b: abs(a * b) // math.gcd(a, b))((lambda a, b: abs(a * b) // math.gcd(a, b))(12, 18), 24), 144)",
        "expected": 72.0,
        "tolerance": 1e-9,
        "difficulty_tier": 3,
        "feedback": "Compute LCM step-by-step for pairs using the formula LCM(a,b) = |a*b|/GCD(a,b), then apply GCD to the final result.",
    },
    {
        "name": "totient-composite",
        "prompt": "Calculate Euler's totient function φ(72)—the count of integers from 1 to 72 that are coprime with 72.",
        "expression": "sum(1 for k in range(1, 73) if math.gcd(k, 72) == 1)",
        "expected": 24.0,
        "tolerance": 1e-9,
        "difficulty_tier": 3,
        "feedback": "Count how many integers in the range have a GCD of 1 with the target number.",
    },
    {
        "name": "alternating-sum-squares",
        "prompt": "Compute the alternating sum 1² - 2² + 3² - 4² + ... + 19² - 20².",
        "expression": "sum(((-1) ** (n + 1)) * (n ** 2) for n in range(1, 21))",
        "expected": -210.0,
        "tolerance": 1e-9,
        "difficulty_tier": 3,
        "feedback": "Use a sign factor that alternates based on the index: positive for odd indices, negative for even.",
    },
    # TIER 4: Adversarial edge cases
    {
        "name": "precision-trap-large-factorial",
        "prompt": "What is 100 factorial divided by 99 factorial?",
        "expression": "math.factorial(100) // math.factorial(99)",
        "expected": 100.0,
        "tolerance": 1e-9,
        "difficulty_tier": 4,
        "feedback": "Notice the mathematical identity: n! / (n-1)! = n. Avoid computing huge factorials separately if simplification is possible.",
    },
    {
        "name": "empty-range-edge",
        "prompt": "Sum all integers from 20 to 10.",
        "expression": "sum(range(20, 10))",
        "expected": 0.0,
        "tolerance": 1e-9,
        "difficulty_tier": 4,
        "feedback": "When the start exceeds the stop in a range, the result is an empty sequence. The sum of an empty sequence is zero.",
    },
    {
        "name": "degenerate-average",
        "prompt": "Find the average of all multiples of 7 between 100 and 105.",
        "expression": "sum(range(105, 106, 7)) / max(len(range(105, 106, 7)), 1)",
        "expected": 105.0,
        "tolerance": 1e-9,
        "difficulty_tier": 4,
        "feedback": "Only one multiple exists in this narrow range. Ensure you handle single-element averages correctly.",
    },
    {
        "name": "sign-heavy-expression",
        "prompt": "Calculate (-1)^50 + (-1)^51 + (-1)^52.",
        "expression": "(-1)**50 + (-1)**51 + (-1)**52",
        "expected": 1.0,
        "tolerance": 1e-9,
        "difficulty_tier": 4,
        "feedback": "Even powers of -1 yield 1, odd powers yield -1. Sum the results directly.",
    },
]

dataset = Dataset[MathProblemInput, MathProblemOutput](
    cases=[
        Case(
            name=case["name"],
            inputs=MathProblemInput(problem=case["prompt"]),
            expected_output=MathProblemOutput(
                answer=case["expected"],
                expression=case["expression"],
                explanation=case["feedback"],
            ),
        )
        for case in CASE_DEFINITIONS
    ]
)

signature_dataset = [
    DataInstWithInput[MathProblemInput](
        input=dataset_case.inputs,
        message_history=None,
        metadata={
            "expected_answer": dataset_case.expected_output.answer
            if dataset_case.expected_output
            else None,
            "tolerance": CASE_DEFINITIONS[index]["tolerance"],
            "feedback": CASE_DEFINITIONS[index]["feedback"],
            "ideal_expression": dataset_case.expected_output.expression
            if dataset_case.expected_output
            else None,
            "difficulty_tier": CASE_DEFINITIONS[index]["difficulty_tier"],
        },
        case_id=dataset_case.name or f"case-{index}",
    )
    for index, dataset_case in enumerate(dataset.cases)
]

agent = Agent(
    model="openai:gpt-5-nano-2025-08-07",
    instructions=(
        "Solve deterministic math problems. Always use the python_eval tool to compute the numeric answer before "
        "responding. Return JSON with fields: answer, expression, explanation."
    ),
    output_type=MathProblemOutput,
)


class PythonEvalOutput(BaseModel):
    numeric_result: float


@agent.tool(
    name="python_eval",
    description="Execute a pure Python expression using math helpers to recover precise numeric answers.",
)
def python_eval_tool(
    ctx: RunContext[Any],
    expression: str,
) -> PythonEvalOutput:
    """Evaluate deterministic numeric Python expressions using the math module."""
    numeric, _ = evaluate_expression(expression)
    return PythonEvalOutput(numeric_result=numeric)


signature_agent = SignatureAgent(
    agent,
    input_type=MathProblemInput,
    optimize_tools=True,
)


def metric(
    data_inst: DataInstWithInput[MathProblemInput],
    output: RolloutOutput[MathProblemOutput],
) -> tuple[float, str | None]:
    if not output.success or output.result is None:
        return 0.0, output.error_message or "Agent failed to produce an output."

    predicted_output = output.result
    predicted = predicted_output.answer
    expression = (predicted_output.expression or "").strip()
    tolerance = data_inst.metadata.get("tolerance", 1e-9)
    target = data_inst.metadata.get("expected_answer")
    base_feedback = data_inst.metadata.get("feedback")
    ideal_expression = data_inst.metadata.get("ideal_expression")

    if not expression:
        hint = "Include the python_eval expression you executed."
        if ideal_expression:
            hint = f"{hint} For reference, one valid approach is `{ideal_expression}`."
        prefix = f"{base_feedback} " if base_feedback else ""
        return 0.0, f"{prefix}Missing expression used for python_eval. {hint}"

    try:
        evaluated, display = evaluate_expression(expression)
    except Exception as exc:  # noqa: BLE001
        prefix = f"{base_feedback} " if base_feedback else ""
        return 0.0, f"{prefix}Expression could not be evaluated: {exc}"

    answer_gap = abs(predicted - evaluated)
    max_internal_gap = max(tolerance, 1e-9)
    if answer_gap > max_internal_gap:
        prefix = f"{base_feedback} " if base_feedback else ""
        return (
            0.0,
            f"{prefix}Reported answer {predicted} disagrees with expression result {display}.",
        )

    if target is None:
        return 0.0, "Missing reference target."

    target_gap = abs(evaluated - target)
    effective_tolerance = max(tolerance, 1e-9)
    if target_gap <= effective_tolerance:
        return 1.0, "Exact match within tolerance."

    normalized_error = target_gap / max(abs(target), 1.0)
    score = max(0.0, 1.0 - min(normalized_error * 10, 1.0))
    base = base_feedback or "Use python_eval to re-check the computation."
    hint = (
        f"Expression result {display} deviates from target {target} by {target_gap:.6g}; "
        "tighten the computation or adjust rounding."
    )
    if ideal_expression and ideal_expression != expression:
        hint += f" A reliable expression is `{ideal_expression}`."
    feedback = f"{base} {hint}"
    return score, feedback


if __name__ == "__main__":
    output_dir = Path("optimization_results")
    output_dir.mkdir(exist_ok=True)

    reflection_model = OpenAIResponsesModel(
        model_name="gpt-5",
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort="medium",
            openai_reasoning_summary="detailed",
            openai_text_verbosity="medium",
        ),
    )

    # 60/40 train/val split to better detect overfitting
    split_index = int(len(signature_dataset) * 0.6)
    trainset = signature_dataset[:split_index]
    valset = signature_dataset[split_index:]

    result = optimize_agent_prompts(
        agent=signature_agent,
        trainset=trainset,
        valset=valset,
        metric=metric,
        input_type=MathProblemInput,
        module_selector="all",
        reflection_model=reflection_model,
        max_metric_calls=150,
        display_progress_bar=True,
        track_best_outputs=True,
        enable_cache=True,
        cache_dir=".gepa_cache",
        cache_verbose=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"math_tools_optimization_{timestamp}.json"

    with output_file.open("w") as file:
        json.dump(result.model_dump(), file, indent=2)

    print(f"\n✅ Optimization result saved to: {output_file}")
    print(f"   Best score: {result.best_score:.4f}")
    print(f"   Iterations: {result.num_iterations}")
    print(f"   Metric calls: {result.num_metric_calls}")
    improvement = result.improvement_ratio()
    if improvement is not None:
        print(f"   Improvement: {improvement * 100:.2f}%")
