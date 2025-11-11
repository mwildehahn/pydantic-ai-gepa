import asyncio
import json
import math
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import pprint
from typing import Any

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import KnownModelName, Model, infer_model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_evals import Case, Dataset

from pydantic_ai_gepa import InspectingModel, InspectionAborted
from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapter
from pydantic_ai_gepa.cache import CacheManager
from pydantic_ai_gepa.gepa_graph import GepaConfig, GepaResult, optimize
from pydantic_ai_gepa.signature_agent import SignatureAgent
from pydantic_ai_gepa.types import DataInstWithInput, MetricResult, RolloutOutput

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


def evaluate_code(code: str) -> tuple[float, str]:
    """Evaluate Python code inside the constrained math sandbox.

    The code can contain multiple statements. The last line should evaluate to a numeric result.
    """
    stripped = code.strip()
    if not stripped:
        raise ValueError("Code is empty.")

    # Create a local namespace for execution
    local_namespace: dict[str, Any] = {}

    try:
        # Try to compile as exec (allows multiple statements)
        compiled = compile(stripped, "<python_eval>", "exec")
        exec(compiled, EVAL_GLOBALS, local_namespace)

        # Get the result from the last expression or from a 'result' variable
        if "result" in local_namespace:
            result = local_namespace["result"]
        else:
            # If no 'result' variable, try to evaluate the last line as an expression
            lines = stripped.split("\n")
            last_line = lines[-1].strip()
            if last_line and not last_line.startswith("#"):
                try:
                    result = eval(last_line, EVAL_GLOBALS, local_namespace)
                except Exception:
                    raise ValueError(
                        "Code must end with an expression or assign the final value to 'result'"
                    )
            else:
                raise ValueError(
                    "Code must end with an expression or assign the final value to 'result'"
                )
    except SyntaxError as exc:
        raise ValueError(f"Code must be valid Python: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Code raised an error: {exc}") from exc

    if isinstance(result, bool):
        raise ValueError("Code returned a boolean, expected a numeric result.")

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
    problem: str = Field(
        description="A math problem that needs an exact numeric answer."
    )


class MathProblemOutput(BaseModel):
    """The solved value and the code that produced it."""

    explanation: str = Field(
        description="Two sentences max summarizing how the code solves the problem."
    )
    expression: str = Field(
        description="The Python code evaluated via python_eval (can be multi-line)."
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
        "feedback": "Use the combinatorics function from the math module to compute binomial coefficients directly.",
    },
    {
        "name": "digit-sum-2-200",
        "prompt": "Compute the sum of the digits of 2 raised to the 200th power.",
        "expression": "sum(int(d) for d in str(2 ** 200))",
        "expected": 256.0,
        "tolerance": 1e-9,
        "feedback": "Convert the large integer to a string first, then sum each digit character after converting back to int.",
    },
    {
        "name": "primorial-product",
        "prompt": "Multiply the primes [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] together.",
        "expression": "math.prod([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])",
        "expected": 6_469_693_230.0,
        "tolerance": 1e-9,
        "feedback": "Use the product aggregation function from the math module to multiply list elements.",
    },
    # TIER 2: Conceptual ambiguity cases (boundary interpretation)
    {
        "name": "range-ambiguity-between",
        "prompt": "Sum all integers between 10 and 20.",
        "expression": "sum(range(11, 20))",
        "expected": 135.0,
        "tolerance": 1e-9,
        "feedback": "The phrase 'between A and B' typically excludes both endpoints. Verify whether the count matches your interpretation.",
    },
    {
        "name": "range-ambiguity-from-through",
        "prompt": "Calculate the average of squares from 5 through 15.",
        "expression": "sum(n**2 for n in range(5, 16)) / len(range(5, 16))",
        "expected": 110.0,
        "tolerance": 1e-9,
        "feedback": "The phrase 'from A through B' indicates inclusive bounds. Check that your range includes both endpoints.",
    },
    {
        "name": "implicit-inclusive-up-to",
        "prompt": "Find the product of all even numbers up to 12.",
        "expression": "math.prod(range(2, 13, 2))",
        "expected": 46080.0,
        "tolerance": 1e-9,
        "feedback": "The phrase 'up to N' is ambiguous—it may include or exclude N. Verify against the expected result which interpretation is correct.",
    },
    {
        "name": "rounding-specification",
        "prompt": "Approximate the square root of 50 to the nearest integer.",
        "expression": "round(math.sqrt(50))",
        "expected": 7.0,
        "tolerance": 1e-9,
        "feedback": "Use the rounding function explicitly when the problem requests rounding to a specific precision.",
    },
    {
        "name": "floor-vs-truncate",
        "prompt": "What is 100 divided by 7, rounded down?",
        "expression": "math.floor(100 / 7)",
        "expected": 14.0,
        "tolerance": 1e-9,
        "feedback": "Rounded down means floor division. Use the appropriate math function for floor operations.",
    },
    {
        "name": "mixed-boundaries",
        "prompt": "Sum integers greater than 5 and less than or equal to 15.",
        "expression": "sum(range(6, 16))",
        "expected": 105.0,
        "tolerance": 1e-9,
        "feedback": "Pay attention to strict inequalities (>) versus inclusive inequalities (≤). Translate each bound correctly.",
    },
    # TIER 3: Multi-step reasoning cases
    {
        "name": "conditional-prime-product",
        "prompt": "Find the sum of all primes less than 50, then multiply that sum by the largest prime less than 50.",
        "expression": "(lambda primes: sum(primes) * max(primes))([n for n in range(2, 50) if all(n % d for d in range(2, int(n**0.5) + 1))])",
        "expected": 15416.0,
        "tolerance": 1e-9,
        "feedback": "Break the problem into steps: first identify all primes in the range, then compute the sum and find the maximum, then multiply them.",
    },
    {
        "name": "nested-digit-sum",
        "prompt": "Calculate the sum of the digits of 15 factorial.",
        "expression": "sum(int(d) for d in str(math.factorial(15)))",
        "expected": 45.0,
        "tolerance": 1e-9,
        "feedback": "Compute the factorial first, convert to string, then sum the individual digit characters.",
    },
    {
        "name": "tribonacci-20",
        "prompt": "Find the 20th Tribonacci number, where T(0)=0, T(1)=1, T(2)=1, and T(n)=T(n-1)+T(n-2)+T(n-3).",
        "expression": "(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(17)], t[-1]][2])()",
        "expected": 35890.0,
        "tolerance": 1e-9,
        "feedback": "Iteratively compute the sequence using a list to track the last three values, updating as you progress.",
    },
    {
        "name": "gcd-lcm-chain",
        "prompt": "Compute the LCM of 12, 18, and 24, then find the GCD of that result and 144.",
        "expression": "math.gcd((lambda a, b: abs(a * b) // math.gcd(a, b))((lambda a, b: abs(a * b) // math.gcd(a, b))(12, 18), 24), 144)",
        "expected": 72.0,
        "tolerance": 1e-9,
        "feedback": "Compute LCM step-by-step for pairs using the formula LCM(a,b) = |a*b|/GCD(a,b), then apply GCD to the final result.",
    },
    {
        "name": "totient-composite",
        "prompt": "Calculate Euler's totient function φ(72)—the count of integers from 1 to 72 that are coprime with 72.",
        "expression": "sum(1 for k in range(1, 73) if math.gcd(k, 72) == 1)",
        "expected": 24.0,
        "tolerance": 1e-9,
        "feedback": "Count how many integers in the range have a GCD of 1 with the target number.",
    },
    {
        "name": "alternating-sum-squares",
        "prompt": "Compute the alternating sum 1² - 2² + 3² - 4² + ... + 19² - 20².",
        "expression": "sum(((-1) ** (n + 1)) * (n ** 2) for n in range(1, 21))",
        "expected": -210.0,
        "tolerance": 1e-9,
        "feedback": "Use a sign factor that alternates based on the index: positive for odd indices, negative for even.",
    },
    # TIER 4: Adversarial edge cases
    {
        "name": "precision-trap-large-factorial",
        "prompt": "What is 100 factorial divided by 99 factorial?",
        "expression": "math.factorial(100) // math.factorial(99)",
        "expected": 100.0,
        "tolerance": 1e-9,
        "feedback": "Notice the mathematical identity: n! / (n-1)! = n. Avoid computing huge factorials separately if simplification is possible.",
    },
    {
        "name": "empty-range-edge",
        "prompt": "Sum all integers from 20 to 10.",
        "expression": "sum(range(20, 10))",
        "expected": 0.0,
        "tolerance": 1e-9,
        "feedback": "When the start exceeds the stop in a range, the result is an empty sequence. The sum of an empty sequence is zero.",
    },
    {
        "name": "degenerate-average",
        "prompt": "Find the average of all multiples of 7 between 100 and 105.",
        "expression": "sum(range(105, 106, 7)) / max(len(range(105, 106, 7)), 1)",
        "expected": 105.0,
        "tolerance": 1e-9,
        "feedback": "Only one multiple exists in this narrow range. Ensure you handle single-element averages correctly.",
    },
    {
        "name": "sign-heavy-expression",
        "prompt": "Calculate (-1)^50 + (-1)^51 + (-1)^52.",
        "expression": "(-1)**50 + (-1)**51 + (-1)**52",
        "expected": 1.0,
        "tolerance": 1e-9,
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
        },
        case_id=dataset_case.name or f"case-{index}",
    )
    for index, dataset_case in enumerate(dataset.cases)
]

agent_model = InspectingModel(infer_model("openai:gpt-5-mini"))
# agent_model = infer_model("openai:gpt-5-mini")

agent = Agent(
    model=agent_model,
    instructions=(
        "Solve math problems using the python_eval tool. "
        "Write clear Python code with intermediate steps if needed (e.g., factorial = math.factorial(15), then sum digits). "
        "Call python_eval ONCE with your code, then immediately return the final result."
    ),
    output_type=MathProblemOutput,
)


class PythonEvalOutput(BaseModel):
    numeric_result: float


@agent.tool(
    name="python_eval",
    description=(
        "Execute Python code to compute a numeric answer. "
        "You can write multiple lines of code with intermediate variables. "
        "The last line should be an expression that evaluates to the final numeric result, "
        "or assign the result to a variable named 'result'. "
        "You have access to the math module (math.factorial, math.comb, math.sqrt, etc.), "
        "basic built-ins (sum, min, max, range, etc.), and Decimal/Fraction types."
    ),
)
def python_eval_tool(
    ctx: RunContext[Any],
    code: str,
) -> PythonEvalOutput:
    """Evaluate deterministic numeric Python code using the math module."""
    numeric, _ = evaluate_code(code)
    return PythonEvalOutput(numeric_result=numeric)


signature_agent = SignatureAgent(
    agent,
    input_type=MathProblemInput,
    optimize_tools=True,
)


def metric(
    data_inst: DataInstWithInput[MathProblemInput],
    output: RolloutOutput[MathProblemOutput],
) -> MetricResult:
    if not output.success or output.result is None:
        return MetricResult(
            score=0.0,
            feedback=output.error_message or "Agent failed to produce an output.",
        )

    predicted_output = output.result
    predicted = predicted_output.answer
    expression = (predicted_output.expression or "").strip()
    tolerance = data_inst.metadata.get("tolerance", 1e-9)
    target = data_inst.metadata.get("expected_answer")
    base_feedback = data_inst.metadata.get("feedback")
    ideal_expression = data_inst.metadata.get("ideal_expression")

    if not expression:
        hint = "Include the python_eval code you executed."
        if ideal_expression:
            hint = f"{hint} For reference, one valid approach is `{ideal_expression}`."
        prefix = f"{base_feedback} " if base_feedback else ""
        return MetricResult(
            score=0.0,
            feedback=f"{prefix}Missing code used for python_eval. {hint}",
        )

    try:
        evaluated, display = evaluate_code(expression)
    except Exception as exc:  # noqa: BLE001
        prefix = f"{base_feedback} " if base_feedback else ""
        return MetricResult(
            score=0.0,
            feedback=f"{prefix}Code could not be evaluated: {exc}",
        )

    answer_gap = abs(predicted - evaluated)
    max_internal_gap = max(tolerance, 1e-9)
    if answer_gap > max_internal_gap:
        prefix = f"{base_feedback} " if base_feedback else ""
        return MetricResult(
            score=0.0,
            feedback=(
                f"{prefix}Reported answer {predicted} disagrees with code result {display}."
            ),
        )

    if target is None:
        return MetricResult(score=0.0, feedback="Missing reference target.")

    target_gap = abs(evaluated - target)
    effective_tolerance = max(tolerance, 1e-9)
    if target_gap <= effective_tolerance:
        return MetricResult(score=1.0, feedback="Exact match within tolerance.")

    normalized_error = target_gap / max(abs(target), 1.0)
    score = max(0.0, 1.0 - min(normalized_error * 10, 1.0))
    base = base_feedback or "Use python_eval to re-check the computation."
    hint = (
        f"Code result {display} deviates from target {target} by {target_gap:.6g}; "
        "tighten the computation or adjust rounding."
    )
    if ideal_expression and ideal_expression != expression:
        hint += f" A reliable approach is `{ideal_expression}`."
    feedback = f"{base} {hint}"
    return MetricResult(score=score, feedback=feedback)


async def run_math_tools_optimization(
    trainset: Sequence[DataInstWithInput[MathProblemInput]],
    valset: Sequence[DataInstWithInput[MathProblemInput]],
    reflection_model: Model | KnownModelName | str,
) -> GepaResult:
    cache_manager = CacheManager(
        cache_dir=".gepa_cache",
        enabled=True,
        verbose=True,
    )

    adapter = AgentAdapter(
        agent=signature_agent,
        metric=metric,
        input_type=MathProblemInput,
        cache_manager=cache_manager,
    )

    config = GepaConfig(
        max_evaluations=150,
        component_selector="all",
        max_concurrent_evaluations=10,
        enable_parallel_reflection=False,
        reflection_model=reflection_model,
    )

    return await optimize(
        adapter=adapter,
        config=config,
        trainset=trainset,
        valset=valset,
    )


async def main() -> None:
    output_dir = Path("optimization_results")
    output_dir.mkdir(exist_ok=True)

    base_reflection_model = OpenAIResponsesModel(
        model_name="gpt-5",
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort="medium",
            openai_reasoning_summary="detailed",
            openai_text_verbosity="medium",
        ),
    )
    reflection_model = InspectingModel(base_reflection_model)

    # 60/40 train/val split to better detect overfitting
    split_index = int(len(signature_dataset) * 0.6)
    trainset = signature_dataset[:split_index]
    valset = signature_dataset[split_index:]

    try:
        result = await run_math_tools_optimization(trainset, valset, reflection_model)
    except InspectionAborted as exc:
        snapshot = exc.snapshot
        print("\n🔎 OpenAI request intercepted for inspection. Payload:")
        pprint.pprint(snapshot)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"math_tools_optimization_{timestamp}.json"

    with output_file.open("w") as file:
        json.dump(result.model_dump(), file, indent=2)

    print(f"\n✅ Optimization result saved to: {output_file}")
    if result.original_score is not None:
        print(f"   Original score: {result.original_score:.4f}")
    else:
        print("   Original score: N/A")
    if result.best_score is not None:
        print(f"   Best score: {result.best_score:.4f}")
    else:
        print("   Best score: N/A")
    print(f"   Iterations: {result.iterations}")
    print(f"   Metric calls: {result.total_evaluations}")
    improvement = result.relative_improvement()
    if improvement is not None:
        print(f"   Improvement: {improvement * 100:.2f}%")
    else:
        print("   Improvement: N/A")


if __name__ == "__main__":
    asyncio.run(main())
