"""Simple test to verify sandbox-based math tool integration."""

import asyncio

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from utils import run_python_tool

logfire.configure(console=False)
logfire.instrument_pydantic_ai()


class MathResult(BaseModel):
    explanation: str = Field(description="Brief explanation of the solution")
    code: str = Field(description="The Python code used")
    answer: float = Field(description="The numeric answer")


async def main():
    """Test the sandbox-backed math agent."""
    test_problem = "What is 100 choose 5?"

    print(f"Testing problem: {test_problem}")
    print("-" * 50)

    agent = Agent(
        model="openai:gpt-5-mini",
        instructions=(
            "Solve math problems by calling the `run_python` sandbox tool. "
            "Write complete Python scripts with all necessary imports, print the final answer, "
            "and stick to the Python standard library (no third-party packages)."
        ),
        output_type=MathResult,
        tools=[run_python_tool],
    )

    async with agent:
        result = await agent.run(test_problem)

    print(f"\nExplanation: {result.output.explanation}")
    print(f"\nCode:\n{result.output.code}")
    print(f"\nAnswer: {result.output.answer}")
    print(f"\nExpected: 75287520")


if __name__ == "__main__":
    asyncio.run(main())
