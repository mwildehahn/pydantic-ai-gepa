"""Simple test to verify MCP math tools integration."""
import asyncio

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from mcp_run_python import async_prepare_deno_env

logfire.configure(console=False)
logfire.instrument_pydantic_ai()
logfire.instrument_mcp()


class MathResult(BaseModel):
    explanation: str = Field(description="Brief explanation of the solution")
    code: str = Field(description="The Python code used")
    answer: float = Field(description="The numeric answer")


async def main():
    """Test the MCP-based math agent."""
    test_problem = "What is 100 choose 5?"

    print(f"Testing problem: {test_problem}")
    print("-" * 50)

    # Set up Deno environment and MCP server
    async with async_prepare_deno_env('stdio') as deno_env:
        mcp_server = MCPServerStdio('deno', args=deno_env.args, cwd=deno_env.cwd, timeout=30)

        agent = Agent(
            model='openai:gpt-5-mini',
            instructions=(
                "Solve math problems using the run_python tool from MCP. "
                "Write complete Python scripts with all necessary imports. "
                "You have access to ALL Python libraries. "
                "Print the final answer."
            ),
            output_type=MathResult,
            toolsets=[mcp_server],
        )

        async with agent:
            result = await agent.run(test_problem)

        print(f"\nExplanation: {result.output.explanation}")
        print(f"\nCode:\n{result.output.code}")
        print(f"\nAnswer: {result.output.answer}")
        print(f"\nExpected: 75287520")


if __name__ == '__main__':
    asyncio.run(main())
