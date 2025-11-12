"""Helpers for exposing the mcp-run-python sandbox as a PydanticAI tool."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Tool
from typing_extensions import TypeAliasType

from mcp_run_python import code_sandbox

JsonValue = TypeAliasType(
    "JsonValue",
    str | bool | int | float | None | list["JsonValue"] | dict[str, "JsonValue"],
)


class SandboxExecutionResult(BaseModel):
    """Structured result returned by the sandbox tool."""

    status: Literal["success", "install-error", "run-error"] = Field(
        description="Overall execution status returned by the sandbox.",
    )
    stdout: str = Field(
        default="",
        description="Combined stdout/stderr emitted by the executed code.",
    )
    return_value: JsonValue | None = Field(
        default=None,
        description="JSON-serialisable return value from the script, if any.",
    )
    error: str | None = Field(
        default=None,
        description="Error message when the sandbox reports an installation or runtime failure.",
    )
    logs: list[str] = Field(
        default_factory=list,
        description="Diagnostic log lines emitted while preparing the sandbox environment.",
    )
    elapsed_seconds: float = Field(
        description="Total wall-clock time spent creating the sandbox and executing the script.",
    )


async def _run_python_in_sandbox(
    code: str,
    *,
    globals: dict[str, JsonValue] | None = None,
) -> SandboxExecutionResult:
    """Execute arbitrary Python inside the mcp-run-python sandbox (stdlib only).

    Args:
        code: Complete Python script to run. Include all required imports and print the final answer.
        globals: Optional JSON-compatible mapping that will be injected as global variables when the script starts.

    Notes:
        Third-party dependencies (like numpy) are intentionally unsupported to keep each run isolated
        and predictable. Use only the Python standard library.
    """

    logs: list[str] = []

    def log_handler(level: str, message: str) -> None:
        logs.append(f"{level.lower()}: {message}")

    started = time.perf_counter()

    try:
        async with code_sandbox(
            dependencies=None,
            log_handler=log_handler,
            allow_networking=False,
        ) as sandbox:
            sandbox_result = await sandbox.eval(code, globals)
    except Exception as exc:  # pragma: no cover - surfaced to the model instead
        elapsed = time.perf_counter() - started
        return SandboxExecutionResult(
            status="run-error",
            stdout="",
            return_value=None,
            error=f"Sandbox failed before execution: {exc}",
            logs=logs,
            elapsed_seconds=elapsed,
        )

    elapsed = time.perf_counter() - started
    stdout = "\n".join(sandbox_result.get("output", []))

    if sandbox_result["status"] == "success":
        return SandboxExecutionResult(
            status="success",
            stdout=stdout,
            return_value=sandbox_result.get("return_value"),
            error=None,
            logs=logs,
            elapsed_seconds=elapsed,
        )

    return SandboxExecutionResult(
        status=sandbox_result["status"],
        stdout=stdout,
        return_value=None,
        error=sandbox_result.get("error"),
        logs=logs,
        elapsed_seconds=elapsed,
    )


run_python_tool = Tool(
    _run_python_in_sandbox,
    name="run_python",
    description=(
        "Execute Python code inside an isolated sandbox using only the Python standard library. "
        "Provide fully self-contained scripts that print their final answer."
    ),
)

__all__ = ["JsonValue", "SandboxExecutionResult", "run_python_tool"]
