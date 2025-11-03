"""Wrapper agent for signature-based prompts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
)
from typing import TYPE_CHECKING, Any, overload, cast

from typing_extensions import Never

from pydantic import BaseModel
from pydantic_ai import messages as _messages, models, usage as _usage
from pydantic_ai.agent import AgentRunResult, EventStreamHandler, WrapperAgent
from pydantic_ai.agent.abstract import RunOutputDataT, Instructions
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, DeferredToolResults
from pydantic_ai.toolsets import AbstractToolset

from .signature import generate_system_instructions, generate_user_content

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent

UserPromptInput = Sequence[_messages.UserContent] | _messages.UserContent | str


class SignatureAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    """Wrapper agent that enables signature-based prompts.

    This wrapper allows you to run agents using structured input models instead
    of raw prompt strings. It handles conversion from Pydantic models to prompt
    format and can apply GEPA optimizations. When the wrapped agent already has
    an output type configured, the SignatureAgent will reuse it automatically.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.temporal import TemporalAgent
        from pydantic_ai_gepa import SignatureAgent
        from pydantic import BaseModel, Field

        class Query(BaseModel):
            '''Answer questions about geography'''

            question: str = Field(description="The geography question to answer")
            context: str = Field(description="Additional context if needed")

        class GeographyAnswer(BaseModel):
            answer: str
            confidence: str

        # Create base agent
        agent = Agent(
            'openai:gpt-4o',
            instructions="You're an expert in geography.",
            name='geography',
        )

        # Wrap with Temporal if needed
        temporal_agent = TemporalAgent(agent)

        # Add signature support (output_type inferred from wrapped agent)
        signature_agent = SignatureAgent(
            temporal_agent,
            input_type=Query,
        )

        # Run with signature
        sig = Query(
            question="What's the capital of France?",
            context="Focus on current political capital"
        )
        result = await signature_agent.run_signature(sig)
        ```
    """

    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        input_type: type[BaseModel],
        output_type: OutputSpec[OutputDataT] | type[OutputDataT] | None = None,
        *,
        append_instructions: bool = True,
    ):
        """Initialize the SignatureAgent wrapper.

        Args:
            wrapped: The agent to wrap (can be any AbstractAgent, including TemporalAgent).
            input_type: The Pydantic model class representing the structured input schema.
            output_type: Optional output type or spec expected from the wrapped agent.
            append_instructions: If True, append signature instructions to the agent's instructions.
        """
        if not issubclass(input_type, BaseModel):
            raise TypeError(
                f"input_type must subclass BaseModel, got {input_type!r}"
            )

        inferred_output_type = (
            output_type
            if output_type is not None
            else getattr(wrapped, "output_type", None)
        )
        if inferred_output_type is None:
            raise TypeError(
                "SignatureAgent requires an output_type. Provide one explicitly, "
                "configure the wrapped agent with an output_type, or supply one "
                "per-call when invoking signature runs."
            )

        super().__init__(wrapped)
        self.append_instructions = append_instructions
        self._input_model = input_type
        self._default_output_type = inferred_output_type

    @property
    def input_type(self) -> type[BaseModel]:
        """Return the input model class this agent expects."""
        return self._input_model

    @property
    def output_type(self) -> OutputSpec[OutputDataT] | type[OutputDataT]:
        """Return the default output type used by the agent."""
        return self._default_output_type

    def _require_input_instance(self, signature: BaseModel) -> None:
        """Ensure the provided signature instance matches the configured input type."""
        if not isinstance(signature, self._input_model):
            raise TypeError(
                f"Expected signature of type {self._input_model.__name__}, "
                f"got {signature.__class__.__name__}"
            )

    def _prepare_user_content(
        self,
        signature: BaseModel,
    ) -> Sequence[_messages.UserContent]:
        """Extract user content from a signature.

        Args:
            signature: The structured input instance to convert.

        Returns:
            The user content without system instructions.
        """
        return generate_user_content(signature)

    def _prepare_system_instructions(
        self,
        signature: BaseModel,
        candidate: dict[str, str] | None = None,
    ) -> str | None:
        """Extract system instructions from a signature.

        Args:
            signature: The structured input instance to convert.

        Returns:
            The system instructions string or None if empty.
        """
        if not self.append_instructions:
            return None

        return generate_system_instructions(signature, candidate=candidate)

    def _compose_instructions_override(
        self,
        base_instructions: Instructions[AgentDepsT] | None,
        system_instructions: str | None,
    ) -> Instructions[AgentDepsT] | None:
        """Combine candidate/base instructions with signature instructions."""
        if system_instructions:
            if base_instructions:
                if isinstance(base_instructions, Sequence) and not isinstance(
                    base_instructions, str
                ):
                    return (*base_instructions, system_instructions)
                return (base_instructions, system_instructions)
            return system_instructions

        return base_instructions

    def _prepare_run_arguments(
        self,
        signature: BaseModel,
        *,
        candidate: dict[str, str] | None,
        message_history: Sequence[_messages.ModelMessage] | None,
        user_prompt: UserPromptInput | None,
    ) -> tuple[UserPromptInput | None, Instructions[AgentDepsT] | None]:
        """Prepare the user prompt and instructions override for a run."""
        self._require_input_instance(signature)
        if user_prompt is not None and message_history is None:
            raise ValueError(
                "user_prompt can only be provided when message_history is set"
            )

        if user_prompt is not None:
            run_user_prompt = user_prompt
        else:
            run_user_prompt = self._prepare_user_content(signature)

        system_instructions = self._prepare_system_instructions(signature, candidate)
        base_instructions = (
            candidate["instructions"]
            if candidate and "instructions" in candidate
            else getattr(self.wrapped, "_instructions", None)
        )
        instructions_override = self._compose_instructions_override(
            base_instructions, system_instructions
        )
        return run_user_prompt, instructions_override

    @staticmethod
    def _normalize_user_prompt(
        user_prompt: UserPromptInput | None,
    ) -> str | Sequence[_messages.UserContent] | None:
        """Convert user prompt input into the shape expected by wrapped agents."""
        if user_prompt is None:
            return None
        if isinstance(user_prompt, str):
            return user_prompt
        if isinstance(user_prompt, Sequence):
            return user_prompt
        return (user_prompt,)

    @overload
    async def run_signature(
        self,
        signature: BaseModel,
        *,
        output_type: None = None,
        candidate: dict[str, str] | None = None,
        user_prompt: UserPromptInput | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run_signature(
        self,
        signature: BaseModel,
        *,
        output_type: OutputSpec[RunOutputDataT] | type[RunOutputDataT],
        candidate: dict[str, str] | None = None,
        user_prompt: UserPromptInput | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run_signature(
        self,
        signature: BaseModel,
        *,
        output_type: OutputSpec[RunOutputDataT] | type[RunOutputDataT] | None = None,
        candidate: dict[str, str] | None = None,
        user_prompt: UserPromptInput | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Run the agent with a signature-based prompt.

        Args:
            signature: The structured input instance containing the input data.
            output_type: Custom output type to use for this run; defaults to the configured signature output type.
            candidate: Optional GEPA candidate with optimized text for components.
            user_prompt: Explicit user prompt to send for follow-ups; requires message_history.
            message_history: History of the conversation so far; if provided, assumes the signature input is already represented in the history unless user_prompt is supplied.
            deferred_tool_results: Optional results for deferred tool calls.
            model: Optional model to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with.
            infer_name: Whether to try to infer the agent name from the call frame.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional event stream handler to use for this run.

        Returns:
            The result of the run.

        Raises:
            ValueError: If user_prompt is provided without message_history.
        """
        # Prepare user content and system instructions from signature
        prepared_user_prompt, instructions_override = self._prepare_run_arguments(
            signature,
            candidate=candidate,
            message_history=message_history,
            user_prompt=user_prompt,
        )
        normalized_user_prompt = self._normalize_user_prompt(prepared_user_prompt)

        effective_output_type = (
            output_type if output_type is not None else self._default_output_type
        )
        if effective_output_type is None:
            raise TypeError(
                "SignatureAgent requires an output_type to execute. "
                "Ensure the wrapped agent has an output_type or pass one to "
                "run_signature(..., output_type=...)."
            )

        if instructions_override is None:
            return await self.wrapped.run(
                user_prompt=normalized_user_prompt,
                output_type=effective_output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                **_deprecated_kwargs,
            )

        with self.wrapped.override(instructions=instructions_override):
            return await self.wrapped.run(
                user_prompt=normalized_user_prompt,
                output_type=effective_output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                **_deprecated_kwargs,
            )

    @overload
    def run_signature_sync(
        self,
        signature: BaseModel,
        *,
        output_type: None = None,
        candidate: dict[str, str] | None = None,
        user_prompt: UserPromptInput | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    def run_signature_sync(
        self,
        signature: BaseModel,
        *,
        output_type: OutputSpec[RunOutputDataT] | type[RunOutputDataT],
        candidate: dict[str, str] | None = None,
        user_prompt: UserPromptInput | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    def run_signature_sync(
        self,
        signature: BaseModel,
        *,
        output_type: OutputSpec[RunOutputDataT] | type[RunOutputDataT] | None = None,
        candidate: dict[str, str] | None = None,
        user_prompt: UserPromptInput | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Synchronously run the agent with a signature-based prompt.

        Args:
            signature: The structured input instance containing the input data.
            output_type: Custom output type to use for this run; defaults to the configured signature output type.
            candidate: Optional GEPA candidate with optimized text for components.
            user_prompt: Explicit user prompt to send for follow-ups; requires message_history.
            message_history: History of the conversation so far; if provided, assumes the signature input is already represented in the history unless user_prompt is supplied.
            deferred_tool_results: Optional results for deferred tool calls.
            model: Optional model to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with.
            infer_name: Whether to try to infer the agent name from the call frame.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional event stream handler to use for this run.

        Returns:
            The result of the run.

        Raises:
            ValueError: If user_prompt is provided without message_history.
        """
        # Prepare user content and system instructions from signature
        prepared_user_prompt, instructions_override = self._prepare_run_arguments(
            signature,
            candidate=candidate,
            message_history=message_history,
            user_prompt=user_prompt,
        )
        normalized_user_prompt = self._normalize_user_prompt(prepared_user_prompt)

        effective_output_type = (
            output_type if output_type is not None else self._default_output_type
        )
        if effective_output_type is None:
            raise TypeError(
                "SignatureAgent requires an output_type to execute. "
                "Ensure the wrapped agent has an output_type or pass one to "
                "run_signature_sync(..., output_type=...)."
            )

        if instructions_override is None:
            return self.wrapped.run_sync(
                user_prompt=normalized_user_prompt,
                output_type=effective_output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                **_deprecated_kwargs,
            )

        with self.wrapped.override(instructions=instructions_override):
            return self.wrapped.run_sync(
                user_prompt=normalized_user_prompt,
                output_type=effective_output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                **_deprecated_kwargs,
            )

    @overload
    def run_signature_stream(
        self,
        signature: BaseModel,
        *,
        output_type: None = None,
        candidate: dict[str, str] | None = None,
        user_prompt: UserPromptInput | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, OutputDataT]]: ...

    @overload
    def run_signature_stream(
        self,
        signature: BaseModel,
        *,
        output_type: OutputSpec[RunOutputDataT] | type[RunOutputDataT],
        candidate: dict[str, str] | None = None,
        user_prompt: UserPromptInput | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def run_signature_stream(
        self,
        signature: BaseModel,
        *,
        output_type: OutputSpec[RunOutputDataT] | type[RunOutputDataT] | None = None,
        candidate: dict[str, str] | None = None,
        user_prompt: UserPromptInput | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]:
        """Stream the agent execution with a signature-based prompt.

        Args:
            signature: The structured input instance containing the input data.
            output_type: Custom output type to use for this run; defaults to the configured signature output type.
            candidate: Optional GEPA candidate with optimized text for components.
            user_prompt: Explicit user prompt to send for follow-ups; requires message_history.
            message_history: History of the conversation so far; if provided, assumes the signature input is already represented in the history unless user_prompt is supplied.
            deferred_tool_results: Optional results for deferred tool calls.
            model: Optional model to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with.
            infer_name: Whether to try to infer the agent name from the call frame.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional event stream handler to use for this run.

        Returns:
            A context manager that yields a StreamedRunResult.

        Raises:
            ValueError: If user_prompt is provided without message_history.
        """
        # Prepare user content and system instructions from signature
        prepared_user_prompt, instructions_override = self._prepare_run_arguments(
            signature,
            candidate=candidate,
            message_history=message_history,
            user_prompt=user_prompt,
        )
        normalized_user_prompt = self._normalize_user_prompt(prepared_user_prompt)

        effective_output_type = (
            output_type if output_type is not None else self._default_output_type
        )
        if effective_output_type is None:
            raise TypeError(
                "SignatureAgent requires an output_type to execute. "
                "Ensure the wrapped agent has an output_type or pass one to "
                "run_signature_stream(..., output_type=...)."
            )

        if instructions_override is None:
            async with self.wrapped.run_stream(
                user_prompt=normalized_user_prompt,
                output_type=effective_output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                **_deprecated_kwargs,
            ) as stream:
                yield stream
            return

        with self.wrapped.override(instructions=instructions_override):
            async with self.wrapped.run_stream(
                user_prompt=normalized_user_prompt,
                output_type=effective_output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                **_deprecated_kwargs,
            ) as stream:
                yield stream
