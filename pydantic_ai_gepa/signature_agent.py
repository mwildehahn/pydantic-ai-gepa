"""Wrapper agent for signature-based prompts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, overload

from typing_extensions import Never

from pydantic_ai import messages as _messages, models, usage as _usage
from pydantic_ai.agent import AgentRunResult, EventStreamHandler, RunOutputDataT, WrapperAgent
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, DeferredToolResults
from pydantic_ai.toolsets import AbstractToolset

from .components import apply_candidate_to_agent_and_signatures
from .signature import Signature

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent


class SignatureAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    """Wrapper agent that enables signature-based prompts.

    This wrapper allows you to run agents using Signature instances instead of
    raw prompt strings. It handles conversion from Signature to prompt format
    and can apply GEPA optimizations.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.temporal import TemporalAgent
        from pydantic_ai_gepa import SignatureAgent, Signature
        from pydantic import Field

        class QuerySignature(Signature):
            '''Answer questions about geography'''

            question: str = Field(description="The geography question to answer")
            context: str = Field(description="Additional context if needed")

        # Create base agent
        agent = Agent(
            'openai:gpt-4o',
            instructions="You're an expert in geography.",
            name='geography',
        )

        # Wrap with Temporal if needed
        temporal_agent = TemporalAgent(agent)

        # Add signature support
        signature_agent = SignatureAgent(temporal_agent)

        # Run with signature
        sig = QuerySignature(
            question="What's the capital of France?",
            context="Focus on current political capital"
        )
        result = await signature_agent.run_signature(sig)
        ```
    """

    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        default_candidate: dict[str, str] | None = None,
    ):
        """Initialize the SignatureAgent wrapper.

        Args:
            wrapped: The agent to wrap (can be any AbstractAgent, including TemporalAgent).
            default_candidate: Optional default GEPA candidate to apply to all runs.
        """
        super().__init__(wrapped)
        self.default_candidate = default_candidate

    def _prepare_prompt(
        self, signature: Signature, candidate: dict[str, str] | None = None
    ) -> Sequence[_messages.UserContent]:
        """Convert a signature to a prompt string.

        Args:
            signature: The Signature instance to convert.
            candidate: Optional GEPA candidate to apply.

        Returns:
            The formatted prompt string.
        """
        # Use provided candidate or fall back to default
        effective_candidate = candidate or self.default_candidate

        # Convert signature to prompt parts
        return signature.to_user_content(candidate=effective_candidate)

    @overload
    async def run_signature(
        self,
        signature: Signature,
        *,
        output_type: None = None,
        candidate: dict[str, str] | None = None,
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
        signature: Signature,
        *,
        output_type: OutputSpec[RunOutputDataT],
        candidate: dict[str, str] | None = None,
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
        signature: Signature,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        candidate: dict[str, str] | None = None,
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
            signature: The Signature instance containing the input data.
            output_type: Custom output type to use for this run.
            candidate: Optional GEPA candidate with optimized text for components.
            message_history: History of the conversation so far.
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
        """
        # Prepare the prompt from signature
        user_prompt = self._prepare_prompt(signature, candidate)

        # Apply candidate to agent and signatures if provided
        effective_candidate = candidate or self.default_candidate
        if effective_candidate:
            with apply_candidate_to_agent_and_signatures(
                effective_candidate, agent=self.wrapped, signatures=[signature.__class__]
            ):
                return await self.wrapped.run(
                    user_prompt,
                    output_type=output_type,
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
        else:
            return await self.wrapped.run(
                user_prompt,
                output_type=output_type,
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
        signature: Signature,
        *,
        output_type: None = None,
        candidate: dict[str, str] | None = None,
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
        signature: Signature,
        *,
        output_type: OutputSpec[RunOutputDataT],
        candidate: dict[str, str] | None = None,
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
        signature: Signature,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        candidate: dict[str, str] | None = None,
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
            signature: The Signature instance containing the input data.
            output_type: Custom output type to use for this run.
            candidate: Optional GEPA candidate with optimized text for components.
            message_history: History of the conversation so far.
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
        """
        # Prepare the prompt from signature
        user_prompt = self._prepare_prompt(signature, candidate)

        # Apply candidate to agent and signatures if provided
        effective_candidate = candidate or self.default_candidate
        if effective_candidate:
            with apply_candidate_to_agent_and_signatures(
                effective_candidate, agent=self.wrapped, signatures=[signature.__class__]
            ):
                return self.wrapped.run_sync(
                    user_prompt,
                    output_type=output_type,
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
        else:
            return self.wrapped.run_sync(
                user_prompt,
                output_type=output_type,
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
        signature: Signature,
        *,
        output_type: None = None,
        candidate: dict[str, str] | None = None,
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
        signature: Signature,
        *,
        output_type: OutputSpec[RunOutputDataT],
        candidate: dict[str, str] | None = None,
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
        signature: Signature,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        candidate: dict[str, str] | None = None,
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
            signature: The Signature instance containing the input data.
            output_type: Custom output type to use for this run.
            candidate: Optional GEPA candidate with optimized text for components.
            message_history: History of the conversation so far.
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
        """
        # Prepare the prompt from signature
        user_prompt = self._prepare_prompt(signature, candidate)

        # Apply candidate to agent and signatures if provided
        effective_candidate = candidate or self.default_candidate
        if effective_candidate:
            with apply_candidate_to_agent_and_signatures(
                effective_candidate, agent=self.wrapped, signatures=[signature.__class__]
            ):
                async with self.wrapped.run_stream(
                    user_prompt,
                    output_type=output_type,
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
        else:
            async with self.wrapped.run_stream(
                user_prompt,
                output_type=output_type,
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

    @contextmanager
    def with_candidate(self, candidate: dict[str, str]) -> Iterator[SignatureAgent[AgentDepsT, OutputDataT]]:
        """Context manager to temporarily use a GEPA candidate.

        Args:
            candidate: GEPA candidate with optimized text for components.

        Yields:
            A SignatureAgent configured with the candidate.

        Example:
            ```python
            signature_agent = SignatureAgent(agent)
            candidate = {'instructions': 'Be more concise'}

            with signature_agent.with_candidate(candidate) as optimized:
                result = await optimized.run_signature(sig)
            ```
        """
        old_candidate = self.default_candidate
        self.default_candidate = candidate
        try:
            yield self
        finally:
            self.default_candidate = old_candidate
