"""GEPA adapter for pydantic-ai agents with single signature optimization."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import json
import logging
import os
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Literal

from pydantic import BaseModel
from pydantic_ai import capture_run_messages
from pydantic_ai._tool_types import ToolDefinition
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    VideoUrl,
)

from ..cache import CacheManager
from ..components import (
    apply_candidate_to_agent,
    extract_seed_candidate_with_input_type,
)
from ..evaluation_models import EvaluationBatch
from ..inspection import InspectionAborted
from ..logging_utils import get_structured_logger, log_structured
from ..signature import BoundInputSpec, InputSpec, build_input_spec
from ..signature_agent import SignatureAgent
from ..tool_components import get_or_create_tool_optimizer
from ..types import (
    DataInst,
    DataInstT,
    DataInstWithPrompt,
    MetricResult,
    RolloutOutput,
    Trajectory,
)
from ..adapter import Adapter, ReflectiveDataset, SharedReflectiveDataset


if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent

try:  # pragma: no cover - optional dependency guard
    from pydantic_ai.exceptions import ToolRetryError as _ToolRetryError
except Exception:  # pragma: no cover
    _TOOL_ERROR_TYPES: tuple[type[BaseException], ...] = ()
else:  # pragma: no cover - import path depends on runtime env
    _TOOL_ERROR_TYPES = (_ToolRetryError,)


logger = logging.getLogger(__name__)
_structured_logger = get_structured_logger()

ErrorKind = Literal["tool", "system"]

_LIBRARY_MODULE_PREFIXES = ("pydantic_ai", "pydantic_ai_gepa", "pydantic_graph")
_LIBRARY_PATH_MARKERS = tuple(
    f"{os.sep}{name}{os.sep}" for name in ("pydantic_ai", "pydantic_ai_gepa", "pydantic_graph")
)


def _traceback_originates_from_library(tb: TracebackType | None) -> bool:
    """Return True if the deepest frame in the traceback belongs to known library code."""
    last_tb = tb
    while last_tb and last_tb.tb_next:
        last_tb = last_tb.tb_next
    if last_tb is None:
        return False
    frame = last_tb.tb_frame
    module_name = frame.f_globals.get("__name__")
    if isinstance(module_name, str) and module_name.startswith(_LIBRARY_MODULE_PREFIXES):
        return True
    filename = frame.f_code.co_filename
    if filename and any(marker in filename for marker in _LIBRARY_PATH_MARKERS):
        return True
    return False


def _classify_exception(exc: BaseException) -> ErrorKind:
    """Differentiate tool call failures from systemic/library issues."""
    if _TOOL_ERROR_TYPES and isinstance(exc, _TOOL_ERROR_TYPES):
        return "tool"
    module_name = type(exc).__module__
    if isinstance(module_name, str) and module_name.startswith(_LIBRARY_MODULE_PREFIXES):
        return "system"
    if _traceback_originates_from_library(exc.__traceback__):
        return "system"
    return "tool"


def _truncate_text(value: str, limit: int = 2000) -> str:
    if len(value) <= limit:
        return value
    trimmed = value[:limit]
    omitted = len(value) - limit
    return f"{trimmed}... [truncated {omitted} chars]"


def _compact_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _timestamp_iso(timestamp: Any) -> str | None:
    return timestamp.isoformat() if hasattr(timestamp, "isoformat") else None


def _describe_binary_content(content: BinaryContent) -> str:
    size = len(content.data) if getattr(content, "data", None) else 0
    media_type = getattr(content, "media_type", "unknown")
    identifier = getattr(content, "identifier", None)
    identifier_str = f", id={identifier}" if identifier else ""
    return f"[binary media_type={media_type}{identifier_str}, bytes={size}]"


def _describe_user_content_item(item: UserContent) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, (ImageUrl, AudioUrl, DocumentUrl, VideoUrl)):
        meta: list[str] = [f"url={item.url}"]
        media_type = getattr(item, "media_type", None)
        if media_type:
            meta.append(f"media_type={media_type}")
        if getattr(item, "force_download", False):
            meta.append("force_download=true")
        return f"[{item.kind} {' '.join(meta)}]"
    if isinstance(item, BinaryContent):
        return _describe_binary_content(item)
    return repr(item)


def _serialize_user_prompt_content(content: str | Sequence[UserContent]) -> str:
    if isinstance(content, str):
        return content
    described = [_describe_user_content_item(item) for item in content]
    return "\n".join(described)


def _serialize_tool_return(
    part: ToolReturnPart | BuiltinToolReturnPart,
) -> dict[str, Any]:
    content_str = _truncate_text(part.model_response_str())
    serialized = {
        "type": "tool_return",
        "tool_name": part.tool_name,
        "content": content_str,
        "tool_call_id": part.tool_call_id,
        "timestamp": _timestamp_iso(part.timestamp),
    }
    metadata = getattr(part, "metadata", None)
    if metadata is not None:
        serialized["metadata"] = repr(metadata)
    provider_name = getattr(part, "provider_name", None)
    if provider_name:
        serialized["provider_name"] = provider_name
    return _compact_dict(serialized)


def _serialize_retry_part(part: RetryPromptPart) -> dict[str, Any]:
    if isinstance(part.content, str):
        reason = part.content
    else:
        reason = json.dumps(part.content, ensure_ascii=False, default=str)
    return _compact_dict(
        {
            "type": "retry_prompt",
            "tool_name": part.tool_name,
            "tool_call_id": part.tool_call_id,
            "content": _truncate_text(reason),
            "timestamp": _timestamp_iso(part.timestamp),
        }
    )


def _serialize_request_part(part: Any) -> dict[str, Any]:
    """Serialize a ModelRequest part."""
    if isinstance(part, SystemPromptPart):
        return _compact_dict(
            {
                "type": "system_prompt",
                "role": "system",
                "content": _truncate_text(part.content),
                "timestamp": _timestamp_iso(part.timestamp),
            }
        )
    if isinstance(part, UserPromptPart):
        return _compact_dict(
            {
                "type": "user_prompt",
                "role": "user",
                "content": _truncate_text(_serialize_user_prompt_content(part.content)),
                "timestamp": _timestamp_iso(part.timestamp),
            }
        )
    if isinstance(part, (ToolReturnPart, BuiltinToolReturnPart)):
        return _serialize_tool_return(part)
    if isinstance(part, RetryPromptPart):
        return _serialize_retry_part(part)
    return {
        "type": getattr(part, "part_kind", type(part).__name__),
        "repr": repr(part),
    }


def _serialize_response_part(part: Any) -> dict[str, Any]:
    """Serialize a ModelResponse part."""
    if isinstance(part, TextPart):
        return _compact_dict(
            {
                "type": "text",
                "role": "assistant",
                "content": _truncate_text(part.content),
                "id": part.id,
            }
        )
    if isinstance(part, ThinkingPart):
        return _compact_dict(
            {
                "type": "thinking",
                "role": "assistant",
                "content": _truncate_text(part.content),
                "id": part.id,
                "provider_name": part.provider_name,
            }
        )
    if isinstance(part, (ToolCallPart, BuiltinToolCallPart)):
        serialized = {
            "type": "tool_call",
            "role": "assistant",
            "tool_name": part.tool_name,
            "arguments": _truncate_text(part.args_as_json_str()),
            "tool_call_id": part.tool_call_id,
            "id": part.id,
        }
        provider_name = getattr(part, "provider_name", None)
        if provider_name:
            serialized["provider_name"] = provider_name
        return _compact_dict(serialized)
    if isinstance(part, BuiltinToolReturnPart):
        return _serialize_tool_return(part)
    if isinstance(part, FilePart):
        return _compact_dict(
            {
                "type": "file",
                "role": "assistant",
                "description": _describe_binary_content(part.content),
                "id": part.id,
                "provider_name": getattr(part, "provider_name", None),
            }
        )
    if hasattr(part, "content"):
        return _compact_dict(
            {
                "type": getattr(part, "part_kind", type(part).__name__),
                "role": "assistant",
                "content": _truncate_text(str(part.content)),
            }
        )
    return _compact_dict(
        {
            "type": getattr(part, "part_kind", type(part).__name__),
            "role": "assistant",
            "repr": repr(part),
        }
    )


def _serialize_model_message(
    message: ModelMessage,
    *,
    include_instructions: bool,
) -> dict[str, Any]:
    if isinstance(message, ModelRequest):
        data = {
            "kind": "request",
            "parts": [_serialize_request_part(part) for part in message.parts],
        }
        if include_instructions and message.instructions is not None:
            data["instructions"] = message.instructions
        return _compact_dict(data)
    if isinstance(message, ModelResponse):
        base: dict[str, Any] = {
            "kind": "response",
            "model_name": message.model_name,
            "provider_name": message.provider_name,
            "finish_reason": message.finish_reason,
            "timestamp": _timestamp_iso(message.timestamp),
            "parts": [_serialize_response_part(part) for part in message.parts],
        }
        if message.provider_response_id:
            base["provider_response_id"] = message.provider_response_id
        if message.provider_details:
            base["provider_details"] = message.provider_details
        if message.usage and hasattr(message.usage, "__dataclass_fields__"):
            base["usage"] = asdict(message.usage)
        return _compact_dict(base)
    return _compact_dict(
        {
            "kind": getattr(message, "kind", type(message).__name__),
            "repr": repr(message),
        }
    )


@dataclass
class AgentAdapterTrajectory(Trajectory):
    """Execution trajectory capturing the agent run for reflection."""

    messages: list[ModelMessage]
    final_output: Any
    instructions: str | None = None
    function_tools: list[ToolDefinition] | None = None
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    data_inst: DataInst | None = None
    metric_feedback: str | None = None

    def _extract_user_content(self, part: UserPromptPart) -> str:
        if isinstance(part.content, str):
            return part.content
        if part.content:
            for content_item in part.content:
                if isinstance(content_item, str):
                    return content_item
            return "Multi-modal content"
        return "No content"

    def _extract_user_message(self) -> str | None:
        for msg in self.messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        return self._extract_user_content(part)
        return None

    def _extract_assistant_message(self) -> str | None:
        for msg in reversed(self.messages):
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        return part.content
        return None

    def _serialize_messages_with_instructions(self) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        instructions_recorded = False
        for message in self.messages:
            include_instructions = (
                isinstance(message, ModelRequest)
                and not instructions_recorded
                and getattr(message, "instructions", None) is not None
            )
            if include_instructions:
                instructions_recorded = True
            serialized.append(
                _serialize_model_message(
                    message,
                    include_instructions=include_instructions,
                )
            )
        return serialized

    def _serialize_tools(self) -> list[dict[str, Any]] | None:
        """Serialize function tools to JSON schema format."""
        if not self.function_tools:
            return None

        serialized_tools = []
        for tool in self.function_tools:
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "parameters": tool.parameters_json_schema,
                },
            }
            if tool.description:
                tool_dict["function"]["description"] = tool.description
            serialized_tools.append(tool_dict)
        return serialized_tools

    def to_reflective_record(self) -> dict[str, Any]:
        user_msg = self._extract_user_message()
        assistant_msg = self._extract_assistant_message()

        if assistant_msg:
            response = assistant_msg
        elif self.final_output is not None:
            if isinstance(self.final_output, BaseModel):
                response = self.final_output.model_dump_json()
            else:
                response = str(self.final_output)
        else:
            response = "No output"

        record = {
            "user_prompt": user_msg or "No user message",
            "assistant_response": response,
            "error": self.error,
            "messages": self._serialize_messages_with_instructions(),
            "run_usage": self.usage or None,
        }

        # Add tools if available
        tools = self._serialize_tools()
        if tools:
            record["tools"] = tools

        return record


class AgentAdapter(Adapter[DataInstT], Generic[DataInstT]):
    """GEPA adapter for optimizing a single pydantic-ai agent with an optional input_type.

    This adapter connects pydantic-ai agents to the GEPA optimization engine,
    enabling prompt optimization through evaluation and reflection. It focuses on
    optimizing a single agent's instructions, optionally with a single structured
    input model class for formatting.
    """

    def __init__(
        self,
        agent: AbstractAgent[Any, Any],
        metric: Callable[[DataInstT, RolloutOutput[Any]], MetricResult],
        *,
        input_type: InputSpec[BaseModel] | None = None,
        cache_manager: CacheManager | None = None,
        optimize_tools: bool = False,
    ):
        """Initialize the adapter.

        Args:
            agent: The pydantic-ai agent to optimize.
            metric: A function that computes (score, feedback) for a data instance
                   and its output. Higher scores are better. The feedback string
                   (second element) is optional but recommended for better optimization.
            input_type: Optional structured input specification whose instructions and field
                            descriptions will be optimized alongside the agent's prompts.
            cache_manager: The cache manager to use for caching.
            optimize_tools: If True, install tool optimization support so plain agents
                expose tool description/parameter components.
        """
        self.agent = agent
        self.metric = metric
        self.input_spec: BoundInputSpec[BaseModel] | None = (
            build_input_spec(input_type) if input_type else None
        )
        self.cache_manager = cache_manager
        self.optimize_tools = optimize_tools
        if optimize_tools:
            self._configure_tool_optimizer()

    def _configure_tool_optimizer(self) -> None:
        """Install tool optimization support for plain agents when requested."""
        try:
            get_or_create_tool_optimizer(self.agent)
        except Exception:
            logger.debug(
                "Tool optimization not available for agent %s",
                getattr(self.agent, "name", self.agent.__class__.__name__),
                exc_info=True,
            )

    async def evaluate(
        self,
        batch: Sequence[DataInstT],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate the candidate on a batch of data instances.

        Args:
            batch: List of data instances to evaluate.
            candidate: Candidate mapping component names to text.
            capture_traces: Whether to capture trajectories for reflection.

        Returns:
            EvaluationBatch with outputs, scores, and optionally trajectories.
        """
        outputs: list[RolloutOutput[Any]] = []
        scores: list[float] = []
        trajectories: list[AgentAdapterTrajectory] | None = (
            [] if capture_traces else None
        )

        with self._apply_candidate(candidate):
            results = await asyncio.gather(
                *(
                    self.process_data_instance(
                        data_inst,
                        capture_traces,
                        candidate,
                    )
                    for data_inst in batch
                )
            )

        for result in results:
            outputs.append(result["output"])
            scores.append(result["score"])

            if trajectories is not None and "trajectory" in result:
                trajectories.append(result["trajectory"])

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def _apply_candidate(self, candidate: dict[str, str]):
        """Context manager to apply candidate to both agent and signature.

        Args:
            candidate: The candidate mapping component names to text.

        Returns:
            Context manager that applies the candidate.
        """
        stack = ExitStack()

        # Apply to agent
        stack.enter_context(apply_candidate_to_agent(self.agent, candidate))

        # Apply to input if provided
        if self.input_spec:
            stack.enter_context(self.input_spec.apply_candidate(candidate))

        # TODO: look at applying to output_spec too (optimizing output docs)
        return stack

    async def process_data_instance(
        self,
        data_inst: DataInstT,
        capture_traces: bool = False,
        candidate: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Process a single data instance and return results.

        Args:
            data_inst: The data instance to process.
            capture_traces: Whether to capture trajectory information.

        Returns:
            Dictionary containing 'output', 'score', and optionally 'trajectory'.
        """
        try:
            # Check cache first for agent run (if we have a current candidate)
            metric_result: MetricResult | None = None

            if self.cache_manager and candidate:
                cached_agent_result = self.cache_manager.get_cached_agent_run(
                    data_inst,
                    candidate,
                    capture_traces,
                )

                if cached_agent_result is not None:
                    trajectory, output = cached_agent_result
                else:
                    if capture_traces:
                        trajectory, output = await self._run_with_trace(
                            data_inst, candidate
                        )
                    else:
                        output = await self._run_simple(data_inst, candidate)
                        trajectory = None

                    self.cache_manager.cache_agent_run(
                        data_inst,
                        candidate,
                        trajectory,
                        output,
                        capture_traces,
                    )
            else:
                if capture_traces:
                    trajectory, output = await self._run_with_trace(
                        data_inst, candidate
                    )
                else:
                    output = await self._run_simple(data_inst, candidate)
                    trajectory = None

            # Compute score using the metric and capture optional feedback
            # Use caching if available and we have a current candidate
            if self.cache_manager and candidate:
                # Check cache first
                cached_result = self.cache_manager.get_cached_metric_result(
                    data_inst,
                    output,
                    candidate,
                )

                if cached_result is not None:
                    metric_result = cached_result
                else:
                    # Call metric and cache result
                    metric_result = self.metric(data_inst, output)
                    self.cache_manager.cache_metric_result(
                        data_inst,
                        output,
                        candidate,
                        metric_result,
                    )
            else:
                # No caching, call metric directly
                metric_result = self.metric(data_inst, output)

            # Attach metric-provided feedback to the trajectory if captured
            if trajectory is not None and metric_result is not None:
                trajectory.metric_feedback = metric_result.feedback

            assert metric_result is not None
            score = metric_result.score
            metric_feedback = metric_result.feedback

            result: dict[str, Any] = {
                "output": output,
                "score": score,
                "feedback": metric_feedback,
            }

            if trajectory is not None:
                result["trajectory"] = trajectory

            return result

        except InspectionAborted:
            raise
        except Exception as e:
            error_kind = _classify_exception(e)
            case_id = getattr(data_inst, "case_id", "unknown")
            log_structured(
                _structured_logger,
                "error",
                "AgentAdapter failed to process data instance",
                case_id=case_id,
                error_kind=error_kind,
                capture_traces=capture_traces,
                candidate_keys=sorted(candidate.keys()) if candidate else None,
            )
            logger.exception("Failed to process data instance %s", case_id)
            output = RolloutOutput.from_error(e, kind=error_kind)
            trajectory = (
                AgentAdapterTrajectory(
                    messages=[],
                    instructions=None,
                    function_tools=None,
                    final_output=None,
                    error=str(e),
                    usage={},
                    data_inst=data_inst,
                )
                if capture_traces and error_kind == "tool"
                else None
            )

            error_result: dict[str, Any] = {
                "output": output,
                "score": 0.0,  # Failed execution gets score 0
            }
            if trajectory is not None:
                error_result["trajectory"] = trajectory

            return error_result

    async def _run_with_trace(
        self, instance: DataInstT, candidate: dict[str, str] | None
    ) -> tuple[AgentAdapterTrajectory | None, RolloutOutput[Any]]:
        """Run the agent and capture the trajectory.

        Args:
            instance: The data instance to run.

        Returns:
            Tuple of (trajectory, output).
        """
        messages: list[ModelMessage] = []
        captured_messages: list[ModelMessage] = []
        run_result: Any | None = None

        try:
            with capture_run_messages() as run_messages:
                captured_messages = run_messages
                if isinstance(instance, DataInstWithPrompt):
                    run_result = await self.agent.run(
                        instance.user_prompt.content,
                        message_history=instance.message_history,
                    )
                else:
                    assert isinstance(self.agent, SignatureAgent)
                    run_result = await self.agent.run_signature(
                        instance.input,
                        message_history=instance.message_history,
                        candidate=candidate,
                    )

                messages = run_result.new_messages()
        except InspectionAborted:
            raise
        except Exception as e:
            error_kind = _classify_exception(e)
            case_id = getattr(instance, "case_id", "unknown")
            log_structured(
                _structured_logger,
                "error",
                "AgentAdapter run_with_trace failed",
                case_id=case_id,
                error_kind=error_kind,
                candidate_keys=sorted(candidate.keys()) if candidate else None,
            )
            logger.exception(
                "Failed to run agent with traces for instance %s",
                case_id,
            )
            all_messages = list(messages or captured_messages)
            trajectory = (
                AgentAdapterTrajectory(
                    messages=all_messages,
                    instructions=None,
                    function_tools=None,
                    final_output=None,
                    error=str(e),
                    # TODO: handle usage properly
                    usage={},
                    data_inst=instance,
                )
                if error_kind == "tool"
                else None
            )
            output = RolloutOutput.from_error(e, kind=error_kind)
            return trajectory, output

        assert run_result is not None
        final_messages = list(messages or captured_messages)
        final_output = run_result.output
        target_agent = self.agent
        if isinstance(target_agent, WrapperAgent):
            target_agent = target_agent.wrapped

        instructions_text = None
        function_tools = None
        for message in final_messages:
            if not isinstance(message, ModelRequest):
                continue
            if message.instructions is None:
                continue
            instructions_text = message.instructions
            function_tools = message.function_tools
            break

        trajectory = AgentAdapterTrajectory(
            messages=final_messages,
            instructions=instructions_text,
            function_tools=function_tools,
            final_output=final_output,
            error=None,
            usage=asdict(run_result.usage()),  # Convert RunUsage to dict
            data_inst=instance,
        )
        output = RolloutOutput.from_success(final_output)

        return trajectory, output

    async def _run_simple(
        self, instance: DataInstT, candidate: dict[str, str] | None
    ) -> RolloutOutput[Any]:
        """Run the agent without capturing traces.

        Args:
            instance: The data instance to run.

        Returns:
            The rollout output.
        """
        try:
            if isinstance(instance, DataInstWithPrompt):
                result = await self.agent.run(
                    instance.user_prompt.content,
                    message_history=instance.message_history,
                )
            else:
                assert isinstance(self.agent, SignatureAgent)
                result = await self.agent.run_signature(
                    instance.input,
                    message_history=instance.message_history,
                    candidate=candidate,
                )

            return RolloutOutput.from_success(result.output)
        except InspectionAborted:
            raise
        except Exception as e:
            error_kind = _classify_exception(e)
            case_id = getattr(instance, "case_id", "unknown")
            log_structured(
                _structured_logger,
                "error",
                "AgentAdapter run_simple failed",
                case_id=case_id,
                error_kind=error_kind,
                candidate_keys=sorted(candidate.keys()) if candidate else None,
            )
            logger.exception(
                "Failed to run agent without traces for instance %s",
                case_id,
            )
            return RolloutOutput.from_error(e, kind=error_kind)

    def make_reflective_dataset(
        self,
        *,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: Sequence[str],
    ) -> ReflectiveDataset:
        """Build a reflective dataset for instruction refinement.

        Args:
            candidate: The candidate that was evaluated.
            eval_batch: The evaluation results with trajectories.
            components_to_update: Component names to update.

        Returns:
            ReflectiveDataset containing shared reflection records for all components.
        """
        if not eval_batch.trajectories:
            return SharedReflectiveDataset(records=[])

        reflection_records: list[dict[str, Any]] = []
        for trajectory, output, score in zip(
            eval_batch.trajectories,
            eval_batch.outputs,
            eval_batch.scores,
        ):
            record: dict[str, Any] = trajectory.to_reflective_record()
            record["score"] = score
            record["success"] = output.success
            if output.error_message:
                record["error_message"] = output.error_message

            if trajectory.instructions:
                record["instructions"] = trajectory.instructions

            feedback_text = trajectory.metric_feedback
            if not feedback_text:
                if score >= 0.8:
                    feedback_text = "Good response"
                elif score >= 0.5:
                    feedback_text = "Adequate response, could be improved"
                else:
                    feedback_text = f"Poor response (score: {score:.2f})"
                    if output.error_message:
                        feedback_text += f" - Error: {output.error_message}"

            record["feedback"] = feedback_text
            reflection_records.append(record)

        return SharedReflectiveDataset(records=reflection_records)

    def get_components(self) -> dict[str, str]:
        """Return the current components extracted from the agent and signature."""
        return extract_seed_candidate_with_input_type(
            agent=self.agent,
            input_type=self.input_spec,
        )
