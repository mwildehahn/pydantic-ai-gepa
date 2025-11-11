"""GEPA adapter for pydantic-ai agents with single signature optimization."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import logging
import json
from typing import TYPE_CHECKING, Any, Generic

from pydantic import BaseModel
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

from .components import apply_candidate_to_agent
from .evaluation_models import EvaluationBatch
from .inspection import InspectionAborted
from .signature import BoundInputSpec, InputSpec, build_input_spec
from .signature_agent import SignatureAgent
from .cache import CacheManager
from .types import (
    DataInst,
    DataInstT,
    DataInstWithPrompt,
    MetricResult,
    RolloutOutput,
    Trajectory,
)

logger = logging.getLogger(__name__)


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


def _serialize_tool_return(part: ToolReturnPart | BuiltinToolReturnPart) -> dict[str, Any]:
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
    return _compact_dict({
        "type": "retry_prompt",
        "tool_name": part.tool_name,
        "tool_call_id": part.tool_call_id,
        "content": _truncate_text(reason),
        "timestamp": _timestamp_iso(part.timestamp),
    })


def _serialize_request_part(part: Any) -> dict[str, Any]:
    """Serialize a ModelRequest part."""
    if isinstance(part, SystemPromptPart):
        return _compact_dict({
            "type": "system_prompt",
            "role": "system",
            "content": _truncate_text(part.content),
            "timestamp": _timestamp_iso(part.timestamp),
        })
    if isinstance(part, UserPromptPart):
        return _compact_dict({
            "type": "user_prompt",
            "role": "user",
            "content": _truncate_text(_serialize_user_prompt_content(part.content)),
            "timestamp": _timestamp_iso(part.timestamp),
        })
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
        return _compact_dict({
            "type": "text",
            "role": "assistant",
            "content": _truncate_text(part.content),
            "id": part.id,
        })
    if isinstance(part, ThinkingPart):
        return _compact_dict({
            "type": "thinking",
            "role": "assistant",
            "content": _truncate_text(part.content),
            "id": part.id,
            "provider_name": part.provider_name,
        })
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
        return _compact_dict({
            "type": "file",
            "role": "assistant",
            "description": _describe_binary_content(part.content),
            "id": part.id,
            "provider_name": getattr(part, "provider_name", None),
        })
    if hasattr(part, "content"):
        return _compact_dict({
            "type": getattr(part, "part_kind", type(part).__name__),
            "role": "assistant",
            "content": _truncate_text(str(part.content)),
        })
    return _compact_dict({
        "type": getattr(part, "part_kind", type(part).__name__),
        "role": "assistant",
        "repr": repr(part),
    })


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
    return _compact_dict({
        "kind": getattr(message, "kind", type(message).__name__),
        "repr": repr(message),
    })


@dataclass
class AdapterTrajectory(Trajectory):
    """Execution trajectory capturing the agent run for reflection."""

    messages: list[ModelMessage]
    final_output: Any
    instructions: str | None = None
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

        return {
            "user_prompt": user_msg or "No user message",
            "assistant_response": response,
            "error": self.error,
            "messages": self._serialize_messages_with_instructions(),
            "run_usage": self.usage or None,
        }


if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent


class AgentAdapter(Generic[DataInstT]):
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
        """
        self.agent = agent
        self.metric = metric
        self.input_spec: BoundInputSpec[BaseModel] | None = (
            build_input_spec(input_type) if input_type else None
        )
        self.cache_manager = cache_manager

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
        trajectories: list[AdapterTrajectory] | None = [] if capture_traces else None

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

        # Apply to signature if provided
        if self.input_spec:
            stack.enter_context(self.input_spec.apply_candidate(candidate))

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
                        trajectory, output = await self._run_with_trace(data_inst)
                    else:
                        output = await self._run_simple(data_inst)
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
                    trajectory, output = await self._run_with_trace(data_inst)
                else:
                    output = await self._run_simple(data_inst)
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
            }
            if trajectory is not None:
                result["trajectory"] = trajectory

            return result

        except InspectionAborted:
            raise
        except Exception as e:
            logger.exception(
                "Failed to process data instance %s",
                getattr(data_inst, "case_id", "unknown"),
            )
            output = RolloutOutput.from_error(e)
            trajectory = (
                AdapterTrajectory(
                    messages=[],
                    instructions=None,
                    final_output=None,
                    error=str(e),
                    usage={},
                    data_inst=data_inst,
                )
                if capture_traces
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
        self, instance: DataInstT
    ) -> tuple[AdapterTrajectory, RolloutOutput[Any]]:
        """Run the agent and capture the trajectory.

        Args:
            instance: The data instance to run.

        Returns:
            Tuple of (trajectory, output).
        """
        messages: list[ModelMessage] = []

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
                )

            messages = result.new_messages()
            final_output = result.output
            target_agent = self.agent
            if isinstance(target_agent, WrapperAgent):
                target_agent = target_agent.wrapped

            instructions_text = None
            for message in messages:
                if isinstance(message, ModelRequest):
                    instructions_text = message.instructions
                    break

            trajectory = AdapterTrajectory(
                messages=messages,
                instructions=instructions_text,
                final_output=final_output,
                error=None,
                usage=asdict(result.usage()),  # Convert RunUsage to dict
                data_inst=instance,
            )
            output = RolloutOutput.from_success(final_output)

            return trajectory, output
        except InspectionAborted:
            raise
        except Exception as e:
            logger.exception(
                "Failed to run agent with traces for instance %s",
                getattr(instance, "case_id", "unknown"),
            )
            trajectory = AdapterTrajectory(
                messages=messages,
                instructions=None,
                final_output=None,
                error=str(e),
                usage={},
                data_inst=instance,
            )
            output = RolloutOutput.from_error(e)
            return trajectory, output

    async def _run_simple(self, instance: DataInstT) -> RolloutOutput[Any]:
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
                )

            return RolloutOutput.from_success(result.output)
        except InspectionAborted:
            raise
        except Exception as e:
            logger.exception(
                "Failed to run agent without traces for instance %s",
                getattr(instance, "case_id", "unknown"),
            )
            return RolloutOutput.from_error(e)


    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build a reflective dataset for instruction refinement.

        Args:
            candidate: The candidate that was evaluated.
            eval_batch: The evaluation results with trajectories.
            components_to_update: Component names to update.

        Returns:
            Mapping from component name to list of reflection records.
        """
        if not eval_batch.trajectories:
            # No trajectories available, return empty dataset
            return {comp: [] for comp in components_to_update}

        # Build reflection records from trajectories
        reflection_records: list[dict[str, Any]] = []
        for trajectory, output, score in zip(
            eval_batch.trajectories,
            eval_batch.outputs,
            eval_batch.scores,
        ):
            record: dict[str, Any] = trajectory.to_reflective_record()

            # Add score and success information
            record["score"] = score
            record["success"] = output.success
            if output.error_message:
                record["error_message"] = output.error_message

            if trajectory.instructions:
                record["instructions"] = trajectory.instructions

            # Use metric feedback if available, otherwise use a simple fallback
            feedback_text = trajectory.metric_feedback

            if not feedback_text:
                # Simple fallback when metric doesn't provide feedback
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

        # For pydantic-ai, all components work together, so they all need
        # the same reflection data to understand the full context
        return {comp: reflection_records for comp in components_to_update}
