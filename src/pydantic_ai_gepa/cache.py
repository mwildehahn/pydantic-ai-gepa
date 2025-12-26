"""Internal caching system for GEPA optimization to support resumable runs."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import is_dataclass
from pathlib import Path
from collections.abc import Awaitable
from typing import Any, Callable, TypeVar

import cloudpickle
import logfire

from pydantic_evals import Case

from .gepa_graph.models import CandidateMap, candidate_texts
from .types import (
    MetadataWithMessageHistory,
    MetricResult,
    RolloutOutput,
    Trajectory,
)

CaseInputT = TypeVar("CaseInputT")
CaseOutputT = TypeVar("CaseOutputT")
CaseMetadataT = TypeVar("CaseMetadataT")


class CacheManager:
    """Manages caching of metric evaluation results for GEPA optimization.

    This cache allows optimization runs to be resumed without re-running
    expensive LLM calls that have already been completed.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        enabled: bool = True,
        verbose: bool = False,
        model_identifier: str | None = None,
    ):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files. If None, uses '.gepa_cache'
                      in the current working directory.
            enabled: Whether caching is enabled.
            verbose: Whether to log cache hits and misses.
            model_identifier: Optional string that scopes cache entries to a specific
                model (e.g., ``openai:gpt-4o``). When provided, cache keys include
                this identifier so different models never share cached artifacts.
        """
        self.enabled = enabled
        self.verbose = verbose
        self.model_identifier = model_identifier

        if cache_dir is None:
            cache_dir = Path.cwd() / ".gepa_cache"

        self.cache_dir = Path(cache_dir)

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                logfire.info(
                    "Cache enabled",
                    cache_dir=str(self.cache_dir),
                )

    @staticmethod
    def _serialize_for_key(obj: Any) -> str:
        """Convert an object to a stable string representation for cache key generation.

        This handles various types including dataclasses, dicts, lists, and primitives.
        """
        if obj is None:
            return "None"
        elif isinstance(obj, (str, int, float, bool)):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return (
                f"[{','.join(CacheManager._serialize_for_key(item) for item in obj)}]"
            )
        elif isinstance(obj, dict):
            # Sort dict keys for stable serialization
            sorted_items = sorted(obj.items())
            return f"{{{','.join(f'{CacheManager._serialize_for_key(k)}:{CacheManager._serialize_for_key(v)}' for k, v in sorted_items)}}}"
        # Special handling for pydantic-ai message parts to exclude timestamp
        elif type(obj).__name__ in [
            "UserPromptPart",
            "SystemPromptPart",
            "ToolResponsePart",
            "ModelRequestPart",
            "ModelResponsePart",
            "RetryPromptPart",
            "ToolReturnPart",
            "TextPart",
        ]:
            # For message parts, exclude timestamp field for stable cache keys
            obj_dict = obj.__dict__.copy() if hasattr(obj, "__dict__") else {}
            obj_dict.pop("timestamp", None)  # Remove timestamp if present
            return CacheManager._serialize_for_key(obj_dict)
        elif is_dataclass(obj):
            # Convert dataclass to dict and serialize
            # Handle dataclass instances
            if not isinstance(obj, type):
                from dataclasses import asdict

                return CacheManager._serialize_for_key(asdict(obj))
            else:
                # If it's a dataclass type (not instance), use its name
                return f"DataclassType:{obj.__name__}"
        elif hasattr(obj, "__dict__"):
            # For other objects, try to use their __dict__
            return CacheManager._serialize_for_key(obj.__dict__)
        else:
            # Fallback to string representation
            return str(obj)

    @staticmethod
    def _extract_message_history(case: Case[Any, Any, Any]) -> list[Any] | None:
        metadata = case.metadata
        if isinstance(metadata, MetadataWithMessageHistory):
            return metadata.message_history
        return None

    @staticmethod
    def _case_label(case: Case[Any, Any, Any], case_index: int | None) -> str:
        if case.name:
            return case.name
        if case_index is not None:
            return f"case-{case_index}"
        return "case-unknown"

    def _generate_cache_key(
        self,
        case: Case[Any, Any, Any],
        case_index: int | None,
        output: RolloutOutput[Any] | None,
        candidate: dict[str, str],
        key_type: str = "metric",
        model_identifier: str | None = None,
    ) -> str:
        """Generate a unique cache key.

        The key is based on:
        - The key type ("metric" or "agent_run")
        - The case inputs/metadata/name (prompt or structured signature)
        - The output from the agent run (if provided, for metric caching)
        - The candidate prompts being evaluated
        """
        key_parts = [f"type:{key_type}"]

        resolved_model_identifier = model_identifier or self.model_identifier
        if resolved_model_identifier:
            key_parts.append(
                f"model:{self._serialize_for_key(resolved_model_identifier)}"
            )

        case_name = self._case_label(case, case_index)
        key_parts.append(f"case_name:{case_name}")

        serialized_inputs = self._serialize_for_key(case.inputs)
        key_parts.append(f"inputs:{serialized_inputs}")

        serialized_metadata = self._serialize_for_key(case.metadata)
        key_parts.append(f"metadata:{serialized_metadata}")

        message_history = self._extract_message_history(case)
        if message_history:
            key_parts.append(f"history:{self._serialize_for_key(message_history)}")

        # Add output information (only for metric caching)
        if output is not None:
            key_parts.append(f"result:{self._serialize_for_key(output.result)}")
            key_parts.append(f"success:{output.success}")
            key_parts.append(f"error:{output.error_message or 'None'}")

        # Add candidate prompts (sorted for stability)
        sorted_candidate = sorted(candidate.items())
        key_parts.append(f"candidate:{self._serialize_for_key(sorted_candidate)}")

        # Create a hash of all parts
        combined = "|".join(key_parts)
        hash_obj = hashlib.sha256(combined.encode("utf-8"))
        return hash_obj.hexdigest()

    def set_model_identifier(self, model_identifier: str | None) -> None:
        """Configure the default model identifier used for cache keys."""
        self.model_identifier = model_identifier

    def get_cached_metric_result(
        self,
        case: Case[Any, Any, Any],
        case_index: int | None,
        output: RolloutOutput[Any],
        candidate: CandidateMap,
        model_identifier: str | None = None,
    ) -> MetricResult | None:
        """Get cached metric result if available.

        Args:
            case: The case being evaluated.
            case_index: Optional stable index for logging/cache keys when the case has no name.
            output: The output from the agent run.
            candidate: The candidate prompts being evaluated.
            model_identifier: Optional override for the model identifier to use when
                computing the cache key. Defaults to the manager-level identifier.

        Returns:
            Cached (score, feedback) tuple if found, None otherwise.
        """
        if not self.enabled:
            return None

        case_label = self._case_label(case, case_index)
        candidate_text = candidate_texts(candidate)
        cache_key = self._generate_cache_key(
            case,
            case_index,
            output,
            candidate_text,
            "metric",
            model_identifier=model_identifier,
        )
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_result: MetricResult = cloudpickle.load(f)

                if self.verbose:
                    logfire.info(
                        "Cache hit for metric",
                        case_label=case_label,
                        score=cached_result.score,
                    )

                return cached_result
            except Exception as e:
                logfire.warn(
                    "Failed to load metric cache file",
                    cache_file=str(cache_file),
                    exception=e,
                )
                return None

        if self.verbose:
            logfire.debug("Cache miss for metric", case_label=case_label)

        return None

    def cache_metric_result(
        self,
        case: Case[Any, Any, Any],
        case_index: int | None,
        output: RolloutOutput[Any],
        candidate: CandidateMap,
        metric_result: MetricResult,
        model_identifier: str | None = None,
    ) -> None:
        """Cache a metric evaluation result.

        Args:
            case: The case that was evaluated.
            case_index: Optional index associated with the case.
            output: The output from the agent run.
            candidate: The candidate prompts that were evaluated.
            metric_result: The computed metric result.
        """
        if not self.enabled:
            return

        candidate_text = candidate_texts(candidate)
        cache_key = self._generate_cache_key(
            case,
            case_index,
            output,
            candidate_text,
            "metric",
            model_identifier=model_identifier,
        )
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                cloudpickle.dump(metric_result, f)

            if self.verbose:
                logfire.debug(
                    "Cached metric result",
                    case_label=self._case_label(case, case_index),
                    score=metric_result.score,
                )
        except Exception as e:
            logfire.warn("Failed to cache metric result", exception=e)

    def clear_cache(self) -> None:
        """Clear all cached results."""
        if not self.enabled:
            return

        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logfire.warn(
                        "Failed to delete cache file",
                        cache_file=str(cache_file),
                        exception=e,
                    )

            if self.verbose:
                logfire.info("Cache cleared", cache_dir=str(self.cache_dir))

    def get_cached_agent_run(
        self,
        case: Case[Any, Any, Any],
        case_index: int | None,
        candidate: CandidateMap,
        capture_traces: bool,
        model_identifier: str | None = None,
    ) -> tuple[Trajectory | None, RolloutOutput[Any]] | None:
        """Get cached agent run result if available.

        Args:
            case: The case being evaluated.
            case_index: Optional index associated with the case.
            candidate: The candidate prompts being evaluated.
            capture_traces: Whether traces were captured.

        Returns:
            Cached (trajectory, output) tuple if found, None otherwise.
        """
        if not self.enabled:
            return None

        candidate_text = candidate_texts(candidate)
        cache_key = self._generate_cache_key(
            case,
            case_index,
            None,
            candidate_text,
            "agent_run",
            model_identifier=model_identifier,
        )
        # Add capture_traces to the key to differentiate
        cache_key = f"{cache_key}_traces_{capture_traces}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_result = cloudpickle.load(f)

                if self.verbose:
                    logfire.info(
                        "Cache hit for agent run",
                        case_label=self._case_label(case, case_index),
                    )

                return cached_result
            except Exception as e:
                logfire.warn(
                    "Failed to load agent run cache file",
                    cache_file=str(cache_file),
                    exception=e,
                )
                return None

        if self.verbose:
            logfire.debug(
                "Cache miss for agent run",
                case_label=self._case_label(case, case_index),
            )

        return None

    def cache_agent_run(
        self,
        case: Case[Any, Any, Any],
        case_index: int | None,
        candidate: CandidateMap,
        trajectory: Trajectory | None,
        output: RolloutOutput[Any],
        capture_traces: bool,
        model_identifier: str | None = None,
    ) -> None:
        """Cache an agent run result.

        Args:
            case: The case that was evaluated.
            case_index: Optional index associated with the case.
            candidate: The candidate prompts that were evaluated.
            trajectory: The execution trajectory (if captured).
            output: The output from the agent run.
            capture_traces: Whether traces were captured.
        """
        if not self.enabled:
            return

        candidate_text = candidate_texts(candidate)
        cache_key = self._generate_cache_key(
            case,
            case_index,
            None,
            candidate_text,
            "agent_run",
            model_identifier=model_identifier,
        )
        # Add capture_traces to the key to differentiate
        cache_key = f"{cache_key}_traces_{capture_traces}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                cloudpickle.dump((trajectory, output), f)

            if self.verbose:
                logfire.debug(
                    "Cached agent run",
                    case_label=self._case_label(case, case_index),
                )
        except Exception as e:
            logfire.warn("Failed to cache agent run", exception=e)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the cache.

        Returns:
            Dictionary with cache statistics.
        """
        if not self.enabled:
            return {"enabled": False}

        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "num_cached_results": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }


def create_cached_metric(
    metric: Callable[
        [Case[CaseInputT, CaseOutputT, CaseMetadataT], RolloutOutput[Any]], MetricResult
    ],
    cache_manager: CacheManager,
    candidate: CandidateMap,
    *,
    model_identifier: str | None = None,
) -> Callable[
    [Case[CaseInputT, CaseOutputT, CaseMetadataT], RolloutOutput[Any]],
    MetricResult | Awaitable[MetricResult],
]:
    """Create a cached version of a metric function.

    This wrapper function checks the cache before calling the actual metric,
    and caches the result afterward.

    Args:
        metric: The original metric function that accepts a Case.
        cache_manager: The cache manager to use.
        candidate: The current candidate being evaluated.
        model_identifier: Optional override for the model identifier. When not
            provided, the cache manager's configured identifier (if any) is used.

    Returns:
        A wrapped metric function that uses caching.
    """

    def cached_metric(
        case: Case[CaseInputT, CaseOutputT, CaseMetadataT],
        output: RolloutOutput[Any],
    ) -> MetricResult | Awaitable[MetricResult]:
        # Check cache first
        cached_result = cache_manager.get_cached_metric_result(
            case,
            None,
            output,
            candidate,
            model_identifier=model_identifier,
        )

        if cached_result is not None:
            return cached_result

        # Call the actual metric
        metric_result = metric(case, output)
        if inspect.isawaitable(metric_result):

            async def cache_and_return() -> MetricResult:
                awaited = await metric_result
                cache_manager.cache_metric_result(
                    case,
                    None,
                    output,
                    candidate,
                    awaited,
                    model_identifier=model_identifier,
                )
                return awaited

            return cache_and_return()

        # Cache the result
        cache_manager.cache_metric_result(
            case,
            None,
            output,
            candidate,
            metric_result,
            model_identifier=model_identifier,
        )

        return metric_result

    return cached_metric
