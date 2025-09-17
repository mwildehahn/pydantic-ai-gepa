"""Internal caching system for GEPA optimization to support resumable runs."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import cloudpickle

from .types import (
    DataInst,
    DataInstWithPrompt,
    DataInstWithSignature,
    RolloutOutput,
    Trajectory,
)

logger = logging.getLogger(__name__)

# Type variable for the DataInst type
DataInstT = TypeVar("DataInstT", bound=DataInst)


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
    ):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files. If None, uses '.gepa_cache'
                      in the current working directory.
            enabled: Whether caching is enabled.
            verbose: Whether to log cache hits and misses.
        """
        self.enabled = enabled
        self.verbose = verbose

        if cache_dir is None:
            cache_dir = Path.cwd() / ".gepa_cache"

        self.cache_dir = Path(cache_dir)

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                logger.info(f"Cache enabled at: {self.cache_dir}")

    def _serialize_for_key(self, obj: Any) -> str:
        """Convert an object to a stable string representation for cache key generation.

        This handles various types including dataclasses, dicts, lists, and primitives.
        """
        if obj is None:
            return "None"
        elif isinstance(obj, (str, int, float, bool)):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return f"[{','.join(self._serialize_for_key(item) for item in obj)}]"
        elif isinstance(obj, dict):
            # Sort dict keys for stable serialization
            sorted_items = sorted(obj.items())
            return f"{{{','.join(f'{self._serialize_for_key(k)}:{self._serialize_for_key(v)}' for k, v in sorted_items)}}}"
        elif is_dataclass(obj):
            # Convert dataclass to dict and serialize
            # Handle dataclass instances
            if not isinstance(obj, type):
                from dataclasses import asdict

                return self._serialize_for_key(asdict(obj))
            else:
                # If it's a dataclass type (not instance), use its name
                return f"DataclassType:{obj.__name__}"
        elif hasattr(obj, "__dict__"):
            # For other objects, try to use their __dict__
            return self._serialize_for_key(obj.__dict__)
        else:
            # Fallback to string representation
            return str(obj)

    def _generate_cache_key(
        self,
        data_inst: DataInst,
        output: RolloutOutput[Any] | None,
        candidate: dict[str, str],
        key_type: str = "metric",
    ) -> str:
        """Generate a unique cache key.

        The key is based on:
        - The key type ("metric" or "agent_run")
        - The data instance (prompt/signature, metadata, case_id)
        - The output from the agent run (if provided, for metric caching)
        - The candidate prompts being evaluated
        """
        key_parts = [f"type:{key_type}"]

        # Add data instance information
        if isinstance(data_inst, DataInstWithPrompt):
            key_parts.append(f"prompt:{self._serialize_for_key(data_inst.user_prompt)}")
        elif isinstance(data_inst, DataInstWithSignature):
            key_parts.append(
                f"signature:{self._serialize_for_key(data_inst.signature)}"
            )

        # Add metadata and case_id
        key_parts.append(f"metadata:{self._serialize_for_key(data_inst.metadata)}")
        key_parts.append(f"case_id:{data_inst.case_id}")

        # Add message history if present
        if data_inst.message_history:
            key_parts.append(
                f"history:{self._serialize_for_key(data_inst.message_history)}"
            )

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

    def get_cached_metric_result(
        self,
        data_inst: DataInst,
        output: RolloutOutput[Any],
        candidate: dict[str, str],
    ) -> tuple[float, str | None] | None:
        """Get cached metric result if available.

        Args:
            data_inst: The data instance being evaluated.
            output: The output from the agent run.
            candidate: The candidate prompts being evaluated.

        Returns:
            Cached (score, feedback) tuple if found, None otherwise.
        """
        if not self.enabled:
            return None

        cache_key = self._generate_cache_key(data_inst, output, candidate, "metric")
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_result = cloudpickle.load(f)

                if self.verbose:
                    logger.info(
                        f"Cache hit for case {data_inst.case_id}: score={cached_result[0]}"
                    )

                return cached_result
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
                return None

        if self.verbose:
            logger.debug(f"Cache miss for case {data_inst.case_id}")

        return None

    def cache_metric_result(
        self,
        data_inst: DataInst,
        output: RolloutOutput[Any],
        candidate: dict[str, str],
        score: float,
        feedback: str | None,
    ) -> None:
        """Cache a metric evaluation result.

        Args:
            data_inst: The data instance that was evaluated.
            output: The output from the agent run.
            candidate: The candidate prompts that were evaluated.
            score: The computed score.
            feedback: Optional feedback string.
        """
        if not self.enabled:
            return

        cache_key = self._generate_cache_key(data_inst, output, candidate, "metric")
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                cloudpickle.dump((score, feedback), f)

            if self.verbose:
                logger.debug(
                    f"Cached result for case {data_inst.case_id}: score={score}"
                )
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    def clear_cache(self) -> None:
        """Clear all cached results."""
        if not self.enabled:
            return

        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")

            if self.verbose:
                logger.info("Cache cleared")

    def get_cached_agent_run(
        self,
        data_inst: DataInst,
        candidate: dict[str, str],
        capture_traces: bool,
    ) -> tuple[Trajectory | None, RolloutOutput[Any]] | None:
        """Get cached agent run result if available.

        Args:
            data_inst: The data instance being evaluated.
            candidate: The candidate prompts being evaluated.
            capture_traces: Whether traces were captured.

        Returns:
            Cached (trajectory, output) tuple if found, None otherwise.
        """
        if not self.enabled:
            return None

        cache_key = self._generate_cache_key(data_inst, None, candidate, "agent_run")
        # Add capture_traces to the key to differentiate
        cache_key = f"{cache_key}_traces_{capture_traces}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_result = cloudpickle.load(f)

                if self.verbose:
                    logger.info(f"Cache hit for agent run on case {data_inst.case_id}")

                return cached_result
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
                return None

        if self.verbose:
            logger.debug(f"Cache miss for agent run on case {data_inst.case_id}")

        return None

    def cache_agent_run(
        self,
        data_inst: DataInst,
        candidate: dict[str, str],
        trajectory: Trajectory | None,
        output: RolloutOutput[Any],
        capture_traces: bool,
    ) -> None:
        """Cache an agent run result.

        Args:
            data_inst: The data instance that was evaluated.
            candidate: The candidate prompts that were evaluated.
            trajectory: The execution trajectory (if captured).
            output: The output from the agent run.
            capture_traces: Whether traces were captured.
        """
        if not self.enabled:
            return

        cache_key = self._generate_cache_key(data_inst, None, candidate, "agent_run")
        # Add capture_traces to the key to differentiate
        cache_key = f"{cache_key}_traces_{capture_traces}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                cloudpickle.dump((trajectory, output), f)

            if self.verbose:
                logger.debug(f"Cached agent run for case {data_inst.case_id}")
        except Exception as e:
            logger.warning(f"Failed to cache agent run: {e}")

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
    metric: Callable[[DataInstT, RolloutOutput[Any]], tuple[float, str | None]],
    cache_manager: CacheManager,
    candidate: dict[str, str],
) -> Callable[[DataInstT, RolloutOutput[Any]], tuple[float, str | None]]:
    """Create a cached version of a metric function.

    This wrapper function checks the cache before calling the actual metric,
    and caches the result afterward.

    Args:
        metric: The original metric function.
        cache_manager: The cache manager to use.
        candidate: The current candidate being evaluated.

    Returns:
        A wrapped metric function that uses caching.
    """

    def cached_metric(
        data_inst: DataInstT,
        output: RolloutOutput[Any],
    ) -> tuple[float, str | None]:
        # Check cache first
        cached_result = cache_manager.get_cached_metric_result(
            data_inst, output, candidate
        )

        if cached_result is not None:
            return cached_result

        # Call the actual metric
        score, feedback = metric(data_inst, output)

        # Cache the result
        cache_manager.cache_metric_result(data_inst, output, candidate, score, feedback)

        return score, feedback

    return cached_metric
