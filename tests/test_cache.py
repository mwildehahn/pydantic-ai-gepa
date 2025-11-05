"""Tests for the caching system."""

from __future__ import annotations

import tempfile

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_gepa.cache import CacheManager, create_cached_metric
from pydantic_ai_gepa.reflection import ProposalOutput, UpdatedComponent
from pydantic_ai_gepa.runner import optimize_agent_prompts
from pydantic_ai_gepa.types import DataInstWithInput, DataInstWithPrompt, RolloutOutput


def test_cache_manager_basic():
    """Test basic cache manager operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        # Create test data
        data_inst = DataInstWithPrompt(
            user_prompt=UserPromptPart(content="Test prompt"),
            message_history=None,
            metadata={"test": "data"},
            case_id="test-1",
        )

        output = RolloutOutput.from_success("Test result")
        candidate = {"instructions": "Test instructions"}

        # Initially, cache should miss
        result = cache.get_cached_metric_result(data_inst, output, candidate)
        assert result is None

        # Cache a result
        cache.cache_metric_result(data_inst, output, candidate, 0.95, "Good job")

        # Now cache should hit
        result = cache.get_cached_metric_result(data_inst, output, candidate)
        assert result is not None
        assert result == (0.95, "Good job")

        # Different candidate should miss
        different_candidate = {"instructions": "Different instructions"}
        result = cache.get_cached_metric_result(data_inst, output, different_candidate)
        assert result is None

        # Different output should miss
        different_output = RolloutOutput.from_success("Different result")
        result = cache.get_cached_metric_result(data_inst, different_output, candidate)
        assert result is None

        # Check cache stats
        stats = cache.get_cache_stats()
        assert stats["enabled"] is True
        assert stats["num_cached_results"] == 1

        # Clear cache
        cache.clear_cache()
        stats = cache.get_cache_stats()
        assert stats["num_cached_results"] == 0


def test_cache_manager_with_signature():
    """Test cache manager with signature-based data instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        class TestSignature(BaseModel):
            text: str
            value: int = 42

        # Create test data with signature
        data_inst = DataInstWithInput(
            input=TestSignature(text="Test input", value=100),
            message_history=None,
            metadata={"label": "positive"},
            case_id="sig-test-1",
        )

        output = RolloutOutput.from_success("positive")
        candidate = {
            "instructions": "Classify the text",
            "signature:TestSignature:text:desc": "Input text",
        }

        # Cache a result
        cache.cache_metric_result(data_inst, output, candidate, 1.0, "Correct")

        # Should get cache hit with same inputs
        result = cache.get_cached_metric_result(data_inst, output, candidate)
        assert result == (1.0, "Correct")

        # Different signature value should miss
        data_inst2 = DataInstWithInput(
            input=TestSignature(text="Different input", value=100),
            message_history=None,
            metadata={"label": "positive"},
            case_id="sig-test-2",
        )
        result = cache.get_cached_metric_result(data_inst2, output, candidate)
        assert result is None


def test_cache_manager_disabled():
    """Test that cache manager does nothing when disabled."""
    cache = CacheManager(cache_dir=None, enabled=False, verbose=False)

    data_inst = DataInstWithPrompt(
        user_prompt=UserPromptPart(content="Test"),
        message_history=None,
        metadata={},
        case_id="test-1",
    )
    output = RolloutOutput.from_success("Result")
    candidate = {"instructions": "Do something"}

    # Should always return None when disabled
    result = cache.get_cached_metric_result(data_inst, output, candidate)
    assert result is None

    # Caching should be no-op
    cache.cache_metric_result(data_inst, output, candidate, 0.5, "Feedback")
    result = cache.get_cached_metric_result(data_inst, output, candidate)
    assert result is None

    # Stats should show disabled
    stats = cache.get_cache_stats()
    assert stats == {"enabled": False}


def test_create_cached_metric():
    """Test the cached metric wrapper function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_manager = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        # Create a mock metric that counts calls
        call_count = 0

        def mock_metric(data_inst, output):
            nonlocal call_count
            call_count += 1
            return 0.8, f"Call {call_count}"

        # Create cached version
        candidate = {"instructions": "Test"}
        cached_metric = create_cached_metric(mock_metric, cache_manager, candidate)

        # Create test data
        data_inst = DataInstWithPrompt(
            user_prompt=UserPromptPart(content="Test"),
            message_history=None,
            metadata={},
            case_id="test-1",
        )
        output = RolloutOutput.from_success("Result")

        # First call should invoke the metric
        score, feedback = cached_metric(data_inst, output)
        assert score == 0.8
        assert feedback == "Call 1"
        assert call_count == 1

        # Second call with same inputs should use cache
        score, feedback = cached_metric(data_inst, output)
        assert score == 0.8
        assert feedback == "Call 1"  # Same feedback
        assert call_count == 1  # Metric not called again

        # Different inputs should invoke metric again
        data_inst2 = DataInstWithPrompt(
            user_prompt=UserPromptPart(content="Different"),
            message_history=None,
            metadata={},
            case_id="test-2",
        )
        score, feedback = cached_metric(data_inst2, output)
        assert score == 0.8
        assert feedback == "Call 2"
        assert call_count == 2


def test_optimize_agent_prompts_with_caching():
    """Test that optimize_agent_prompts works with caching enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple dataset
        trainset = [
            DataInstWithPrompt(
                user_prompt=UserPromptPart(content=f"Classify: {label}"),
                message_history=None,
                metadata={"label": label},
                case_id=f"case-{i}",
            )
            for i, label in enumerate(["positive", "negative", "neutral"])
        ]

        # Track metric calls
        metric_calls = []

        def metric(data_inst, output):
            metric_calls.append(data_inst.case_id)
            predicted = str(output.result).lower() if output.success else ""
            expected = data_inst.metadata.get("label", "").lower()
            score = 1.0 if predicted == expected else 0.0
            return score, f"Score: {score}"

        # Create agent
        agent = Agent(
            TestModel(custom_output_text="positive"),
            instructions="Classify text as positive, negative, or neutral.",
        )

        reflection_output = ProposalOutput(
            updated_components=[
                UpdatedComponent(
                    component_name="instructions", optimized_value="Updated"
                )
            ]
        )
        reflection_model = TestModel(
            custom_output_args=reflection_output.model_dump(mode="python")
        )

        # First run with caching enabled
        result1 = optimize_agent_prompts(
            agent=agent,
            trainset=trainset,
            metric=metric,
            reflection_model=reflection_model,
            max_metric_calls=15,
            display_progress_bar=False,
            track_best_outputs=False,
            seed=42,
            enable_cache=True,
            cache_dir=tmpdir,
            cache_verbose=False,
        )

        first_run_calls = len(metric_calls)
        assert first_run_calls > 0
        assert result1.num_metric_calls <= 15

        # Clear metric calls
        metric_calls.clear()

        # Second run should use cache for overlapping evaluations
        result2 = optimize_agent_prompts(
            agent=agent,
            trainset=trainset,
            metric=metric,
            reflection_model=reflection_model,
            max_metric_calls=15,
            display_progress_bar=False,
            track_best_outputs=False,
            seed=42,  # Same seed to get same behavior
            enable_cache=True,
            cache_dir=tmpdir,
            cache_verbose=False,
        )

        # Should have fewer metric calls due to caching
        second_run_calls = len(metric_calls)
        assert second_run_calls < first_run_calls

        # Results should be consistent
        assert result2.original_candidate == result1.original_candidate


def test_cache_handles_errors():
    """Test that cache handles errors gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        data_inst = DataInstWithPrompt(
            user_prompt=UserPromptPart(content="Test"),
            message_history=None,
            metadata={},
            case_id="test-1",
        )

        # Test with error output
        error_output = RolloutOutput.from_error(Exception("Test error"))
        candidate = {"instructions": "Test"}

        # Should be able to cache error results
        cache.cache_metric_result(
            data_inst, error_output, candidate, 0.0, "Error occurred"
        )

        # Should retrieve cached error result
        result = cache.get_cached_metric_result(data_inst, error_output, candidate)
        assert result == (0.0, "Error occurred")

        # Success output with same data should be different cache key
        success_output = RolloutOutput.from_success("Result")
        result = cache.get_cached_metric_result(data_inst, success_output, candidate)
        assert result is None


def test_cache_agent_runs():
    """Test caching of agent execution results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        # Create test data
        data_inst = DataInstWithPrompt(
            user_prompt=UserPromptPart(content="Test prompt"),
            message_history=None,
            metadata={"test": "data"},
            case_id="test-1",
        )

        output = RolloutOutput.from_success("Agent result")
        from pydantic_ai_gepa.types import Trajectory

        trajectory = Trajectory(messages=[], final_output="Agent result", error=None)
        candidate = {"instructions": "Test instructions"}

        # Initially, cache should miss
        result = cache.get_cached_agent_run(data_inst, candidate, capture_traces=True)
        assert result is None

        # Cache an agent run with traces
        cache.cache_agent_run(
            data_inst, candidate, trajectory, output, capture_traces=True
        )

        # Now cache should hit
        result = cache.get_cached_agent_run(data_inst, candidate, capture_traces=True)
        assert result is not None
        cached_trajectory, cached_output = result
        assert cached_output.result == "Agent result"
        assert cached_trajectory is not None
        assert cached_trajectory.final_output == "Agent result"

        # Different capture_traces value should miss
        result = cache.get_cached_agent_run(data_inst, candidate, capture_traces=False)
        assert result is None

        # Cache without traces
        cache.cache_agent_run(data_inst, candidate, None, output, capture_traces=False)

        # Should hit for non-trace version
        result = cache.get_cached_agent_run(data_inst, candidate, capture_traces=False)
        assert result is not None
        cached_trajectory, cached_output = result
        assert cached_trajectory is None
        assert cached_output.result == "Agent result"

        # Different candidate should miss
        different_candidate = {"instructions": "Different"}
        result = cache.get_cached_agent_run(
            data_inst, different_candidate, capture_traces=True
        )
        assert result is None


def test_cache_key_stability():
    """Test that cache keys are stable across different orderings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        data_inst = DataInstWithPrompt(
            user_prompt=UserPromptPart(content="Test"),
            message_history=None,
            metadata={"b": 2, "a": 1},  # Dict with specific order
            case_id="test-1",
        )
        output = RolloutOutput.from_success("Result")

        # Candidates with different key orders but same content
        candidate1 = {
            "instructions": "Test",
            "signature:ExampleSignature:instructions": "InputType",
        }
        candidate2 = {
            "signature:ExampleSignature:instructions": "InputType",
            "instructions": "Test",
        }

        # Cache with first candidate
        cache.cache_metric_result(data_inst, output, candidate1, 0.9, "Good")

        # Should get cache hit with reordered candidate
        result = cache.get_cached_metric_result(data_inst, output, candidate2)
        assert result == (0.9, "Good")

        # Test with reordered metadata
        data_inst2 = DataInstWithPrompt(
            user_prompt=UserPromptPart(content="Test"),
            message_history=None,
            metadata={"a": 1, "b": 2},  # Same content, different order
            case_id="test-1",
        )

        # Should still get cache hit
        result = cache.get_cached_metric_result(data_inst2, output, candidate1)
        assert result == (0.9, "Good")
