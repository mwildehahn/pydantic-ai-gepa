#!/usr/bin/env python
"""Simple demonstration of the comprehensive caching system.

This shows how both agent runs (LLM calls) and metric evaluations are cached.
"""

import time
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_gepa.cache import CacheManager
from pydantic_ai_gepa.runner import optimize_agent_prompts
from pydantic_ai_gepa.types import DataInstWithPrompt, RolloutOutput
from pydantic_ai_gepa.reflection import ProposalOutput


class SimpleOutput(BaseModel):
    category: str = Field(description="The category")


def create_simple_dataset():
    """Create a tiny dataset for demonstration."""
    return [
        DataInstWithPrompt(
            user_prompt=UserPromptPart(content=f"Classify item {i}"),
            message_history=None,
            metadata={"expected": f"category_{i % 2}"},
            case_id=f"case-{i}",
        )
        for i in range(3)
    ]


def create_tracked_metric():
    """Create a metric that tracks calls."""
    call_log = []

    def metric(data_inst: DataInstWithPrompt, output: RolloutOutput[SimpleOutput]):
        call_time = time.time()
        call_log.append((data_inst.case_id, call_time))
        print(f"  📊 Metric called for {data_inst.case_id}")

        if output.success and output.result:
            expected = data_inst.metadata["expected"]
            score = 1.0 if output.result.category == expected else 0.0
            return (
                score,
                f"Expected {expected}, got {output.result.category if output.result else 'None'}",
            )
        return 0.0, "Failed to generate output"

    metric.call_log = call_log
    return metric


def main():
    print("=" * 60)
    print("COMPREHENSIVE CACHING DEMONSTRATION")
    print("=" * 60)
    print("\nThis demonstrates caching of both:")
    print("  1. Agent runs (LLM calls)")
    print("  2. Metric evaluations")
    print()

    # Setup
    cache_dir = Path(".test_cache_demo")
    trainset = create_simple_dataset()

    # Create a TestModel that we can track
    test_model = TestModel(
        custom_output_type=SimpleOutput, custom_output_args={"category": "category_0"}
    )

    agent = Agent(
        model=test_model,
        instructions="Classify items into categories.",
        output_type=SimpleOutput,
    )

    # For reflection
    reflection_output = ProposalOutput(
        prompt_components={"instructions": "Updated instructions"}
    )
    reflection_model = TestModel(
        custom_output_args=reflection_output.model_dump(mode="python")
    )

    # Create metrics
    metric1 = create_tracked_metric()

    # First run - everything will be computed
    print("\n🏃 FIRST RUN (no cache)")
    print("-" * 40)

    result1 = optimize_agent_prompts(
        agent=agent,
        trainset=trainset,
        metric=metric1,
        reflection_model=reflection_model,
        max_metric_calls=10,
        display_progress_bar=False,
        track_best_outputs=False,
        seed=42,
        enable_cache=True,
        cache_dir=str(cache_dir),
        cache_verbose=True,
    )

    print(f"\n✅ First run complete")
    print(f"   Metric calls: {len(metric1.call_log)}")
    print(f"   Agent model calls: {test_model.call_count()}")

    # Get cache stats
    cache = CacheManager(cache_dir=cache_dir, enabled=True)
    stats = cache.get_cache_stats()
    print(f"\n📦 Cache statistics after first run:")
    print(f"   Cached items: {stats['num_cached_results']}")
    print(f"   Cache size: {stats['total_size_mb']:.3f} MB")

    # Second run - should use cache
    print("\n🏃 SECOND RUN (with cache)")
    print("-" * 40)

    # Reset test model call count
    initial_model_calls = test_model.call_count()
    metric2 = create_tracked_metric()

    result2 = optimize_agent_prompts(
        agent=agent,
        trainset=trainset,
        metric=metric2,
        reflection_model=reflection_model,
        max_metric_calls=10,
        display_progress_bar=False,
        track_best_outputs=False,
        seed=42,  # Same seed to ensure same optimization path
        enable_cache=True,
        cache_dir=str(cache_dir),
        cache_verbose=True,
    )

    print(f"\n✅ Second run complete")
    print(f"   Metric calls: {len(metric2.call_log)}")
    print(f"   New agent model calls: {test_model.call_count() - initial_model_calls}")

    # Compare
    print("\n📊 COMPARISON")
    print("-" * 40)
    print(f"First run metric calls:  {len(metric1.call_log)}")
    print(f"Second run metric calls: {len(metric2.call_log)}")

    if len(metric1.call_log) > 0:
        reduction = (1 - len(metric2.call_log) / len(metric1.call_log)) * 100
        print(f"\n🚀 Metric call reduction: {reduction:.1f}%")

    model_calls_first = initial_model_calls
    model_calls_second = test_model.call_count() - initial_model_calls
    if model_calls_first > 0:
        model_reduction = (1 - model_calls_second / model_calls_first) * 100
        print(f"🚀 Agent model call reduction: {model_reduction:.1f}%")

    # Show what was cached
    final_stats = cache.get_cache_stats()
    print(f"\n📦 Final cache statistics:")
    print(f"   Total cached items: {final_stats['num_cached_results']}")
    print(f"   Total cache size: {final_stats['total_size_mb']:.3f} MB")

    # Clean up
    print("\n🧹 Cleaning up cache...")
    cache.clear_cache()
    print("✨ Cache cleared")

    # Try to remove the cache directory
    try:
        cache_dir.rmdir()
        print(f"📁 Removed cache directory: {cache_dir}")
    except:
        pass


if __name__ == "__main__":
    main()
