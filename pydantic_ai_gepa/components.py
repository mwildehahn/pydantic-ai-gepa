"""Map between agent prompt components and GEPA candidates."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai.agent import Agent


def extract_seed_candidate(agent: Agent[Any, Any]) -> dict[str, str]:
    """Extract the current prompts from an agent as a GEPA candidate.

    Args:
        agent: The agent to extract prompts from.

    Returns:
        A dictionary mapping component names to their text values.
        - 'instructions': The effective instructions (combining literal and functions)
        - 'system_prompt:N': Each static system prompt by index
    """
    candidate: dict[str, str] = {}

    # Extract instructions
    # Note: In v1, we extract the literal instructions only, not the dynamic ones
    # The dynamic instructions from functions will be disabled during optimization
    agent.instructions
    if hasattr(agent, '_instructions') and agent._instructions:  # type: ignore[attr-defined]
        candidate['instructions'] = agent._instructions  # type: ignore[attr-defined]
    else:
        candidate['instructions'] = ''

    # Extract static system prompts
    if hasattr(agent, '_system_prompts'):
        for i, prompt in enumerate(agent._system_prompts):  # type: ignore[attr-defined]
            candidate[f'system_prompt:{i}'] = prompt

    return candidate


@contextmanager
def apply_candidate_to_agent(agent: Agent[Any, Any], candidate: dict[str, str]) -> Iterator[None]:
    """Apply a GEPA candidate to an agent via override_prompts.

    This returns a context manager that temporarily applies the candidate
    prompts to the agent.

    Args:
        agent: The agent to apply prompts to.
        candidate: The candidate mapping component names to text.

    Returns:
        A context manager for the temporary override.
    """
    # Extract instructions from candidate
    instructions = candidate.get('instructions', None)

    # Extract system prompts from candidate
    system_prompts: list[str] = []
    i = 0
    while f'system_prompt:{i}' in candidate:
        system_prompts.append(candidate[f'system_prompt:{i}'])
        i += 1

    # Apply via override_prompts
    # Only pass non-empty values
    kwargs: dict[str, Any] = {}
    if instructions is not None:
        kwargs['instructions'] = instructions
    if system_prompts:
        kwargs['system_prompts'] = system_prompts

    if kwargs:
        with agent.override_prompts(**kwargs):
            yield
    else:
        # No overrides needed
        yield


def get_component_names(agent: Agent[Any, Any]) -> list[str]:
    """Get the list of optimizable component names for an agent.

    Args:
        agent: The agent to inspect.

    Returns:
        List of component names that can be optimized.
    """
    components: list[str] = []

    # Instructions are always optimizable (even if empty)
    components.append('instructions')

    # Add system prompts
    if hasattr(agent, '_system_prompts'):
        for i in range(len(agent._system_prompts)):  # type: ignore[attr-defined]
            components.append(f'system_prompt:{i}')

    return components


def validate_components(agent: Agent[Any, Any], components: Sequence[str]) -> list[str]:
    """Validate that the requested components exist in the agent.

    Args:
        agent: The agent to check against.
        components: The requested component names.

    Returns:
        The validated list of component names.

    Raises:
        ValueError: If any component doesn't exist in the agent.
    """
    available = set(get_component_names(agent))
    requested = set(components)

    invalid = requested - available
    if invalid:
        raise ValueError(f'Components {invalid} not found in agent. Available components: {sorted(available)}')

    return list(components)
