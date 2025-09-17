"""Map between agent prompt components and GEPA candidates."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from pydantic_ai.agent.wrapper import WrapperAgent

from .signature import Signature, apply_candidate_to_signature

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent


def extract_seed_candidate(agent: AbstractAgent[Any, Any]) -> dict[str, str]:
    """Extract the current prompts from an agent as a GEPA candidate.

    Args:
        agent: The agent to extract prompts from.

    Returns:
        A dictionary mapping component names to their text values.
        - 'instructions': The effective instructions (combining literal and functions)
        - 'system_prompt:N': Each static system prompt by index
    """
    candidate: dict[str, str] = {}

    target_agent = agent
    if isinstance(agent, WrapperAgent):
        target_agent = agent.wrapped

    # Extract instructions
    # Note: In v1, we extract the literal instructions only, not the dynamic ones
    # The dynamic instructions from functions will be disabled during optimization
    if hasattr(target_agent, '_instructions') and target_agent._instructions:  # type: ignore[attr-defined]
        candidate['instructions'] = target_agent._instructions  # type: ignore[attr-defined]
    else:
        candidate['instructions'] = ''

    # Extract static system prompts
    if hasattr(target_agent, '_system_prompts'):
        for i, prompt in enumerate(target_agent._system_prompts):  # type: ignore[attr-defined]
            candidate[f'system_prompt:{i}'] = prompt

    return candidate


@contextmanager
def apply_candidate_to_agent(agent: AbstractAgent[Any, Any], candidate: dict[str, str]) -> Iterator[None]:
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


def get_component_names(agent: AbstractAgent[Any, Any]) -> list[str]:
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


def validate_components(agent: AbstractAgent[Any, Any], components: Sequence[str]) -> list[str]:
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


def extract_seed_candidate_with_signature(
    agent: AbstractAgent[Any, Any],
    signature_class: type[Signature] | None = None,
) -> dict[str, str]:
    """Extract initial prompts from an agent and optionally a signature as a GEPA candidate.

    Args:
        agent: The agent to extract prompts from.
        signature_class: Optional single Signature class to extract from.

    Returns:
        Combined dictionary of all components and their initial text.
    """
    candidate: dict[str, str] = {}

    # Extract from agent
    candidate.update(extract_seed_candidate(agent))

    # Extract from signature if provided
    if signature_class:
        # Use the signature's own extraction method to ensure consistency
        candidate.update(signature_class.get_gepa_components())

    return candidate


@contextmanager
def apply_candidate_to_agent_and_signature(
    candidate: dict[str, str],
    agent: AbstractAgent[Any, Any],
    signature_class: type[Signature] | None = None,
) -> Iterator[None]:
    """Apply a GEPA candidate to an agent and optionally a signature.

    This context manager temporarily applies the candidate to the agent
    (via override_prompts) and optionally to a signature class.

    Args:
        candidate: The candidate mapping component names to text.
        agent: The agent to apply prompts to.
        signature_class: Optional single Signature class to apply to.

    Yields:
        None while the candidate is applied.
    """
    from contextlib import ExitStack

    with ExitStack() as stack:
        # Apply to agent
        stack.enter_context(apply_candidate_to_agent(agent, candidate))

        # Apply to signature if provided
        if signature_class:
            stack.enter_context(apply_candidate_to_signature(signature_class, candidate))

        yield
