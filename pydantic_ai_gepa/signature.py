"""DSPy-style Signature system for pydantic-ai with GEPA optimization."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

from pydantic_ai.format_prompt import format_as_xml
from pydantic_ai.messages import UserContent


class SignatureMeta(type(BaseModel)):
    """Metaclass for Signature classes to handle field processing."""

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any) -> type:
        # Process the class
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        return cls

    @property
    def instructions(cls) -> str:
        """Get the instructions/docstring for this signature."""
        return cls.__doc__ or ''

    @instructions.setter
    def instructions(cls, value: str) -> None:
        """Set the instructions/docstring for this signature."""
        cls.__doc__ = value


class Signature(BaseModel, metaclass=SignatureMeta):
    """Base class for defining input signatures that can be optimized by GEPA.

    Subclass this to define your input schema:

    ```python
    class EmailAnalysis(Signature):
        '''Analyze emails for key information'''

        emails: list[Email] = Field(description="Emails to analyze")
        context: str = Field(description="Additional context")
    ```

    The class docstring and field descriptions can be optimized by GEPA.
    All fields are automatically treated as inputs.
    """

    def to_user_content(self, *, candidate: dict[str, str] | None = None) -> list[UserContent]:
        """Convert this signature instance to UserContent objects for pydantic-ai.

        Args:
            candidate: Optional GEPA candidate with optimized text for components.
                      If not provided, uses the default descriptions.

        Returns:
            List of UserContent objects to pass to an agent.
        """
        # Get the effective instructions and field descriptions
        instructions = self._get_effective_text('instructions', self.__class__.__doc__ or '', candidate)

        # Build the prompt content
        content_parts: list[str] = []

        # Add instructions as plain text if present
        if instructions:
            content_parts.append(instructions)
            content_parts.append('')  # Empty line for spacing

        # Add each field with its description and value
        for field_name, field_info in self.__class__.model_fields.items():
            field_value = getattr(self, field_name)

            # Skip None values for optional fields
            if field_value is None:
                continue

            # Get the effective description for this field
            # Use pydantic's description field if available, otherwise create a default
            default_desc = field_info.description or f'The {field_name} input'
            field_desc = self._get_effective_text(f'{field_name}:desc', default_desc, candidate)

            # Add field description if present
            if field_desc:
                content_parts.append(field_desc)

            # Decide how to format the field value
            if isinstance(field_value, (list, dict, BaseModel)) or (isinstance(field_value, BaseModel)):
                # Use XML formatting for complex structures
                formatted_value = format_as_xml(
                    field_value,
                    root_tag=field_name,
                    item_tag='item' if isinstance(field_value, list) else 'value',
                    indent='  ',
                )
                content_parts.append(formatted_value)
            else:
                # For simple values, use a clean key-value format
                field_name_formatted = field_name.replace('_', ' ').title()
                if isinstance(field_value, str) and '\n' in field_value:
                    # Multi-line strings get special formatting
                    content_parts.append(f'{field_name_formatted}:')
                    content_parts.append(field_value)
                else:
                    # Single-line values
                    content_parts.append(f'{field_name_formatted}: {field_value}')

            content_parts.append('')  # Empty line between fields

        # Join all parts into a single prompt, removing trailing empty lines
        full_prompt = '\n'.join(content_parts).rstrip()
        return [full_prompt]

    def _get_effective_text(self, component_key: str, default: str, candidate: dict[str, str] | None) -> str:
        """Get the effective text for a component, using candidate if available."""
        if candidate is None:
            return default

        # Build the full component name with class name to handle nested models
        class_name = self.__class__.__name__
        if component_key == 'instructions':
            full_key = f'signature:{class_name}:instructions'
        else:
            # For field descriptions
            full_key = f'signature:{class_name}:{component_key}'

        return candidate.get(full_key, default)

    @classmethod
    def get_gepa_components(cls) -> dict[str, str]:
        """Extract GEPA components from this signature class.

        Returns:
            Dictionary mapping component names to their default text values.
        """
        components: dict[str, str] = {}
        class_name = cls.__name__

        # Add the instructions component with class name
        components[f'signature:{class_name}:instructions'] = cls.__doc__ or ''

        # Add field description components
        for field_name, field_info in cls.model_fields.items():
            # Use pydantic's description field
            desc = field_info.description or f'The {field_name} input'
            components[f'signature:{class_name}:{field_name}:desc'] = desc

        return components

    @classmethod
    def apply_candidate(cls, candidate: dict[str, str]) -> None:
        """Apply a GEPA candidate to this signature class.

        This modifies the class in-place with the optimized text.

        Args:
            candidate: GEPA candidate with optimized text for components.
        """
        class_name = cls.__name__

        # Update instructions if present in candidate
        instructions_key = f'signature:{class_name}:instructions'
        if instructions_key in candidate:
            cls.__doc__ = candidate[instructions_key]

        # Update field descriptions
        for field_name, field_info in cls.model_fields.items():
            desc_key = f'signature:{class_name}:{field_name}:desc'
            if desc_key in candidate:
                # Update the pydantic description field
                field_info.description = candidate[desc_key]


@contextmanager
def apply_candidate_to_signature(signature_class: type[Signature], candidate: dict[str, str]) -> Iterator[None]:
    """Context manager to temporarily apply a GEPA candidate to a signature.

    Args:
        signature_class: The Signature class to modify.
        candidate: GEPA candidate with optimized text.

    Yields:
        None while the candidate is applied.
    """
    # Save original state
    original_instructions = signature_class.__doc__
    original_descs: dict[str, str] = {}

    for field_name, field_info in signature_class.model_fields.items():
        original_descs[field_name] = field_info.description or ''

    try:
        # Apply the candidate
        signature_class.apply_candidate(candidate)
        yield
    finally:
        # Restore original state
        signature_class.__doc__ = original_instructions
        for field_name, field_info in signature_class.model_fields.items():
            field_info.description = original_descs.get(field_name, '')


def extract_signature_components(signatures: Sequence[type[Signature]]) -> dict[str, str]:
    """Extract all GEPA components from multiple signature classes.

    Args:
        signatures: List of Signature classes to extract from.

    Returns:
        Combined dictionary of all components.
    """
    all_components: dict[str, str] = {}
    for sig_class in signatures:
        all_components.update(sig_class.get_gepa_components())
    return all_components
