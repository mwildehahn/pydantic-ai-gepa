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
                # Check if this is a Pydantic model or list of models
                model_type = self._get_model_type(field_value)
                if model_type:
                    # Add schema description before the XML data
                    schema_desc = self._format_model_schema(model_type)
                    if schema_desc:
                        content_parts.append('')  # Add spacing
                        content_parts.append(schema_desc)
                        content_parts.append('')  # Add spacing

                # Use XML formatting for complex structures
                # For lists of models, use the lowercase model name as item_tag
                if isinstance(field_value, list) and field_value and isinstance(field_value[0], BaseModel):
                    item_tag = field_value[0].__class__.__name__.lower()
                else:
                    item_tag = 'item' if isinstance(field_value, list) else 'value'

                formatted_value = format_as_xml(
                    field_value,
                    root_tag=field_name,
                    item_tag=item_tag,
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

    @staticmethod
    def _extract_model_schema(model_class: type[BaseModel]) -> dict[str, str]:
        """Extract field descriptions from a Pydantic model.

        Args:
            model_class: The Pydantic model class to extract schema from.

        Returns:
            A dictionary mapping field names to their descriptions.
        """
        schema: dict[str, str] = {}
        for field_name, field_info in model_class.model_fields.items():
            description = field_info.description
            if description:
                schema[field_name] = description
            else:
                # Default description if none provided
                schema[field_name] = f'The {field_name} field'
        return schema

    @staticmethod
    def _format_model_schema(model_class: type[BaseModel], indent: str = '', visited: set[type] | None = None) -> str:
        """Format a Pydantic model's schema as a readable description.

        Args:
            model_class: The Pydantic model class to format.
            indent: Indentation string for formatting.
            visited: Set of already visited model classes to avoid infinite recursion.

        Returns:
            A formatted string describing the model's fields.
        """
        if visited is None:
            visited = set()

        # Avoid infinite recursion for self-referential models
        if model_class in visited:
            return ''
        visited.add(model_class)

        schema = Signature._extract_model_schema(model_class)
        if not schema:
            return ''

        # Make the connection to XML elements explicit with lowercase tag names
        lines = [f'{indent}Each <{model_class.__name__.lower()}> element contains:']

        # Process each field and recursively handle nested models
        lines.extend(Signature._format_fields_recursive(model_class, indent, visited))

        return '\n'.join(lines)

    @staticmethod
    def _format_fields_recursive(
        model_class: type[BaseModel], indent: str = '', visited: set[type] | None = None, base_indent: str = ''
    ) -> list[str]:
        """Recursively format fields of a model, handling nested models inline.

        Args:
            model_class: The Pydantic model class to format fields for.
            indent: Current indentation level for the field list.
            visited: Set of already visited model classes to avoid infinite recursion.
            base_indent: Base indentation for nested fields (accumulates with depth).

        Returns:
            List of formatted field lines.
        """
        if visited is None:
            visited = set()

        lines: list[str] = []
        schema = Signature._extract_model_schema(model_class)

        for field_name, description in schema.items():
            lines.append(f'{indent}- <{field_name}>: {description}')

            # Check if this field is a nested Pydantic model
            field_info = model_class.model_fields.get(field_name)
            if field_info:
                # Get the actual type, handling Optional and List types
                field_type = field_info.annotation
                origin = getattr(field_type, '__origin__', None)

                nested_model_type = None
                # Handle Optional[Model] or List[Model]
                if origin is not None:
                    args = getattr(field_type, '__args__', ())
                    if args:
                        # For Optional, Union, List, etc., get the first argument
                        inner_type = args[0]
                        # Check if it's a BaseModel subclass
                        if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                            nested_model_type = inner_type
                # Handle direct Model type
                elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
                    nested_model_type = field_type

                # If we found a nested model, recursively format its fields
                if nested_model_type and nested_model_type not in visited:
                    visited_copy = visited.copy()
                    visited_copy.add(nested_model_type)
                    nested_lines = Signature._format_fields_recursive(
                        nested_model_type,
                        indent + '  ',  # Increase indentation for nested fields
                        visited_copy,
                        base_indent + '  ',
                    )
                    lines.extend(nested_lines)

        return lines

    @staticmethod
    def _get_model_type(field_value: Any) -> type[BaseModel] | None:
        """Get the Pydantic model type from a field value.

        Args:
            field_value: The field value to check.

        Returns:
            The Pydantic model class if applicable, None otherwise.
        """
        if isinstance(field_value, BaseModel):
            return field_value.__class__
        elif isinstance(field_value, list) and field_value:
            # Check if it's a list of Pydantic models
            first_item = field_value[0]  # type: ignore[no-any-return]
            if isinstance(first_item, BaseModel):
                return first_item.__class__
        return None

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
