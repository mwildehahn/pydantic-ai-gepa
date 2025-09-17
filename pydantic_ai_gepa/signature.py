"""DSPy-style Signature system for pydantic-ai with GEPA optimization."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import json
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from pydantic_ai.format_prompt import format_as_xml
from pydantic_ai.messages import UserContent


class SignatureSuffix:
    """Marker for fields that should be appended as plain text without formatting.

    Use with Annotated to mark a string field as a suffix:
    ```python
    suffix: Annotated[str, SignatureSuffix] = 'Review the above thoroughly...'
    ```
    """


class Signature(BaseModel):
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

    def to_system_instructions(self, *, candidate: dict[str, str] | None = None) -> str:
        """Generate system instructions including field descriptions, schemas, and suffix.

        Args:
            candidate: Optional GEPA candidate with optimized text for components.
                      If not provided, uses the default descriptions.

        Returns:
            System instructions string to be added to the agent's instructions.
        """
        # Get the effective instructions and field descriptions
        instructions = self._get_effective_text(
            "instructions", self.__class__.__doc__ or "", candidate
        )

        # Build the system instructions
        instruction_sections: list[str] = []
        suffix_parts: list[str] = []  # Collect suffix fields to append at the end
        input_lines: list[str] = []
        schema_descriptions: list[str] = []

        # Add signature-level instructions as plain text if present
        if instructions:
            instruction_sections.append(instructions.strip())

        # Add field descriptions and schemas
        for field_name, field_info in self.__class__.model_fields.items():
            # Check if this is a SignatureSuffix field
            if self._is_suffix_field(field_info):
                # For suffix fields, get the default value from field definition
                default_suffix = (
                    field_info.default if field_info.default is not None else ""
                )
                suffix_text = self._get_effective_text(
                    field_name, str(default_suffix), candidate
                )
                if suffix_text:
                    suffix_parts.append(suffix_text)
                continue

            # Get the effective description for this field
            default_desc = field_info.description or f"The {field_name} input"
            field_desc = self._get_effective_text(
                f"{field_name}:desc", default_desc, candidate
            )

            # Add field description in inputs list
            if field_desc:
                type_name = self._get_type_name(field_info.annotation)
                input_lines.append(f"- `{field_name}` ({type_name}): {field_desc}")

            # Check if the field type is a Pydantic model or list of models
            # We want to include schema info even if the current value is None
            field_type = field_info.annotation
            model_type = self._get_model_type_from_annotation(field_type)
            if model_type:
                # Add schema description for the model type
                schema_desc = self._format_model_schema(model_type)
                if schema_desc:
                    schema_descriptions.append(schema_desc)

        if input_lines:
            instruction_sections.append("Inputs")
            instruction_sections.append("\n".join(input_lines))

        if schema_descriptions:
            instruction_sections.append("Schemas")
            instruction_sections.append("\n\n".join(schema_descriptions))

        # Add suffix parts at the end as additional guidance
        if suffix_parts:
            suffix_text = "\n".join(part.strip() for part in suffix_parts if part)
            if suffix_text:
                instruction_sections.append(suffix_text)

        # Join all sections into a single instructions string
        instruction_text = "\n\n".join(
            section.strip() for section in instruction_sections if section.strip()
        )
        return instruction_text

    def to_user_content(self) -> list[UserContent]:
        """Convert this signature instance to UserContent objects for pydantic-ai.

        This method returns only the user data values, without descriptions or instructions.
        System instructions should be retrieved via `to_system_instructions()`.

        Args:
            candidate: Optional GEPA candidate (currently unused for user content).

        Returns:
            List of UserContent objects containing just the user data.
        """
        content_sections: list[str] = []

        # Add each field's value without descriptions
        for field_name, field_info in self.__class__.model_fields.items():
            field_value = getattr(self, field_name)

            # Skip None values for optional fields
            if field_value is None:
                continue

            # Skip SignatureSuffix fields (they go in system instructions)
            if self._is_suffix_field(field_info):
                continue

            format_label, formatted_value = self._format_field_value(
                field_name, field_value
            )

            if formatted_value is None:
                continue
            label = self._format_field_label(field_name)
            content_sections.append(
                self._render_field_section(label, format_label, formatted_value)
            )

        full_prompt = "\n\n".join(
            section.strip() for section in content_sections if section.strip()
        )
        return [full_prompt] if full_prompt else []

    def _get_effective_text(
        self,
        component_key: str,
        default: str,
        candidate: dict[str, str] | None,
    ) -> str:
        """Get the effective text for a component, using candidate if available."""
        if candidate is None:
            return default

        # Build the full component name with class name to handle nested models
        class_name = self.__class__.__name__
        if component_key == "instructions":
            full_key = f"signature:{class_name}:instructions"
        else:
            # For field descriptions
            full_key = f"signature:{class_name}:{component_key}"

        return candidate.get(full_key, default)

    @staticmethod
    def _get_type_name(annotation: Any) -> str:
        """Produce a readable representation of a field annotation."""

        if annotation is None:
            return "Any"

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is None:
            if isinstance(annotation, type):
                if issubclass(annotation, BaseModel):
                    return annotation.__name__
                return annotation.__name__
            return str(annotation)

        origin_name = getattr(origin, "__name__", str(origin))
        if not args:
            return origin_name

        arg_names = ", ".join(Signature._get_type_name(arg) for arg in args)
        return f"{origin_name}[{arg_names}]"

    def _format_field_value(
        self, field_name: str, field_value: Any
    ) -> tuple[str, str | None]:
        """Format a field value for inclusion in user content."""

        # Lists of BaseModels retain the existing XML formatting for readability
        if self._is_list_of_models(field_value):
            item_class = field_value[0].__class__
            formatted = format_as_xml(
                field_value,
                root_tag=field_name,
                item_tag=item_class.__name__,
                indent="  ",
            )
            return "xml", formatted

        # Single BaseModel values get JSON for structure
        if isinstance(field_value, BaseModel):
            return (
                "json",
                json.dumps(field_value.model_dump(), indent=2, ensure_ascii=False),
            )

        # Dictionaries: default to pretty JSON for consistency
        if isinstance(field_value, dict):
            return "json", json.dumps(field_value, indent=2, ensure_ascii=False)

        # Lists: use bullet list if all elements are simple scalars
        if isinstance(field_value, list):
            if not field_value:
                return "json", "[]"

            if all(self._is_simple_scalar(item) for item in field_value):
                lines = "\n".join(f"- {item}" for item in field_value)
                return "list", lines

            if all(isinstance(item, BaseModel) for item in field_value):
                # Already handled above, but keep for safety
                item_class = field_value[0].__class__
                formatted = format_as_xml(
                    field_value,
                    root_tag=field_name,
                    item_tag=item_class.__name__,
                    indent="  ",
                )
                return "xml", formatted

            return "json", json.dumps(field_value, indent=2, ensure_ascii=False)

        # Strings get wrapped in fenced blocks if multiline
        if isinstance(field_value, str):
            if "\n" in field_value:
                return "text", field_value
            return "text", field_value

        # Numbers and booleans fall back to plain text
        if isinstance(field_value, (int, float, bool)):
            return "text", str(field_value)

        # Fallback to JSON serialization when possible
        try:
            return "json", json.dumps(field_value, indent=2, ensure_ascii=False)
        except TypeError:
            return "text", str(field_value)

    @staticmethod
    def _render_field_section(
        field_label: str, format_label: str, formatted_value: str
    ) -> str:
        """Render a field section with format metadata and fenced content when needed."""

        fence = Signature._choose_fence(format_label)

        if format_label == "text" and "\n" not in formatted_value:
            return f"{field_label}: {formatted_value}"

        suffix = (
            "" if format_label in {"text", "", "list"} else f" ({format_label.upper()})"
        )
        heading = f"{field_label}{suffix}"

        if fence:
            return (
                f"{heading}\n"
                f"{fence}{format_label if format_label not in {'text', ''} else ''}\n"
                f"{formatted_value}\n{fence}"
            )

        return f"{heading}\n{formatted_value}"

    @staticmethod
    def _choose_fence(format_label: str) -> str:
        """Choose an appropriate code fence delimiter based on the format label."""

        if format_label in {"json", "jsonl", "yaml", "xml"}:
            return "```"
        if format_label in {"text", "list"}:
            return ""
        return "```"

    @staticmethod
    def _is_list_of_models(value: Any) -> bool:
        """Check whether a value is a list of BaseModel instances."""

        return (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(item, BaseModel) for item in value)
        )

    @staticmethod
    def _is_simple_scalar(value: Any) -> bool:
        """Return True if the value is a simple scalar type suitable for bullet lists."""

        return isinstance(value, (str, int, float, bool)) or value is None

    @staticmethod
    def _format_field_label(field_name: str) -> str:
        """Convert a field name into a human-friendly label."""

        parts = field_name.replace("_", " ").strip().split()
        if not parts:
            return field_name
        return " ".join(part.capitalize() for part in parts)

    @staticmethod
    def _is_suffix_field(field_info: Any) -> bool:
        """Check if a field is marked with SignatureSuffix annotation.

        Args:
            field_info: The Pydantic field info object

        Returns:
            True if the field is annotated with SignatureSuffix
        """
        # Pydantic stores the metadata from Annotated types in field_info.metadata
        if hasattr(field_info, "metadata") and field_info.metadata:
            # Check if SignatureSuffix is in the metadata
            return SignatureSuffix in field_info.metadata or any(
                isinstance(m, type) and issubclass(m, SignatureSuffix)
                for m in field_info.metadata
            )

        return False

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
                schema[field_name] = f"The {field_name} field"
        return schema

    @staticmethod
    def _format_model_schema(
        model_class: type[BaseModel], indent: str = "", visited: set[type] | None = None
    ) -> str:
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
            return ""
        visited.add(model_class)

        schema = Signature._extract_model_schema(model_class)
        if not schema:
            return ""

        # Make the connection to XML elements explicit with lowercase tag names
        lines = [f"{indent}Each <{model_class.__name__}> element contains:"]

        # Process each field and recursively handle nested models
        lines.extend(Signature._format_fields_recursive(model_class, indent, visited))

        return "\n".join(lines)

    @staticmethod
    def _format_fields_recursive(
        model_class: type[BaseModel],
        indent: str = "",
        visited: set[type] | None = None,
        base_indent: str = "",
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
            lines.append(f"{indent}- <{field_name}>: {description}")

            # Check if this field is a nested Pydantic model
            field_info = model_class.model_fields.get(field_name)
            if field_info:
                # Get the actual type, handling Optional and List types
                field_type = field_info.annotation
                origin = getattr(field_type, "__origin__", None)

                nested_model_type = None
                # Handle Optional[Model] or List[Model]
                if origin is not None:
                    args = getattr(field_type, "__args__", ())
                    if args:
                        # For Optional, Union, List, etc., get the first argument
                        inner_type = args[0]
                        # Check if it's a BaseModel subclass
                        if isinstance(inner_type, type) and issubclass(
                            inner_type, BaseModel
                        ):
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
                        indent + "  ",  # Increase indentation for nested fields
                        visited_copy,
                        base_indent + "  ",
                    )
                    lines.extend(nested_lines)

        return lines

    @staticmethod
    def _get_model_type_from_annotation(field_type: Any) -> type[BaseModel] | None:
        """Get the Pydantic model type from a field annotation.

        Args:
            field_type: The field type annotation to check.

        Returns:
            The Pydantic model class if applicable, None otherwise.
        """
        if field_type is None:
            return None

        # Get the origin for generic types
        origin = get_origin(field_type)

        # Handle Optional[Model], List[Model], etc.
        if origin is not None:
            args = get_args(field_type)
            if args:
                # For Optional, Union, List, etc., check the first argument
                inner_type = args[0]
                # Check if it's a BaseModel subclass
                if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                    return inner_type
        # Handle direct Model type
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return field_type

        return None

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
        components[f"signature:{class_name}:instructions"] = cls.__doc__ or ""

        # Add field description components
        for field_name, field_info in cls.model_fields.items():
            # Use pydantic's description field
            desc = field_info.description or f"The {field_name} input"
            components[f"signature:{class_name}:{field_name}:desc"] = desc

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
        instructions_key = f"signature:{class_name}:instructions"
        if instructions_key in candidate:
            cls.__doc__ = candidate[instructions_key]

        # Update field descriptions
        for field_name, field_info in cls.model_fields.items():
            desc_key = f"signature:{class_name}:{field_name}:desc"
            if desc_key in candidate:
                # Update the pydantic description field
                field_info.description = candidate[desc_key]


@contextmanager
def apply_candidate_to_signature(
    signature_class: type[Signature], candidate: dict[str, str]
) -> Iterator[None]:
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
        original_descs[field_name] = field_info.description or ""

    try:
        # Apply the candidate
        signature_class.apply_candidate(candidate)
        yield
    finally:
        # Restore original state
        signature_class.__doc__ = original_instructions
        for field_name, field_info in signature_class.model_fields.items():
            field_info.description = original_descs.get(field_name, "")


def extract_signature_components(
    signatures: Sequence[type[Signature]],
) -> dict[str, str]:
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
