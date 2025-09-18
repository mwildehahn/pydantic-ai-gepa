from typing import Any

from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model

from .signature import Signature
from .signature_agent import SignatureAgent


class ReflectionInput(Signature):
    """Analyze agent performance data and propose improved prompt components.

    Your task is to:
    1. Review the reflection dataset showing how the agent performed with current prompts
    2. Read all the assistant responses and the corresponding feedback
    3. Identify patterns in successes and failures
    4. Identify all niche and domain-specific factual information about the task and include it in
       the instruction, as a lot of it may not be available to the assistant in the future
    5. If the assistant utilized a generalizable strategy to solve the task, include that
       strategy in the instruction as well
    6. Propose specific improvements to the components listed in 'components_to_update'
    7. If useful, include few shot examples of the task to help the assistant understand the task better

    Focus on making prompts clearer, more specific, and better aligned with successful outcomes.
    Extract domain knowledge from the examples to enhance the instructions.
    """

    instructions: str | None = Field(
        default=None,
        description="The instructions that were used by the agent.",
    )
    prompt_components: dict[str, str] = Field(
        description="Current prompt components being used by the agent. These map to the instructions above."
    )
    reflection_dataset: dict[str, list[dict[str, Any]]] = Field(
        description="Performance data showing agent inputs, outputs, scores, and feedback for each component. Analyze these to understand what works and what needs improvement."
    )
    components_to_update: list[str] = Field(
        description="Specific components to optimize in this iteration. Only modify these components in your response while keeping others unchanged."
    )


class UpdatedComponent(BaseModel):
    component_name: str
    optimized_value: str


class ProposalOutput(BaseModel):
    """Optimized prompt components based on performance analysis.

    Provide improved versions of the specified components that:
    - Incorporate specific patterns and domain knowledge from successful examples
    - Address failure patterns identified in the reflection dataset
    - Maintain clarity and specificity while improving effectiveness
    """

    updated_components: list[UpdatedComponent] = Field(
        description="A list of updated prompt components. Only include the components that were specified for update."
    )


agent = Agent(output_type=ProposalOutput)
signature_agent = SignatureAgent(agent)


def propose_new_texts(
    candidate: dict[str, str],
    reflective_dataset: dict[str, list[dict[str, Any]]],
    components_to_update: list[str],
    reflection_model: Model | KnownModelName | str | None = None,
) -> dict[str, str]:
    """Analyze agent performance and propose optimized prompt components.

    This implementation uses a structured reflection agent that:
    - Analyzes performance data from the reflective dataset
    - Identifies patterns in successes and failures
    - Proposes specific improvements to the targeted components
    - Maintains full context of all components while updating specific ones

    Args:
        candidate: Full set of current prompt components
        reflective_dataset: Performance data with scores and feedback per component
        components_to_update: Specific components to optimize in this iteration
        reflection_model: The model to use for reflection analysis

    Returns:
        Complete set of prompt components with optimized versions for components_to_update
    """
    instructions = None
    for item in reflective_dataset.values():
        for record in item:
            if "instructions" in record:
                if not instructions:
                    instructions = record["instructions"]

                record.pop("instructions")

    signature = ReflectionInput(
        instructions=instructions,
        prompt_components=candidate,
        reflection_dataset=reflective_dataset,
        components_to_update=components_to_update,
    )
    result = signature_agent.run_signature_sync(signature, model=reflection_model)
    return {
        component.component_name: component.optimized_value
        for component in result.output.updated_components
    }
