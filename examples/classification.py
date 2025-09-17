from typing import Literal
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset

from pydantic_ai import Agent
from pydantic_ai_gepa.signature import Signature
from pydantic_ai_gepa.signature_agent import SignatureAgent
from pydantic_ai_gepa.types import DataInstWithSignature, RolloutOutput
from pydantic_ai_gepa.runner import optimize_agent_prompts


# Create a basic signature for the classification task
class ClassificationInput(Signature):
    """Classify the text into a category"""

    text: str = Field(description="The text to classify")


# Define the output schema
class ClassificationOutput(BaseModel):
    """The category of the text"""

    category: Literal["positive", "negative", "neutral"]


# Define a challenging dataset with ambiguous cases that force specific classifications
dataset = Dataset[
    ClassificationInput, ClassificationOutput
](
    cases=[
        # Extremely ambiguous cases - could genuinely be any category
        Case(
            name="ambiguous-forced-negative-1",
            inputs=ClassificationInput(text="It is what it is"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing resignation as negative
        ),
        Case(
            name="ambiguous-forced-positive-1",
            inputs=ClassificationInput(
                text="It is what it is"  # Same text but different expected output in different context
            ),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Could force acceptance as positive
        ),
        Case(
            name="ambiguous-forced-neutral-1",
            inputs=ClassificationInput(text="Things happened"),
            expected_output=ClassificationOutput(
                category="neutral"
            ),  # Forcing factual interpretation
        ),
        # Mixed signals where we arbitrarily pick one aspect
        Case(
            name="mixed-forced-negative-1",
            inputs=ClassificationInput(
                text="The food was absolutely incredible but the service was a bit slow"
            ),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing focus on negative aspect
        ),
        Case(
            name="mixed-forced-positive-1",
            inputs=ClassificationInput(
                text="The service was terrible but at least the food was edible"
            ),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing focus on silver lining
        ),
        Case(
            name="mixed-forced-negative-2",
            inputs=ClassificationInput(text="Great price, mediocre quality"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Prioritizing quality over price
        ),
        # Borderline cases between neutral and sentiment
        Case(
            name="borderline-forced-positive-1",
            inputs=ClassificationInput(text="It was fine"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing 'fine' as positive
        ),
        Case(
            name="borderline-forced-negative-1",
            inputs=ClassificationInput(text="It was okay"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing 'okay' as negative
        ),
        Case(
            name="borderline-forced-neutral-1",
            inputs=ClassificationInput(text="Not bad"),
            expected_output=ClassificationOutput(
                category="neutral"
            ),  # Could be positive or neutral
        ),
        # Tone-dependent cases
        Case(
            name="tone-forced-negative-1",
            inputs=ClassificationInput(text="Interesting choice"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing skeptical interpretation
        ),
        Case(
            name="tone-forced-positive-1",
            inputs=ClassificationInput(text="That's different"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing appreciative interpretation
        ),
        Case(
            name="tone-forced-negative-2",
            inputs=ClassificationInput(text="Sure, whatever you say"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing dismissive interpretation
        ),
        # Cultural/contextual dependency
        Case(
            name="cultural-forced-positive-1",
            inputs=ClassificationInput(text="It could be worse"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing optimistic interpretation
        ),
        Case(
            name="cultural-forced-negative-1",
            inputs=ClassificationInput(text="It could be better"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing critical interpretation
        ),
        # Subtle sarcasm that could be genuine
        Case(
            name="maybe-sarcasm-forced-negative-1",
            inputs=ClassificationInput(text="Great, just what I needed"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing sarcastic reading
        ),
        Case(
            name="maybe-sarcasm-forced-positive-1",
            inputs=ClassificationInput(text="Perfect timing"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing genuine reading
        ),
        # Conflicting emotional signals
        Case(
            name="emotional-conflict-1",
            inputs=ClassificationInput(text="I laughed, I cried, I left early"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing focus on leaving
        ),
        Case(
            name="emotional-conflict-2",
            inputs=ClassificationInput(text="Surprisingly disappointing"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Oxymoron leaning negative
        ),
        Case(
            name="emotional-conflict-3",
            inputs=ClassificationInput(text="Disappointingly good"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Oxymoron leaning positive
        ),
        # Questions that imply sentiment
        Case(
            name="question-forced-negative-1",
            inputs=ClassificationInput(text="Really? This is it?"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing disappointed reading
        ),
        Case(
            name="question-forced-positive-1",
            inputs=ClassificationInput(text="Could this get any better?"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing rhetorical positive
        ),
        # Comparative without clear baseline
        Case(
            name="comparative-ambiguous-1",
            inputs=ClassificationInput(text="Better than expected"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Could be neutral if expectations were very low
        ),
        Case(
            name="comparative-ambiguous-2",
            inputs=ClassificationInput(text="Not as good as hoped"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Could be neutral if hopes were unrealistic
        ),
        # Hedged statements
        Case(
            name="hedged-forced-positive-1",
            inputs=ClassificationInput(text="I guess it wasn't terrible"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing focus on "not terrible"
        ),
        Case(
            name="hedged-forced-negative-1",
            inputs=ClassificationInput(text="I suppose it was alright"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing lukewarm as negative
        ),
        # Minimal commitment
        Case(
            name="minimal-forced-neutral-1",
            inputs=ClassificationInput(text="It exists"),
            expected_output=ClassificationOutput(
                category="neutral"
            ),  # Pure existence statement
        ),
        Case(
            name="minimal-forced-negative-2",
            inputs=ClassificationInput(text="It's a thing"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing dismissive reading
        ),
        # Professional euphemisms
        Case(
            name="euphemism-forced-negative-1",
            inputs=ClassificationInput(
                text="It presents some opportunities for enhancement"
            ),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Corporate speak for "needs work"
        ),
        Case(
            name="euphemism-forced-positive-1",
            inputs=ClassificationInput(text="Room to grow"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing optimistic interpretation
        ),
        # Time-dependent sentiment
        Case(
            name="temporal-forced-positive-1",
            inputs=ClassificationInput(text="It'll do for now"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing pragmatic acceptance
        ),
        Case(
            name="temporal-forced-negative-1",
            inputs=ClassificationInput(text="Used to be better"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing decline interpretation
        ),
    ]
)

# Use a diverse set of challenging cases for training
# Include examples from different categories to help the model learn the nuanced steering patterns
signature_dataset = [
    DataInstWithSignature[ClassificationInput](
        signature=case.inputs,
        message_history=None,
        metadata={
            "label": case.expected_output.category
            if case.expected_output
            else "unknown"
        },
        case_id=case.name or f"case-{i}",
    )
    for i, case in enumerate(dataset.cases)
]

agent = Agent(
    model="openai:gpt-3.5-turbo",
    instructions="Classify text sentiment.",  # Intentionally simple to test optimization
    output_type=ClassificationOutput,
)
signature_agent = SignatureAgent(agent)


class EvaluationInput(Signature):
    """Categorized text"""

    text: str = Field(description="The text provided to the student model")
    error_message: str | None = Field(
        description="The error message if the student model failed to categorize the text"
    )
    category: Literal["positive", "negative", "neutral"] = Field(
        description="The student model's categorization of the text"
    )
    desired_category: Literal["positive", "negative", "neutral"] = Field(
        description="The desired category of the text. This is the category that the student model should have categorized the text into."
    )


class EvaluationOutput(BaseModel):
    """The score for how well the model categorizes the text into positive, negative, or neutral."""

    score: float = Field(
        description="The score for how well the model categorizes the text into positive, negative, or neutral. Provide a value between 0 and 1."
    )
    feedback: str = Field(
        description="Feedback on the input categorization, call out what was wrong about the categorization and could have been better. Be extremely detailed."
    )


eval_agent = Agent(
    model="openai:gpt-5",
    instructions="Provide a score between 0 and 1 for how well the model categorizes the text into positive, negative, or neutral.",
    output_type=EvaluationOutput,
)
eval_signature_agent = SignatureAgent(eval_agent)


def metric(
    data_inst: DataInstWithSignature[ClassificationInput],
    output: RolloutOutput[ClassificationOutput],
) -> tuple[float, str | None]:
    print(data_inst)
    print(output)
    if (
        output.success
        and output.result
        and output.result.category == data_inst.metadata["label"]
    ):
        print("Correct")
        return 1.0, "Correct"

    eval_signature = EvaluationInput(
        text=data_inst.signature.text,
        error_message=output.error_message,
        category=output.result.category
        if output.result
        else "neutral",  # Default to neutral if no result
        desired_category=data_inst.metadata["label"],
    )

    eval_output = eval_signature_agent.run_signature_sync(
        eval_signature,
    )
    print(eval_output)

    score = eval_output.output.score
    feedback = eval_output.output.feedback
    return score, feedback


if __name__ == "__main__":
    result = optimize_agent_prompts(
        agent=signature_agent,
        trainset=signature_dataset[:5],
        valset=signature_dataset[5:10],
        metric=metric,
        signature_class=ClassificationInput,
        reflection_model="openai:gpt-5",
        max_metric_calls=20,  # Increased significantly for highly ambiguous cases
        display_progress_bar=True,
        track_best_outputs=True,
        enable_cache=True,
        cache_dir=".gepa_cache",
        cache_verbose=True,
    )

    import ipdb

    ipdb.set_trace()
    print(result)
