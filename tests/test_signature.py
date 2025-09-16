"""Tests for the Signature system."""

from __future__ import annotations

from inline_snapshot import snapshot
from pydantic import BaseModel, Field
from pydantic_ai_gepa import Signature


class Email(BaseModel):
    """An email message."""

    subject: str
    contents: str

    def __str__(self) -> str:
        return f'Subject: {self.subject}\n{self.contents}'


class EmailAnalysis(Signature):
    """Analyze emails for key information and sentiment."""

    emails: list[Email] = Field(description='List of email messages to analyze. Look for sentiment and key topics.')
    context: str = Field(description='Additional context about the email thread or conversation.')


def test_signature_basic():
    """Test basic signature functionality."""
    # Create an instance
    sig = EmailAnalysis(
        emails=[
            Email(subject='Product Issue', contents="I'm having trouble with the login feature."),
            Email(subject='Re: Product Issue', contents='Have you tried resetting your password?'),
        ],
        context='Customer support thread',
    )

    # Get prompt parts
    user_content = sig.to_user_content()
    assert len(user_content) == 1
    assert user_content[0] == snapshot("""\
Analyze emails for key information and sentiment.

List of email messages to analyze. Look for sentiment and key topics.
<emails>
  <Email>
    <subject>Product Issue</subject>
    <contents>I'm having trouble with the login feature.</contents>
  </Email>
  <Email>
    <subject>Re: Product Issue</subject>
    <contents>Have you tried resetting your password?</contents>
  </Email>
</emails>

Additional context about the email thread or conversation.
Context: Customer support thread\
""")


def test_gepa_components():
    """Test extracting GEPA components from a signature."""
    components = EmailAnalysis.get_gepa_components()
    assert components == snapshot(
        {
            'signature:EmailAnalysis:instructions': 'Analyze emails for key information and sentiment.',
            'signature:EmailAnalysis:emails:desc': 'List of email messages to analyze. Look for sentiment and key topics.',
            'signature:EmailAnalysis:context:desc': 'Additional context about the email thread or conversation.',
        }
    )


def test_apply_candidate():
    """Test applying a GEPA candidate to optimize the signature."""
    # Create a candidate with optimized text
    candidate = {
        'signature:EmailAnalysis:instructions': 'Extract actionable insights from customer emails.',
        'signature:EmailAnalysis:emails:desc': 'Customer emails requiring detailed analysis.',
        'signature:EmailAnalysis:context:desc': 'Background information to inform the analysis.',
    }

    # Create an instance
    sig = EmailAnalysis(
        emails=[Email(subject='Test', contents='Test email')],
        context='Test context',
    )

    # Get prompt with the optimized candidate
    user_content = sig.to_user_content(candidate=candidate)
    assert len(user_content) == 1
    assert user_content[0] == snapshot("""\
Extract actionable insights from customer emails.

Customer emails requiring detailed analysis.
<emails>
  <Email>
    <subject>Test</subject>
    <contents>Test email</contents>
  </Email>
</emails>

Background information to inform the analysis.
Context: Test context\
""")


def test_signature_with_context_manager():
    """Test using the context manager to temporarily apply candidates."""
    from pydantic_ai_gepa.signature import apply_candidate_to_signature

    # Save original instructions
    original_instructions = EmailAnalysis.__doc__

    # Create a candidate
    candidate = {
        'signature:EmailAnalysis:instructions': 'Optimized instructions for email analysis.',
    }

    # Apply temporarily
    with apply_candidate_to_signature(EmailAnalysis, candidate):
        assert EmailAnalysis.__doc__ == 'Optimized instructions for email analysis.'

    # Should be restored
    assert EmailAnalysis.__doc__ == original_instructions


def test_signature_without_explicit_field_description():
    """Test that fields without descriptions get default ones."""

    class SimpleSignature(Signature):
        """A simple signature for testing."""

        # This field doesn't have a description
        text: str
        # This one does
        number: int = Field(description='A number to process')

    components = SimpleSignature.get_gepa_components()
    assert components == snapshot(
        {
            'signature:SimpleSignature:instructions': 'A simple signature for testing.',
            'signature:SimpleSignature:text:desc': 'The text input',
            'signature:SimpleSignature:number:desc': 'A number to process',
        }
    )


def test_multiple_signatures():
    """Test working with multiple signature classes."""
    from pydantic_ai_gepa.components import extract_seed_candidate_with_signatures

    class SummarySignature(Signature):
        """Summarize the given text."""

        text: str = Field(description='Text to summarize')
        max_length: int = Field(description='Maximum summary length')

    # Extract components from multiple signatures
    candidate = extract_seed_candidate_with_signatures(signatures=[EmailAnalysis, SummarySignature])
    assert candidate == snapshot(
        {
            'signature:EmailAnalysis:instructions': 'Analyze emails for key information and sentiment.',
            'signature:EmailAnalysis:emails:desc': 'List of email messages to analyze. Look for sentiment and key topics.',
            'signature:EmailAnalysis:context:desc': 'Additional context about the email thread or conversation.',
            'signature:SummarySignature:instructions': 'Summarize the given text.',
            'signature:SummarySignature:text:desc': 'Text to summarize',
            'signature:SummarySignature:max_length:desc': 'Maximum summary length',
        }
    )
