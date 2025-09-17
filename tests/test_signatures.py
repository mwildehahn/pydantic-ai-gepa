"""Tests for Signature integration with agents and GEPA."""

from __future__ import annotations

from typing import Any

from inline_snapshot import snapshot
from pydantic import BaseModel, Field
from pydantic_ai_gepa import DataInst, PydanticAIGEPAAdapter, Signature
from pydantic_ai_gepa.components import extract_seed_candidate_with_signature

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


class EmailSender(BaseModel):
    """An email sender."""

    name: str = Field(description='The name of the sender.')
    address: str = Field(description='The email address of the sender.')


class EmailHeader(BaseModel):
    """An email header."""

    subject: str = Field(description='The subject of the email. Pay specific attention to this.')
    sender: EmailSender


# Define data models
class Email(BaseModel):
    """An email message."""

    header: EmailHeader = Field(description='The header of the email.')
    contents: str


class SupportResponse(BaseModel):
    """A customer support response."""

    priority: str  # "high", "medium", "low"
    category: str  # "technical", "billing", "general"
    suggested_response: str
    needs_escalation: bool


# Define a Signature for structured input
class EmailSupportSignature(Signature):
    """Analyze customer support emails and generate appropriate responses."""

    emails: list[Email] = Field(
        description='Customer emails requiring support. Analyze for urgency, technical issues, and sentiment.'
    )

    previous_interactions: str | None = Field(
        description='Summary of previous interactions with this customer, if available.'
    )

    company_policies: str = Field(description='Relevant company policies and guidelines for customer support.')


def test_signature_to_prompt_parts():
    """Test converting a signature instance to prompt parts."""
    # Create test data
    emails = [
        Email(
            header=EmailHeader(
                sender=EmailSender(
                    name='John Doe',
                    address='john@example.com',
                ),
                subject='Login not working',
            ),
            contents="I've been trying to log in for the past hour but keep getting an error. This is urgent!",
        ),
        Email(
            header=EmailHeader(
                sender=EmailSender(
                    name='John Doe',
                    address='john@example.com',
                ),
                subject='Re: Login not working',
            ),
            contents='I tried resetting my password but the reset email never arrived.',
        ),
    ]

    # Create a signature instance
    sig = EmailSupportSignature(
        emails=emails,
        previous_interactions='Customer contacted us last week about slow performance issues.',
        company_policies='Respond to urgent issues within 2 hours. Escalate authentication problems to tech team.',
    )

    # Convert to prompt parts
    user_content = sig.to_user_content()

    assert len(user_content) == 1
    content = user_content[0]
    assert content == snapshot("""\
<emails>
  <Email>
    <header>
      <subject>Login not working</subject>
      <sender>
        <name>John Doe</name>
        <address>john@example.com</address>
      </sender>
    </header>
    <contents>I've been trying to log in for the past hour but keep getting an error. This is urgent!</contents>
  </Email>
  <Email>
    <header>
      <subject>Re: Login not working</subject>
      <sender>
        <name>John Doe</name>
        <address>john@example.com</address>
      </sender>
    </header>
    <contents>I tried resetting my password but the reset email never arrived.</contents>
  </Email>
</emails>

<previous_interactions>Customer contacted us last week about slow performance issues.</previous_interactions>

<company_policies>Respond to urgent issues within 2 hours. Escalate authentication problems to tech team.</company_policies>\
""")


def test_signature_with_optimized_candidate():
    """Test applying an optimized GEPA candidate to a signature."""
    # Create a signature instance
    sig = EmailSupportSignature(
        emails=[
            Email(
                header=EmailHeader(
                    sender=EmailSender(
                        name='Test User',
                        address='test@example.com',
                    ),
                    subject='Test Issue',
                ),
                contents='This is a test email.',
            )
        ],
        previous_interactions='No previous interactions.',
        company_policies='Standard policies apply.',
    )

    # Create an optimized candidate
    optimized_candidate = {
        'signature:EmailSupportSignature:instructions': (
            'You are an expert support agent. Identify critical issues immediately.'
        ),
        'signature:EmailSupportSignature:emails:desc': (
            'URGENT: Customer emails showing frustration. Extract key problems.'
        ),
        'signature:EmailSupportSignature:previous_interactions:desc': ('Historical context - look for patterns.'),
        'signature:EmailSupportSignature:company_policies:desc': ('Critical policies that must be followed.'),
    }

    # Convert with optimized prompts
    user_content = sig.to_user_content()
    system_instructions = sig.to_system_instructions(candidate=optimized_candidate)
    assert system_instructions == snapshot("""\
You are an expert support agent. Identify critical issues immediately.

<emails>: URGENT: Customer emails showing frustration. Extract key problems.

Each <Email> element contains:
- <header>: The header of the email.
  - <subject>: The subject of the email. Pay specific attention to this.
  - <sender>: The sender field
    - <name>: The name of the sender.
    - <address>: The email address of the sender.
- <contents>: The contents field

<previous_interactions>: Historical context - look for patterns.
<company_policies>: Critical policies that must be followed.\
""")

    content = user_content[0]
    assert content == snapshot("""\
<emails>
  <Email>
    <header>
      <subject>Test Issue</subject>
      <sender>
        <name>Test User</name>
        <address>test@example.com</address>
      </sender>
    </header>
    <contents>This is a test email.</contents>
  </Email>
</emails>

<previous_interactions>No previous interactions.</previous_interactions>

<company_policies>Standard policies apply.</company_policies>\
""")


def test_extract_signature_components():
    """Test extracting GEPA components from a signature."""
    components = EmailSupportSignature.get_gepa_components()
    assert components == snapshot(
        {
            'signature:EmailSupportSignature:instructions': 'Analyze customer support emails and generate appropriate responses.',
            'signature:EmailSupportSignature:emails:desc': 'Customer emails requiring support. Analyze for urgency, technical issues, and sentiment.',
            'signature:EmailSupportSignature:previous_interactions:desc': 'Summary of previous interactions with this customer, if available.',
            'signature:EmailSupportSignature:company_policies:desc': 'Relevant company policies and guidelines for customer support.',
        }
    )


def test_signature_with_agent():
    """Test using a signature with a TestModel agent."""
    # Create an agent with TestModel that returns structured output
    test_output = SupportResponse(
        priority='high',
        category='technical',
        suggested_response='I can help you with the login issue.',
        needs_escalation=True,
    )

    agent = Agent(
        TestModel(custom_output_args=test_output.model_dump()),
        output_type=SupportResponse,
    )

    # Create a signature instance
    sig = EmailSupportSignature(
        emails=[
            Email(
                header=EmailHeader(
                    sender=EmailSender(
                        name='Test User',
                        address='user@example.com',
                    ),
                    subject='Critical Issue',
                ),
                contents='System is down!',
            )
        ],
        previous_interactions=None,
        company_policies='Escalate all critical issues immediately.',
    )

    # Convert to prompt and run agent
    user_content = sig.to_user_content()
    prompt_content = user_content[0]
    assert prompt_content == snapshot("""\
<emails>
  <Email>
    <header>
      <subject>Critical Issue</subject>
      <sender>
        <name>Test User</name>
        <address>user@example.com</address>
      </sender>
    </header>
    <contents>System is down!</contents>
  </Email>
</emails>

<company_policies>Escalate all critical issues immediately.</company_policies>\
""")

    result = agent.run_sync(user_content)
    response = result.output
    assert response == snapshot(
        SupportResponse(
            priority='high',
            category='technical',
            suggested_response='I can help you with the login issue.',
            needs_escalation=True,
        )
    )


def test_gepa_adapter_with_signatures():
    """Test creating a GEPA adapter with signatures."""
    # Create an agent with TestModel
    agent = Agent(
        TestModel(custom_output_text='Test response'),
        output_type=SupportResponse,
    )

    # Define a simple metric
    def support_metric(data_inst: DataInst, output: Any) -> tuple[float, str]:
        return 0.8, 'Good response'

    # Create adapter with signatures
    adapter = PydanticAIGEPAAdapter(
        agent=agent,
        metric=support_metric,
        signature_class=EmailSupportSignature,
    )

    assert adapter.signature_class == EmailSupportSignature
    assert adapter.agent == agent
    assert adapter.metric == support_metric


def test_extract_seed_candidate_with_signatures():
    """Test extracting initial components from both agent and signatures."""
    # Create a real agent with TestModel
    agent = Agent(
        TestModel(),
        instructions='Be helpful and professional.',
        system_prompt=['System prompt 1', 'System prompt 2'],
    )

    # Extract components from both agent and signature
    candidate = extract_seed_candidate_with_signature(
        agent=agent,
        signature_class=EmailSupportSignature,
    )

    # Should have components from both agent and signature
    assert candidate == snapshot(
        {
            'instructions': 'Be helpful and professional.',
            'system_prompt:0': 'System prompt 1',
            'system_prompt:1': 'System prompt 2',
            'signature:EmailSupportSignature:instructions': 'Analyze customer support emails and generate appropriate responses.',
            'signature:EmailSupportSignature:emails:desc': 'Customer emails requiring support. Analyze for urgency, technical issues, and sentiment.',
            'signature:EmailSupportSignature:previous_interactions:desc': 'Summary of previous interactions with this customer, if available.',
            'signature:EmailSupportSignature:company_policies:desc': 'Relevant company policies and guidelines for customer support.',
        }
    )


def test_signature_with_none_field():
    """Test signature with optional fields set to None."""
    sig = EmailSupportSignature(
        emails=[
            Email(
                header=EmailHeader(
                    sender=EmailSender(
                        name='Test User',
                        address='test@example.com',
                    ),
                    subject='Test',
                ),
                contents='Test content',
            )
        ],
        previous_interactions=None,  # Optional field set to None
        company_policies='Default policies',
    )

    user_content = sig.to_user_content()
    content = user_content[0]
    assert content == snapshot("""\
<emails>
  <Email>
    <header>
      <subject>Test</subject>
      <sender>
        <name>Test User</name>
        <address>test@example.com</address>
      </sender>
    </header>
    <contents>Test content</contents>
  </Email>
</emails>

<company_policies>Default policies</company_policies>\
""")


def test_signature_field_without_description():
    """Test that fields without explicit descriptions get default ones."""

    class MinimalSignature(Signature):
        """A minimal signature."""

        # Field without description
        input_text: str
        # Field with description
        config: dict[str, Any] = Field(description='Configuration settings')

    components = MinimalSignature.get_gepa_components()
    assert components == snapshot(
        {
            'signature:MinimalSignature:instructions': 'A minimal signature.',
            'signature:MinimalSignature:input_text:desc': 'The input_text input',
            'signature:MinimalSignature:config:desc': 'Configuration settings',
        }
    )
