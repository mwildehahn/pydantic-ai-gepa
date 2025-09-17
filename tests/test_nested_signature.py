from __future__ import annotations

from inline_snapshot import snapshot
from pydantic import BaseModel, Field
from pydantic_ai_gepa import Signature


class Address(BaseModel):
    """Address information."""

    street: str = Field(description='Street address')
    city: str = Field(description='City name')
    zip_code: str = Field(description='ZIP or postal code')


class CustomerQuery(Signature):
    """Process customer inquiries with full context."""

    customer_name: str = Field(description='Full name of the customer')
    query: str = Field(description="The customer's question or issue")
    billing_address: Address = Field(description="Customer's billing address")
    shipping_address: Address | None = Field(default=None, description='Optional shipping address')


class SimpleQuery(Signature):
    """Process simple queries."""

    question: str = Field(description='The question to answer')


def test_signature_component_extraction_with_nested_models():
    """Test that nested models don't cause key collisions."""
    # Extract components from CustomerQuery
    customer_components = CustomerQuery.get_gepa_components()

    # Should have class-specific keys
    assert 'signature:CustomerQuery:instructions' in customer_components
    assert 'signature:CustomerQuery:customer_name:desc' in customer_components
    assert 'signature:CustomerQuery:billing_address:desc' in customer_components

    # Extract components from SimpleQuery
    simple_components = SimpleQuery.get_gepa_components()

    # Should also have class-specific keys
    assert 'signature:SimpleQuery:instructions' in simple_components
    assert 'signature:SimpleQuery:question:desc' in simple_components

    # Verify no key collisions - each signature has unique keys
    assert len(set(customer_components.keys()) & set(simple_components.keys())) == 0


def test_apply_candidate_with_class_specific_keys():
    """Test that candidates are applied correctly with class-specific keys."""
    # Create a candidate with optimized text
    candidate = {
        'signature:CustomerQuery:instructions': 'OPTIMIZED: Handle customer issues professionally',
        'signature:CustomerQuery:customer_name:desc': 'OPTIMIZED: Customer full legal name',
        'signature:SimpleQuery:instructions': 'OPTIMIZED: Answer concisely',
        'signature:SimpleQuery:question:desc': 'OPTIMIZED: The user question',
    }

    # Apply to CustomerQuery
    original_customer_doc = CustomerQuery.__doc__
    CustomerQuery.apply_candidate(candidate)

    # Check that CustomerQuery was updated with its specific values
    assert CustomerQuery.__doc__ == 'OPTIMIZED: Handle customer issues professionally'
    assert CustomerQuery.model_fields['customer_name'].description == 'OPTIMIZED: Customer full legal name'

    # Apply to SimpleQuery
    original_simple_doc = SimpleQuery.__doc__
    SimpleQuery.apply_candidate(candidate)

    # Check that SimpleQuery was updated with its specific values
    assert SimpleQuery.__doc__ == 'OPTIMIZED: Answer concisely'
    assert SimpleQuery.model_fields['question'].description == 'OPTIMIZED: The user question'

    # Restore original state
    CustomerQuery.__doc__ = original_customer_doc
    SimpleQuery.__doc__ = original_simple_doc


def test_to_user_content_with_nested_models():
    """Test that nested models are formatted correctly in user content."""
    # Create an instance with nested models
    query = CustomerQuery(
        customer_name='John Doe',
        query='Where is my order?',
        billing_address=Address(street='123 Main St', city='Springfield', zip_code='12345'),
    )

    # Convert to user content
    content = query.to_user_content()
    assert content == snapshot(
        [
            """\
Customer Name: John Doe

Query: Where is my order?

Billing Address (JSON)
```json
{
  "street": "123 Main St",
  "city": "Springfield",
  "zip_code": "12345"
}
```\
"""
        ]
    )

    # Test with optimized candidate
    candidate = {
        'signature:CustomerQuery:instructions': 'Help the customer quickly',
    }

    system_instructions = query.to_system_instructions(candidate=candidate)
    assert system_instructions == snapshot("""\
Help the customer quickly

Inputs

- `customer_name` (str): OPTIMIZED: Customer full legal name
- `query` (str): The customer's question or issue
- `billing_address` (Address): Customer's billing address
- `shipping_address` (UnionType[Address, NoneType]): Optional shipping address

Schemas

Each <Address> element contains:
- <street>: Street address
- <city>: City name
- <zip_code>: ZIP or postal code

Each <Address> element contains:
- <street>: Street address
- <city>: City name
- <zip_code>: ZIP or postal code\
""")

    content_optimized = query.to_user_content()
    assert content_optimized == snapshot(
        [
            """\
Customer Name: John Doe

Query: Where is my order?

Billing Address (JSON)
```json
{
  "street": "123 Main St",
  "city": "Springfield",
  "zip_code": "12345"
}
```\
"""
        ]
    )
