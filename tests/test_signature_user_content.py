from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field
from pydantic_ai.messages import ImageUrl, UserContent

from pydantic_ai_gepa.signature import Signature

from inline_snapshot import snapshot


class GallerySignature(Signature):
    """Signature mixing multimodal content with text."""

    gallery: Sequence[UserContent] = Field(description="Gallery content to inspect")
    notes: str = Field(description="Natural-language instructions")


def test_user_content_with_multimodal_resources() -> None:
    first = ImageUrl(url="https://example.com/a.png")
    second = ImageUrl(url="https://example.com/b.png")

    sig = GallerySignature(
        gallery=[
            first,
            "Provide a comparison of the screenshots.",
            second,
            "Highlight any mismatched UI states.",
        ],
        notes="Focus your answer on layout differences.",
    )

    user_content = sig.to_user_content()
    images = user_content[:2]
    assert all([isinstance(image, ImageUrl) for image in images])
    text_content = user_content[2]
    assert text_content == snapshot("""\
<gallery>
  <item><image ref="image1"/></item>
  <item>Provide a comparison of the screenshots.</item>
  <item><image ref="image2"/></item>
  <item>Highlight any mismatched UI states.</item>
</gallery>

<notes>Focus your answer on layout differences.</notes>\
""")


class ReferenceModel(BaseModel):
    attachment: ImageUrl
    remark: str


class NestedSignature(Signature):
    reference: ReferenceModel = Field(description="Structured reference data")
    repeated: Sequence[UserContent] = Field(
        description="Repeated attachments for reuse"
    )


def test_duplicate_attachment_reuses_placeholder() -> None:
    screenshot = ImageUrl(url="https://example.com/reused.png")

    sig = NestedSignature(
        reference=ReferenceModel(attachment=screenshot, remark="Primary capture."),
        repeated=["Revisit the earlier capture here:", screenshot],
    )

    system_instructions = sig.to_system_instructions()
    assert system_instructions == snapshot("""\
Inputs

- `<reference>` (ReferenceModel): Structured reference data
- `<repeated>` (Sequence[UnionType[str, ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent]]): Repeated attachments for reuse. Attached resources are referenced with placeholders like <audio ref="audioN"/>, <binary ref="binaryN"/>, <document ref="documentN"/>, <image ref="imageN"/>, <video ref="videoN"/> where N matches the order the resource was attached.

Schemas

Each <ReferenceModel> element contains:
- <attachment>: The attachment field. Attached resources are referenced with placeholders like <image ref="imageN"/> where N matches the order the resource was attached.
- <remark>: The remark field\
""")

    user_content = sig.to_user_content()
    image = user_content[0]
    assert isinstance(image, ImageUrl)
    text_content = user_content[1]
    assert text_content == snapshot("""\
<reference>
  <attachment><image ref="image1"/></attachment>
  <remark>Primary capture.</remark>
</reference>

<repeated>
  <item>Revisit the earlier capture here:</item>
  <item><image ref="image1"/></item>
</repeated>\
""")
