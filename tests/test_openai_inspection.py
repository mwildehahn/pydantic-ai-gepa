from __future__ import annotations

import pytest

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters, ModelSettings

from pydantic_ai_gepa.openai_inspection import (
    InspectingOpenAIModel,
    OpenAIInspectionAborted,
)


class _FakeOpenAIModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.requests = 0
        self.mapped_payloads: list[list[dict[str, str]]] = []

    @property
    def model_name(self) -> str:
        return "openai:gpt-test"

    @property
    def system(self) -> str:
        return "openai"

    async def request(
        self,
        messages: list[ModelRequest],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ):
        self.requests += 1
        return ModelResponse(parts=[TextPart("ok")])

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ):
        merged: ModelSettings = {}
        if model_settings:
            merged.update(model_settings)
        merged.setdefault("temperature", 0.0)
        return merged, model_request_parameters

    async def _map_messages(
        self,
        messages: list[ModelRequest],
        model_request_parameters: ModelRequestParameters,
    ):
        converted = [
            {"role": "user", "content": part.content}
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        self.mapped_payloads.append(converted)
        return converted


def _make_messages() -> list[ModelRequest]:
    return [
        ModelRequest(
            parts=[UserPromptPart("2+2?")],
            instructions="calc it",
        )
    ]


@pytest.mark.asyncio
async def test_inspection_exception_includes_snapshot() -> None:
    fake = _FakeOpenAIModel()
    wrapper = InspectingOpenAIModel(fake)

    with pytest.raises(OpenAIInspectionAborted) as excinfo:
        await wrapper.request(_make_messages(), {"temperature": 0.5}, ModelRequestParameters())

    assert fake.requests == 0
    snapshot = excinfo.value.snapshot
    assert snapshot.provider_messages[0]["content"] == "2+2?"
    assert snapshot.model_settings["temperature"] == 0.5
