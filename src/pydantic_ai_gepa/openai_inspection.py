"""Wrapper model that exposes OpenAI payloads prior to dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence, cast

from openai import NOT_GIVEN as OPENAI_NOT_GIVEN

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, ModelRequestParameters, ModelSettings
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.models.wrapper import WrapperModel


@dataclass(slots=True)
class OpenAIInspectionSnapshot:
    """Captured request data for OpenAI models."""

    model_name: str
    provider_name: str
    original_messages: Sequence[ModelMessage]
    provider_messages: list[Any]
    model_settings: OpenAIChatModelSettings
    request_parameters: ModelRequestParameters
    extra: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """Return a dict mirroring the OpenAI client payload."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": _json_safe(self.provider_messages),
        }
        if self.model_settings:
            payload["settings"] = _json_safe(dict(self.model_settings))
        if self.extra:
            payload["metadata"] = _json_safe(self.extra)
        return payload


class OpenAIInspectionAborted(RuntimeError):
    """Raised when a request is intercepted for inspection."""

    def __init__(self, snapshot: OpenAIInspectionSnapshot):
        super().__init__("OpenAI request intercepted for inspection.")
        self.snapshot = snapshot


class InspectingOpenAIModel(WrapperModel):
    """Wrap an OpenAI model and surface payloads via exceptions."""

    def __init__(self, wrapped: Model) -> None:
        super().__init__(wrapped)
        self._openai: OpenAIChatModel | OpenAIResponsesModel = (
            self._ensure_openai_model(self.wrapped)
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ):
        snapshot = await self._build_snapshot(
            messages, model_settings, model_request_parameters
        )
        raise OpenAIInspectionAborted(snapshot)

    async def _build_snapshot(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> OpenAIInspectionSnapshot:
        prepared_settings, prepared_params = self._openai.prepare_request(
            model_settings,
            model_request_parameters,
        )
        provider_messages = await self._map_provider_messages(
            messages, prepared_settings, prepared_params
        )
        settings_dict = cast(OpenAIChatModelSettings, prepared_settings or {})
        return OpenAIInspectionSnapshot(
            model_name=self._openai.model_name,
            provider_name=self._openai.system,
            original_messages=list(messages),
            provider_messages=provider_messages,
            model_settings=settings_dict,
            request_parameters=prepared_params,
        )

    @staticmethod
    def _ensure_openai_model(model: Model) -> OpenAIChatModel | OpenAIResponsesModel:
        if isinstance(model, (OpenAIChatModel, OpenAIResponsesModel)):
            return cast(OpenAIChatModel | OpenAIResponsesModel, model)
        raise TypeError(
            "InspectingOpenAIModel requires an OpenAIChatModel or OpenAIResponsesModel."
        )

    async def _map_provider_messages(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> list[Any]:
        if isinstance(self._openai, OpenAIResponsesModel):
            settings = cast(OpenAIResponsesModelSettings, model_settings or {})
            return await self._openai._map_messages(
                messages, settings, model_request_parameters
            )

        return await self._openai._map_messages(messages, model_request_parameters)


__all__ = [
    "InspectingOpenAIModel",
    "OpenAIInspectionAborted",
    "OpenAIInspectionSnapshot",
]


def _json_safe(value: Any) -> Any:
    if value is OPENAI_NOT_GIVEN:
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
