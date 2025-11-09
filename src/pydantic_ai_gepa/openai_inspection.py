"""Wrapper model that exposes OpenAI payloads prior to dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, cast

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, ModelRequestParameters, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModelSettings
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
            "messages": self.provider_messages,
        }
        if self.model_settings:
            payload["settings"] = dict(self.model_settings)
        payload.update(self.extra)
        return payload


class OpenAIInspectionAborted(RuntimeError):
    """Raised when a request is intercepted for inspection."""

    def __init__(self, snapshot: OpenAIInspectionSnapshot):
        super().__init__("OpenAI request intercepted for inspection.")
        self.snapshot = snapshot


class _SupportsOpenAIIntrospection(Protocol):
    """Subset of OpenAIChatModel surface relied on for inspection."""

    model_name: str
    system: str

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        ...

    async def _map_messages(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
    ) -> list[Any]:
        ...


class InspectingOpenAIModel(WrapperModel):
    """Wrap an OpenAI model and surface payloads via exceptions."""

    def __init__(self, wrapped: Model) -> None:
        super().__init__(wrapped)
        self._openai = self._ensure_openai_model(self.wrapped)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ):
        snapshot = await self._build_snapshot(messages, model_settings, model_request_parameters)
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
        provider_messages = await self._openai._map_messages(messages, prepared_params)
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
    def _ensure_openai_model(model: Model) -> _SupportsOpenAIIntrospection:
        required = ("prepare_request", "_map_messages", "model_name", "system")
        missing = [attr for attr in required if not hasattr(model, attr)]
        if missing:
            names = ", ".join(missing)
            raise TypeError(
                "InspectingOpenAIModel requires an OpenAI-compatible model "
                f"implementing: {names}"
            )
        return cast(_SupportsOpenAIIntrospection, model)


__all__ = [
    "InspectingOpenAIModel",
    "OpenAIInspectionAborted",
    "OpenAIInspectionSnapshot",
]
