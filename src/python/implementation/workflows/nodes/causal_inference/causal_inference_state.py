from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.errors import NodeExecutionError
from python.domain.models.models import ArtifactRef, ChatMessage
from python.domain.workflows.state import Action, State, Status
from python.implementation.workflows.nodes.causal_inference.causal_inference_deps import (
    CausalInferenceDeps,
)


class CausalInferencePayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ate_result_raw_json_str: str | None = None
    latest_cate_result_raw_json_str: str | None = None
    latest_cate_request_summary: str | None = None
    assistant_message: str | None = None
    system_message: str | None = None
    message_artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    error_message: str | None = None

    @field_validator(
        "ate_result_raw_json_str",
        "latest_cate_result_raw_json_str",
        "latest_cate_request_summary",
        "assistant_message",
        "system_message",
        "error_message",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        raise TypeError("text fields must be str|null")


class CausalInferenceState(State):
    NAME: ClassVar[str] = "CAUSAL_INFERENCE"

    def __init__(self, payload: CausalInferencePayloadModel) -> None:
        self.payload = payload

    def name(self) -> str:
        return self.NAME

    def status(self) -> Status:
        return "PENDING"

    def action(self) -> Action:
        return "NEEDS_INPUT"

    def set_status_freez(self) -> None:
        return None

    def set_status_pending(self) -> None:
        self.payload.error_message = None

    def messages(self) -> Sequence[ChatMessage]:
        messages: list[ChatMessage] = []
        if self.payload.system_message:
            messages.append(ChatMessage(role="system", content=self.payload.system_message))
        if self.payload.assistant_message:
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=self.payload.assistant_message,
                    artifact_refs=list(self.payload.message_artifact_refs) or None,
                )
            )
        if messages:
            return messages
        return [
            ChatMessage(
                role="assistant",
                content=(
                    "I will now compute and explain the causal effect estimates from the "
                    "trained model, including subgroup effects when clinically requested."
                ),
            )
        ]

    def error(self) -> NodeExecutionError | None:
        return None

    def pre_required_states_names(self) -> Sequence[str]:
        return CausalInferenceDeps.pre_required_states_names()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> CausalInferenceState:
        return cls(CausalInferencePayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> CausalInferenceState:
        return cls(CausalInferencePayloadModel())
