from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.errors import NodeExecutionError
from python.domain.models.models import ChatMessage
from python.domain.workflows.state import State, Status
from python.implementation.workflows.nodes.model_selection.model_selection_deps import (
    ModelSelectionDeps,
)


class ModelRecommendationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    estimator_fqcn: str
    display_label: str
    best_when: str
    why: str
    tradeoffs: str | None = None


class ConfirmedModelSelectionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    selected_model: str | None = None
    reasoning: str | None = None


class ModelSelectionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    recommendations: list[ModelRecommendationModel] = Field(default_factory=lambda: [])
    confirmed_model_selection: ConfirmedModelSelectionPayload | None = None
    assistant_message: str | None = None
    error_message: str | None = None

    @field_validator("assistant_message", "error_message", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        raise TypeError("assistant_message/error_message must be str|null")


class ModelSelectionState(State):
    NAME: ClassVar[str] = "MODEL_SELECTION"

    def __init__(self, payload: ModelSelectionPayload) -> None:
        self.payload = payload

    def name(self) -> str:
        return self.NAME

    def status(self) -> Status:
        if self.payload.error_message is not None:
            return "ABORTED"
        cms = self.payload.confirmed_model_selection
        if cms is not None and cms.selected_model is not None:
            return "DONE"
        return "PENDING"

    def set_status_freez(self) -> None:
        return None

    def set_status_pending(self) -> None:
        if self.payload.error_message is not None:
            self.payload.error_message = None

    def messages(self) -> Sequence[ChatMessage]:
        if self.payload.assistant_message:
            return [ChatMessage(role="assistant", content=self.payload.assistant_message)]
        return [
            ChatMessage(
                role="assistant",
                content=(
                    "I will now recommend causal models that fit the confirmed protocol, "
                    "the reviewed warnings, and the available column types."
                ),
            )
        ]

    def error(self) -> NodeExecutionError | None:
        if self.payload.error_message is None:
            return None
        return NodeExecutionError(state_name=self.NAME, error=self.payload.error_message)

    def pre_required_states_names(self) -> Sequence[str]:
        return ModelSelectionDeps.pre_required_states_names()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ModelSelectionState:
        return cls(ModelSelectionPayload.model_validate(payload))

    @classmethod
    def init_empty(cls) -> ModelSelectionState:
        return cls(ModelSelectionPayload())

