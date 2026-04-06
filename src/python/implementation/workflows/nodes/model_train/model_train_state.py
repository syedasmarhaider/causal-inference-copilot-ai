from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.errors import NodeExecutionError
from python.domain.models.models import ChatMessage
from python.domain.workflows.state import Action, State, Status
from python.implementation.workflows.nodes.model_train.model_train_deps import ModelTrainDeps
from python.implementation.workflows.utils.utils import uuid_from_any


class ModelTrainPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_id: UUID | None = None
    training_signature: str | None = None
    trained_model_id: UUID | None = None
    training_warnings: list[str] = Field(default_factory=list)
    assistant_message: str | None = None
    error_message: str | None = None

    @field_validator("dataset_id", "trained_model_id", mode="before")
    @classmethod
    def _parse_uuid(cls, value: Any) -> UUID | None:
        return uuid_from_any(value)

    @field_validator("assistant_message", "error_message", "training_signature", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        raise TypeError("text fields must be str|null")


class ModelTrainState(State):
    NAME: ClassVar[str] = "MODEL_TRAIN"

    def __init__(self, payload: ModelTrainPayloadModel) -> None:
        self.payload = payload

    def name(self) -> str:
        return self.NAME

    def status(self) -> Status:
        if self.payload.error_message is not None:
            return "ABORTED"
        if self.payload.trained_model_id is not None:
            return "DONE"
        return "PENDING"

    def action(self) -> Action:
        return "NONE"

    def set_status_freez(self) -> None:
        return None

    def set_status_pending(self) -> None:
        self.payload.error_message = None

    def messages(self) -> Sequence[ChatMessage]:
        if self.payload.assistant_message:
            return [ChatMessage(role="assistant", content=self.payload.assistant_message)]
        return [
            ChatMessage(
                role="assistant",
                content=(
                    "I will now train the confirmed causal model using the active cleaned "
                    "dataset and the confirmed preprocessing plan."
                ),
            )
        ]

    def error(self) -> NodeExecutionError | None:
        if self.payload.error_message is None:
            return None
        return NodeExecutionError(state_name=self.NAME, error=self.payload.error_message)

    def pre_required_states_names(self) -> Sequence[str]:
        return ModelTrainDeps.pre_required_states_names()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ModelTrainState:
        return cls(ModelTrainPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> ModelTrainState:
        return cls(ModelTrainPayloadModel())
