from __future__ import annotations

from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.workflows.node_state import NodeState
from python.implementation.workflows.utils.utils import uuid_from_any


class ModelTrainPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    training_signature: str | None = None
    trained_model_id: UUID | None = None
    training_warnings: list[str] = Field(default_factory=list)
    assistant_message: str | None = None
    error_message: str | None = None

    @field_validator("trained_model_id", mode="before")
    @classmethod
    def _parse_uuid(cls, value: Any) -> UUID | None:
        return uuid_from_any(value)

    @field_validator(
        "assistant_message",
        "error_message",
        "training_signature",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        raise TypeError("text fields must be str|null")

    def reset_for_signature(self, *, training_signature: str) -> ModelTrainPayloadModel:
        return self.model_copy(
            update={
                "training_signature": training_signature,
                "trained_model_id": None,
                "training_warnings": [],
                "assistant_message": None,
                "error_message": None,
            }
        )


class ModelTrainState(NodeState):
    NAME: ClassVar[str] = "MODEL_TRAIN"

    def __init__(self, payload: ModelTrainPayloadModel | None = None) -> None:
        self.payload = payload or ModelTrainPayloadModel()

    def name(self) -> str:
        return self.NAME

    def clear_state(self) -> None:
        self.payload = ModelTrainPayloadModel()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ModelTrainState:
        return cls(ModelTrainPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> ModelTrainState:
        return cls()


__all__ = [
    "ModelTrainPayloadModel",
    "ModelTrainState",
]
