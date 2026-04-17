from __future__ import annotations

from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from python.domain.workflows.node_state import NodeState
from python.implementation.workflows.utils.utils import uuid_from_any

ProtocolDiscussionPhase = Literal["DISCUSSING", "REVIEW_READY", "CONFIRMED"]


class ProtocolDiscussionPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_id: UUID | None = None
    discussion: str = ""
    phase: ProtocolDiscussionPhase = "DISCUSSING"
    pending_dataset_change_request: str | None = None
    assistant_message: str | None = None

    @field_validator("dataset_id", mode="before")
    @classmethod
    def _parse_dataset_id(cls, value: Any) -> UUID | None:
        return uuid_from_any(value)

    @field_validator(
        "discussion",
        "pending_dataset_change_request",
        "assistant_message",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        raise TypeError("text fields must be str|null")


class ProtocolDiscussionState(NodeState):
    NAME: ClassVar[str] = "PROTOCOL_DISCUSSION"

    def __init__(self, payload: ProtocolDiscussionPayloadModel | None = None) -> None:
        self.payload = payload or ProtocolDiscussionPayloadModel()

    def name(self) -> str:
        return self.NAME

    def clear_state(self) -> None:
        self.payload = ProtocolDiscussionPayloadModel()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ProtocolDiscussionState:
        return cls(ProtocolDiscussionPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> ProtocolDiscussionState:
        return cls()


__all__ = [
    "ProtocolDiscussionPayloadModel",
    "ProtocolDiscussionPhase",
    "ProtocolDiscussionState",
]
