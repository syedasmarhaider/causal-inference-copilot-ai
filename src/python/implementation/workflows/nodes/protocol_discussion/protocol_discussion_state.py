from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from python.domain.models.models import ChatMessage
from python.domain.workflows.state import Action, State, Status
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    initial_user_message,
)

from python.implementation.workflows.utils.utils import uuid_from_any

ProtocolDiscussionPhase = Literal["DISCUSSING", "REVIEW_READY", "CONFIRMED"]


class ProtocolDiscussionPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_id: UUID | None = None
    discussion: str = ""
    phase: ProtocolDiscussionPhase = "DISCUSSING"
    pending_dataset_change_request: str | None = None
    assistant_message: str | None = None
    system_message: str | None = None

    @field_validator("dataset_id", mode="before")
    @classmethod
    def _parse_dataset_id(cls, value: Any) -> UUID | None:
        return uuid_from_any(value)

    @field_validator(
        "discussion",
        "pending_dataset_change_request",
        "assistant_message",
        "system_message",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        raise TypeError("text fields must be str|null")


class ProtocolDiscussionState(State):
    NAME: ClassVar[str] = "PROTOCOL_DISCUSSION"

    def __init__(self, payload: ProtocolDiscussionPayloadModel) -> None:
        self.payload = payload

    def name(self) -> str:
        return self.NAME

    def status(self) -> Status:
        if self.payload.phase == "CONFIRMED":
            return "DONE"
        return "PENDING"

    def action(self) -> Action:
        if self.status() == "DONE":
            return "NONE"
        return "NEEDS_INPUT"

    def set_status_pending(self) -> None:
        if self.payload.phase == "CONFIRMED":
            self.payload.phase = "DISCUSSING"
            self.payload.pending_dataset_change_request = None

    def messages(self) -> Sequence[ChatMessage]:
        messages: list[ChatMessage] = []
        if self.payload.system_message:
            messages.append(ChatMessage(role="system", content=self.payload.system_message))
        if self.payload.assistant_message:
            messages.append(ChatMessage(role="assistant", content=self.payload.assistant_message))
        if messages:
            return messages
        return [
            ChatMessage(
                role="assistant",
                content=initial_user_message(),
            )
        ]

    def error(self) -> None:
        return None

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ProtocolDiscussionState:
        return cls(ProtocolDiscussionPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> ProtocolDiscussionState:
        return cls(ProtocolDiscussionPayloadModel())


__all__ = [
    "ProtocolDiscussionPayloadModel",
    "ProtocolDiscussionState",
]
