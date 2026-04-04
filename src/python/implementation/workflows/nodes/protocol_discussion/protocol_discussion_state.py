from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from python.domain.models.errors import NodeExecutionError
from python.domain.workflows.state import State, StateMessage, Status
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_deps import (
    ProtocolDiscussionDeps,
)
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)
from python.implementation.workflows.utils.utils import uuid_from_any

Readiness = Literal["PENDING", "READY", "ABORT"]


class ProtocolDiscussionPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_id: UUID | None = None
    dataset_summary: DatasetSummaryModel | None = None
    discussion: str = ""
    readiness: Readiness = "PENDING"
    node_message: str | None = None
    error_message: str | None = None

    @field_validator("dataset_id", mode="before")
    @classmethod
    def _parse_dataset_id(cls, value: Any) -> UUID | None:
        return uuid_from_any(value)

    @field_validator("discussion", "node_message", "error_message", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped
        raise TypeError("discussion/node_message/error_message must be str|null")


class ProtocolDiscussionState(State):
    NAME: ClassVar[str] = "PROTOCOL_DISCUSSION"

    def __init__(self, payload: ProtocolDiscussionPayloadModel) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        if self.payload.readiness == "ABORT" or self.payload.error_message is not None:
            return "ABORTED"
        if self.payload.readiness == "READY":
            return "DONE"
        return "PENDING"

    @property
    def message(self) -> StateMessage:
        if self.payload.node_message is None:
            raise ValueError(
                "ProtocolDiscussionState message is required but missing. "
                "State must have node message in node context."
            )
        action = "NEEDS_INPUT" if self.payload.readiness == "PENDING" else "NONE"
        return StateMessage(txt_message=self.payload.node_message, action=action)

    @property
    def error(self) -> NodeExecutionError | None:
        if self.payload.error_message is not None:
            return NodeExecutionError(state_name=self.NAME, error=self.payload.error_message)
        return None

    def freeze_status(self) -> None:
        return None

    def pre_required_states_names(self) -> Sequence[str]:
        return ProtocolDiscussionDeps.pre_required_states_names()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ProtocolDiscussionState:
        model = ProtocolDiscussionPayloadModel.model_validate(payload)
        return cls(model)

    @classmethod
    def init_empty(cls) -> ProtocolDiscussionState:
        return cls(ProtocolDiscussionPayloadModel())


__all__ = [
    "ProtocolDiscussionPayloadModel",
    "ProtocolDiscussionState",
]
