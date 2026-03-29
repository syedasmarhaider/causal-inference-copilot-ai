from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from python.domain.models.errors import NodeExecutionError
from python.domain.workflows.state import State, StateMessage, Status
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_deps import (
    ProtocolDiscussionDeps,
)


class ProtocolDiscussionPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    discussion: str = ""
    readiness: Literal["READY", "PENDING", "ABORT"] = "PENDING"
    node_message: str | None = None
    error_message: str | None = None

    @field_validator("discussion", mode="before")
    @classmethod
    def _ensure_discussion_str(cls, v: Any) -> str:
        if v is None:
            return ""
        if not isinstance(v, str):
            raise TypeError("discussion must be str")
        return v

    @field_validator("node_message", "error_message", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        raise TypeError("message fields must be str|null")


# =============================================================================
# State wrapper (payload-only storage)
# =============================================================================
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
            raise ValueError("ProtocolDiscussionState message is required but missing. State must have node message. Dont call this property if this is not runned in the node context where node_message is guaranteed to be set.")
        action = "NEEDS_INPUT" if self.payload.readiness == "PENDING" else "NONE"
        return StateMessage(txt_message=self.payload.node_message, action=action)

    @property
    def error(self) -> NodeExecutionError | None:
        if self.payload.error_message is not None:
            return NodeExecutionError(state_name=self.NAME, error=self.payload.error_message)
        return None

    def pre_required_states_names(self) -> Sequence[str]:
        return  ProtocolDiscussionDeps.pre_required_states_names()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ProtocolDiscussionState:
        model = ProtocolDiscussionPayloadModel.model_validate(payload)
        return cls(model)
    
    @classmethod
    def init_empty(cls) -> ProtocolDiscussionState:
        return cls(ProtocolDiscussionPayloadModel())
