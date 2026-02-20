from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState


class ProtocolDiscussionPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    discussion: str = ""
    node_message: Optional[str] = None
    error_message: Optional[str] = None
    action: ACTION = "NONE"
    node_status: Status = "PENDING"

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
    def _empty_str_to_none(cls, v: Any) -> Optional[str]:
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
        return self.payload.node_status

    @property
    def message(self) -> Optional[str]:
        return self.payload.node_message

    @property
    def error(self) -> Optional[str]:
        return self.payload.error_message

    @property
    def needs_action(self) -> ACTION:
        return self.payload.action

    def required_states_keys(self) -> Sequence[str]:
        return (LoadDatasetState.NAME,)

    def to_json_dict(self) -> Dict[str, Any]:
        return self.payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "ProtocolDiscussionState":
        model = ProtocolDiscussionPayloadModel.model_validate(payload)
        return cls(model)