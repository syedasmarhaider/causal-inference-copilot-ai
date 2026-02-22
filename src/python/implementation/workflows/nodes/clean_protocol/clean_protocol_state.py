from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.utils.utils import uuid_from_any


# =============================================================================
# Payload (strict, payload-only)
# =============================================================================
class CleanProtocolPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    clean_dataset_id: Optional[UUID] = None
    cleaning_error: Optional[str] = None
    user_message: Optional[str] = None

    @field_validator("cleaning_error", "user_message", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        raise TypeError("cleaning_error/user_message must be str|null")

    @field_validator("clean_dataset_id", mode="before")
    @classmethod
    def _parse_uuid(cls, v: Any) -> Optional[UUID]:
        # Use your project-wide parser (handles UUID|str|None safely)
        return uuid_from_any(v)


# =============================================================================
# State wrapper (payload-only, NO embedded name in JSON)
# =============================================================================
class CleanProtocolState(State):
    NAME: ClassVar[str] = "CLEAN_PROTOCOL"

    def __init__(self, payload: CleanProtocolPayloadModel) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        if self.payload.cleaning_error is not None:
            return "ABORTED"
        if self.payload.clean_dataset_id is not None:
            return "DONE"
        return "PENDING"

    @property
    def message(self) -> str:
        if self.payload.user_message is None:
            raise ValueError("CleanProtocolState message is required but missing. State must have user message. Dont call this property if this is not runned in the node context where user_message is guaranteed to be set.")
        return self.payload.user_message

    @property
    def error(self) -> Optional[str]:
        return self.payload.cleaning_error

    @property
    def needs_action(self) -> ACTION:
        return "NONE"

    def required_states_keys(self) -> Sequence[str]:
        return (LoadDatasetState.NAME, CompileProtocolState.NAME)

    def to_json_dict(self) -> Dict[str, Any]:
        return self.payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "CleanProtocolState":
        model = CleanProtocolPayloadModel.model_validate(payload)
        return cls(model)
    
    @classmethod
    def init_empty(cls) -> "CleanProtocolState":
        return cls(CleanProtocolPayloadModel())