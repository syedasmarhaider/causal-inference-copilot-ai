from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence, cast

from pydantic import BaseModel, ConfigDict, field_validator

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState
from python.implementation.workflows.utils.utils import json_sanitize


class CompileProtocolPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    protocol: Optional[ProtocolSpec] = None
    compile_error: Optional[str] = None
    compile_issues: Optional[List[Dict[str, Any]]] = None
    user_message: Optional[str] = None

    @field_validator("compile_error", "user_message", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        raise TypeError("compile_error/user_message must be str|null")

    @field_validator("compile_issues", mode="before")
    @classmethod
    def _validate_issues(cls, v: Any) -> Optional[List[Dict[str, Any]]]:
        if v is None:
            return None
        if not isinstance(v, list):
            raise TypeError("compile_issues must be list[dict]|null")
        if any(not isinstance(x, dict) for x in v): # pyright: ignore[reportUnknownVariableType]
            raise TypeError("compile_issues must be list[dict]|null")
        return cast(List[Dict[str, Any]], v)

    @field_validator("protocol", mode="before")
    @classmethod
    def _validate_protocol(cls, v: Any) -> Optional[ProtocolSpec]:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise TypeError("protocol must be object|null")
        # ProtocolSpec is a TypedDict-style object in your code; keep as-is.
        return cast(ProtocolSpec, v)


# =============================================================================
# State wrapper (payload-only storage; derives interface fields)
# =============================================================================
class CompileProtocolState(State):
    NAME: ClassVar[str] = "COMPILE_PROTOCOL"

    def __init__(self, payload: CompileProtocolPayloadModel) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        # Preserve your existing semantics:
        # DONE only when protocol exists and no compile error; otherwise ABORTED.
        if self.payload.protocol is not None and self.payload.compile_error is None:
            return "DONE"
        return "ABORTED"

    @property
    def message(self) -> str:
        if self.payload.user_message is None:
            raise ValueError("CompileProtocolState message is required but missing. State must have user message. Dont call this property if this is not runned in the node context where user_message is guaranteed to be set.")
        return self.payload.user_message

    @property
    def error(self) -> Optional[str]:
        return self.payload.compile_error

    @property
    def needs_action(self) -> ACTION:
        return "NONE"

    def required_states_keys(self) -> Sequence[str]:
        return (LoadDatasetState.NAME, ProtocolDiscussionState.NAME)

    def to_json_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = self.payload.model_dump(mode="json")
        if self.payload.protocol is not None:
            out["protocol"] = json_sanitize(cast(Any, self.payload.protocol))
        if self.payload.compile_issues is not None:
            out["compile_issues"] = json_sanitize(cast(Any, self.payload.compile_issues))

        return out

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "CompileProtocolState":
        model = CompileProtocolPayloadModel.model_validate(payload)
        return cls(model)
    
    @classmethod
    def init_empty(cls) -> "CompileProtocolState":
        return cls(CompileProtocolPayloadModel())