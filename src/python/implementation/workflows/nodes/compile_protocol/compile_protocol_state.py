from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence, cast

from pydantic import BaseModel, ConfigDict, field_validator

from python.domain.workflows.state import ACTION, State, StateMessage, Status
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_deps import CompileProtocolDeps
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.data_processing.data_processing_tool import ExclusionRulesModel


class CompileProtocolPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    causal_specs: Optional[CausalSpec] = None
    exclusion: Optional[ExclusionRulesModel] = None
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
        # DONE only when causal_specs exists and no compile error; otherwise ABORTED.
        if self.payload.causal_specs is not None and self.payload.compile_error is None:
            return "DONE"
        return "ABORTED"

    @property
    def message(self) -> StateMessage:
        if self.payload.user_message is None:
            raise ValueError("CompileProtocolState message is required but missing. State must have user message. Dont call this property if this is not runned in the node context where user_message is guaranteed to be set.")
        return StateMessage(txt_message=self.payload.user_message)

    @property
    def error(self) -> Optional[str]:
        return self.payload.compile_error

    @property
    def needs_action(self) -> ACTION:
        return "NONE"

    def pre_required_states_names(self) -> Sequence[str]:
        return CompileProtocolDeps.pre_required_states_names()

    def to_json_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = self.payload.model_dump(mode="json")
        return out

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "CompileProtocolState":
        model = CompileProtocolPayloadModel.model_validate(payload)
        return cls(model)
    
    @classmethod
    def init_empty(cls) -> "CompileProtocolState":
        return cls(CompileProtocolPayloadModel())