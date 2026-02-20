from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.utils.validation import ValidationIssueModel

ValidationSeverity = Literal["WARN", "FAIL"]
ValidationStatus = Literal["PASS", "WARN", "FAIL"]



class CleanProtocolValidationPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    issues: List[ValidationIssueModel] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]
    validation_error: Optional[str] = None
    user_message: Optional[str] = None


@dataclass(frozen=True)
class ValidateCleanProtocolState(State):
    NAME: ClassVar[str] = "VALIDATE_CLEAN_PROTOCOL"
    payload: Optional[CleanProtocolValidationPayloadModel] = None

    @property
    def status(self) -> Status:
        if self.payload is not None:
            if any(issue.severity == "FAIL" for issue in self.payload.issues):
                return "ABORTED"
        return "DONE"
    
    @property
    def message(self) -> Optional[str]:
        return self.payload.user_message if self.payload else None

    @property
    def error(self) -> Optional[str]:
        return self.payload.validation_error if self.payload else None

    @property
    def needs_action(self) -> ACTION:
        return "NONE"

    def required_states_keys(self) -> Sequence[str]:
        return [CompileProtocolState.NAME, CleanProtocolState.NAME]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "payload": self.payload.model_dump(mode="json") if self.payload else None,
        }

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "ValidateCleanProtocolState":
        raw = payload.get("payload")
        model = CleanProtocolValidationPayloadModel.model_validate(raw) if isinstance(raw, dict) else None
        return cls(
                payload=model,
       )
