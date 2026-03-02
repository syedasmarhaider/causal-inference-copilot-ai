from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.workflows.state import ACTION, State, StateMessage, Status
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_deps import ValidateCleanProtocolDeps
from python.implementation.workflows.utils.validation import ValidationIssueModel


class ValidateCleanProtocolPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    issues: List[ValidationIssueModel] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    validation_error: Optional[str] = None
    user_message: Optional[str] = None

    @field_validator("validation_error", "user_message", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        raise TypeError("validation_error/user_message must be str|null")


class ValidateCleanProtocolState(State):
    NAME: ClassVar[str] = "VALIDATE_CLEAN_PROTOCOL"

    def __init__(self, payload: ValidateCleanProtocolPayloadModel) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        # ABORTED if any FAIL issue; else DONE
        if any(i.severity == "FAIL" for i in self.payload.issues):
            return "ABORTED"
        return "DONE"

    @property
    def message(self) -> StateMessage:
        if self.payload.user_message is None:
            raise ValueError("ValidateCleanProtocolState message is required but missing. State must have user message. Dont call this property if this is not runned in the node context where user_message is guaranteed to be set.")
        return StateMessage(txt_message=self.payload.user_message)

    @property
    def error(self) -> Optional[str]:
        return self.payload.validation_error

    @property
    def needs_action(self) -> ACTION:
        return "NONE"

    def pre_required_states_names(self) -> Sequence[str]:
        return ValidateCleanProtocolDeps.pre_required_states_names()

    def to_json_dict(self) -> Dict[str, Any]:
        return self.payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "ValidateCleanProtocolState":
        model = ValidateCleanProtocolPayloadModel.model_validate(payload)
        return cls(model)
    
    @classmethod
    def init_empty(cls) -> "ValidateCleanProtocolState":
        return cls(ValidateCleanProtocolPayloadModel())